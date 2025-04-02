import os
import math
import random
import gc
from dataclasses import asdict
from typing import Any, Optional, Union, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from data.iterators.arrow_iterator import ArrowFileIterator
from args import TrainArgs

from bytelatent.model.blt import ByteLatentTransformer, ByteLatentTransformerArgs
from bytelatent.optim import build_optimizer
from bytelatent.transformer import LMTransformer
from bytelatent.eval import EVAL_FOLDER_NAME, launch_eval



###############################################
# Simple Tokenizer Implementation
###############################################

class SimpleTokenizer:
    def __init__(self):
        self.n_words = 4  # e.g., for ACTG in a genomic context
        self.unk_token = 78  # unknown token id, e.g., for "N"
    
    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))
    
    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode("utf-8")

###############################################
# Helper Functions
###############################################

def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Flatten a nested dictionary for logging purposes."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def to_py_num(num: Union[int, float, torch.Tensor, np.ndarray]) -> Union[int, float]:
    """Convert a tensor or ndarray to a native Python number."""
    if isinstance(num, (torch.Tensor, np.ndarray)):
        return num.item()
    else:
        return num

def compute_loss(predictions, targets, mask, scale):
    """Compute cross-entropy loss with optional masking."""
    tok_loss = scale * F.cross_entropy(
        predictions.flatten(0, 1), targets.flatten(0, 1), reduction="none"
    )
    if mask is None:
        loss = tok_loss.mean()
    else:
        mask = mask.flatten(0, 1)
        tok_loss = tok_loss * mask
        loss = tok_loss.sum() / (mask.sum() + 1e-6)
    return loss, tok_loss

###############################################
# Lightning Module
###############################################

class ByteLatentLightningModule(pl.LightningModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        self.save_hyperparameters(asdict(args))
        
        # Build tokenizer (fallback to SimpleTokenizer if no build() method is provided)
        self.tokenizer = (args.data.tokenizer_args.build() 
                          if hasattr(args.data.tokenizer_args, "build") 
                          else SimpleTokenizer())
        
        # Initialize model: either an entropy model or the main model.
        if args.train_entropy_model:
            assert args.entropy_model is not None, "Entropy model must be provided."
            self.model = LMTransformer(args.entropy_model)
            self.model_args = args.entropy_model
        else:
            assert args.model is not None, "Model configuration must be provided."
            self.model = ByteLatentTransformer(args.model)
            self.model_args = args.model
        
        # Initialize model weights
        self.model.init_weights()
        

        
    def forward(self, x, patch_lengths=None, ngram_ids=None):
        if self.args.train_entropy_model:
            return self.model(x)
        else:
            return self.model(x, patch_lengths=patch_lengths, ngram_ids=ngram_ids)
    
    def training_step(self, batch, batch_idx):
        batch_x = batch.x
        batch_y = batch.y
        batch_patch_lengths = batch.patch_lengths  # may be None
        mask = batch.mask  # may be None
        ngram_ids = batch.ngram_ids  # may be None
        
        # Update byte count for metrics based on tokenizer type
        if self.args.data.tokenizer_args.name in ["bytes", "blt"]:
            self.n_bytes += batch_y.numel() if mask is None else mask.sum().item()
        elif self.args.data.tokenizer_args.name in ["sp", "tiktoken"]:
            for example in batch.y:
                target_tokens = self.tokenizer.decode(example.tolist())
                self.n_bytes += (
                    len(target_tokens.encode("utf-8")) +
                    (example == self.tokenizer.eos_id).sum().item() +
                    (example == self.tokenizer.bos_id).sum().item()
                )
        else:
            raise ValueError(f"Unexpected tokenizer: {self.args.data.tokenizer_args.name}")
        
        # Forward pass and loss computation
        pred = self.forward(batch_x, batch_patch_lengths, ngram_ids)
        loss, tok_loss = compute_loss(pred, batch_y, mask, scale=1.0)
        
        
        
        return loss
    
    def configure_optimizers(self):
        optimizer, scheduler = build_optimizer(self.model, self.args.optim, self.args.steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }


###############################################
# DataModule and Iterator Wrapper (No IterableDataset)
###############################################

class ArrowIteratorWrapper:
    """
    A minimal wrapper that simply returns a fresh iterator from the arrow iterator each time.
    """
    def __init__(self, arrow_iterator: ArrowFileIterator):
        self.arrow_iterator = arrow_iterator

    def __iter__(self):
        # Each call returns a new iterator from the arrow iterator.
        return self.arrow_iterator.create_iter()

class ByteLatentDataModule(pl.LightningDataModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        self.data_loader = None
    
    def setup(self, stage=None):
        # Build the arrow iterator for single-process operation.
        self.data_loader = self.args.data.build_from_rank(rank=0, world_size=1)
    
    def train_dataloader(self):
        # Instead of wrapping with an IterableDataset, we wrap the arrow iterator in a minimal wrapper.
        return ArrowIteratorWrapper(self.data_loader)

###############################################
# Training Function and Main Entrypoint
###############################################

def train(args: TrainArgs):


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize the Lightning module and datamodule.
    model = ByteLatentLightningModule(args)
    data_module = ByteLatentDataModule(args)
    
    # Set up a checkpoint callback.
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint.path or os.path.join(args.dump_dir, "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=args.checkpoint.dump.keep if args.checkpoint.dump.keep > 0 else -1,
        every_n_train_steps=args.checkpoint.dump.every,
        save_on_train_epoch_end=False,
    )
    
   
    
    # Initialize the Lightning Trainer for a single device.
    trainer = pl.Trainer(
        max_steps=args.steps,
        accelerator="auto",  # Will use CPU if GPU is unavailable.
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=args.optim.clip,
        accumulate_grad_batches=args.grad_acc_steps,
        precision=32,
    )
    
    # Train the model.
    trainer.fit(model, datamodule=data_module)
 
    
    gc.collect()

def main():

    train_args = TrainArgs()
    train(train_args)

if __name__ == "__main__":
    main()
