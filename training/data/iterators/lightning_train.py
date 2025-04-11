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
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from args import TrainArgs

from arrow_iterator import ArrowFileIterator

from bytelatent.model.blt import ByteLatentTransformer
# make sure to import the BLT here
from optim import build_optimizer
# from transformer import LMTransformer


class EntropyEvaluationCallback(Callback):
    def __init__(self, entropy_dataset_path, eval_every_n_steps=200, batch_size=32):
        """
        Callback to evaluate the model on the entropy dataset during training
        
        Args:
            entropy_dataset_path: Path to the entropy dataset
            eval_every_n_steps: Evaluate every N training steps
            batch_size: Batch size for evaluation
        """
        super().__init__()
        self.entropy_dataset_path = entropy_dataset_path
        self.eval_every_n_steps = eval_every_n_steps
        self.batch_size = batch_size
        self._last_global_step = -1
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Run evaluation periodically during training"""
        global_step = trainer.global_step
        

        if global_step % self.eval_every_n_steps == 0 and global_step > 0 and global_step != self._last_global_step:
            self._last_global_step = global_step
            
            print(f"\n=== Evaluating on entropy dataset at step {global_step} ===")
            

            if not hasattr(self, 'data_iterator') or self.data_iterator is None:
                self.data_iterator = self.load_entropy_dataset()
            

            training = pl_module.training
            pl_module.eval()
            
            # Run evaluation
            metrics = self.evaluate_model(pl_module, trainer.accelerator.device)
            

            for metric_name, metric_value in metrics.items():
                trainer.logger.log_metrics({f"entropy_{metric_name}": metric_value}, step=global_step)
            

            print(f"Entropy dataset evaluation results at step {global_step}: {metrics}")
            
            if training:
                pl_module.train()
    
    def load_entropy_dataset(self):
        """Load the entropy dataset"""
        return ArrowFileIterator(
            file_path=self.entropy_dataset_path,
            batch_size=self.batch_size,
            shuffle=False,
            repeat=False
        )
    
    def evaluate_model(self, pl_module, device):
        """Evaluate the model on the entropy dataset"""
        model = pl_module.model
        model.eval()
        
        total_loss = 0.0
        total_examples = 0
        total_tokens = 0
        total_bpc = 0.0
        

        max_batches = 500 ## random value, pls update accordingly arnav
        batch_count = 0
        
        with torch.no_grad():
            for batch in self.data_iterator:
                # Move batch to device
                batch_x = batch.x.to(device)
                batch_y = batch.y.to(device)
                
                # Get additional inputs
                batch_patch_lengths = batch.patch_lengths.to(device) if batch.patch_lengths is not None else None
                mask = batch.mask.to(device) if batch.mask is not None else None
                ngram_ids = batch.ngram_ids.to(device) if batch.ngram_ids is not None else None
                
                # Forward pass
                predictions = model(batch_x, patch_lengths=batch_patch_lengths, ngram_ids=ngram_ids)
                
                # Compute loss
                loss, tok_loss = compute_loss(predictions, batch_y, mask, scale=1.0)
                
                # Compute tokens count
                if mask is None:
                    num_tokens = batch_y.numel()
                else:
                    num_tokens = mask.sum().item()
                
                # Compute bits per character
                bpc = loss.item() * np.log2(np.e)
                
                # Accumulate metrics
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                total_examples += batch_x.size(0)
                total_bpc += bpc * num_tokens
                
                batch_count += 1
                if batch_count >= max_batches:
                    break
        
        # Reset the iterator for next evaluation
        self.data_iterator = self.load_entropy_dataset()
        
        # Compute final metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        avg_bpc = total_bpc / total_tokens if total_tokens > 0 else float('inf')
        
        return {
            "loss": avg_loss,
            "bpc": avg_bpc,
            "examples": total_examples,
            "tokens": total_tokens,
        }


class DummyModel(torch.nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x, patch_lengths=None, ngram_ids=None):
        # For simplicity, we ignore patch_lengths and ngram_ids.
        return self.linear(x)
    
    def init_weights(self):
        # Initialize weights (e.g., Xavier initialization)
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)



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
        # self.save_hyperparameters(asdict(args))
        self.n_bytes = 0

        self.save_hyperparameters(args.model_dump())
        
        # Build tokenizer (fallback to SimpleTokenizer if no build() method is provided)
        self.tokenizer = (args.data.tokenizer_args.build() 
                          if hasattr(args.data.tokenizer_args, "build") 
                          else SimpleTokenizer())
        
        # Initialize model: either an entropy model or the main model.
        # if args.train_entropy_model:
        #     assert args.entropy_model is not None, "Entropy model must be provided."
        #     self.model = LMTransformer(args.entropy_model)
        #     self.model_args = args.entropy_model
        # else:
        assert args.model is not None, "Model configuration must be provided."
        self.model = ByteLatentTransformer(args.model)  # Or whatever values make sense for your test
        self.model_args = args.model

        ## change this so we call the BLT here and not the dummy model
        
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
class ByteLatentDataModule(pl.LightningDataModule):
    def __init__(self, args: TrainArgs, test_mode: bool = False):
        super().__init__()
        self.args = args
        self.data_loader = None
        self.test_mode = test_mode
    
    def setup(self, stage=None):
        if self.test_mode:
            # For testing, create a dummy data loader instead of loading from files
            from collections import namedtuple
            Batch = namedtuple('Batch', ['x', 'y', 'patch_lengths', 'mask', 'ngram_ids'])
            
            class DummyIterator:
                def __iter__(self):
                    return self.create_iter()
                
                def create_iter(self):
                    from collections import namedtuple
                    Batch = namedtuple('Batch', ['x', 'y', 'patch_lengths', 'mask', 'ngram_ids'])
                    
                    # Create a generator function
                    def generator():
                        for _ in range(10):  # Generate 10 dummy batches
                            yield Batch(
                                x=torch.randn(2, 10, 128),  # batch_size=2, seq_len=10, dim=128
                                y=torch.randint(0, 4, (2, 10)),  # batch_size=2, seq_len=10
                                patch_lengths=torch.ones(2, 10) * 5,  # batch_size=2, seq_len=10
                                mask=torch.ones(2, 10),  # batch_size=2, seq_len=10
                                ngram_ids=None
                            )
                    
                    return generator()
            
            self.data_loader = DummyIterator()
        else:
            # Regular data loading logic using Lightning's distributed parameters
            rank = self.trainer.global_rank if self.trainer is not None else 0
            world_size = self.trainer.world_size if self.trainer is not None else 1
            self.data_loader = self.args.data.build_from_rank(rank=rank, world_size=world_size)
    
    def train_dataloader(self):
        return self.data_loader

###############################################
# Training Function and Main Entrypoint
###############################################

def train(args: TrainArgs, test_mode = False, entropy_dataset_path = None):


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize the Lightning module and datamodule.
    model = ByteLatentLightningModule(args)
    
    # Use test_mode parameter
    data_module = ByteLatentDataModule(args, test_mode=test_mode)
    

    callbacks = []
    

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint.path or os.path.join(args.dump_dir, "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=args.checkpoint.dump.keep if args.checkpoint.dump.keep > 0 else -1,
        every_n_train_steps=args.checkpoint.dump.every,
        save_on_train_epoch_end=False,
    )
    callbacks.append(checkpoint_callback)
    

    if entropy_dataset_path:
        entropy_eval_callback = EntropyEvaluationCallback(
            entropy_dataset_path=entropy_dataset_path,
            eval_every_n_steps=200, 
            batch_size=args.data.batch_size  # Use the same batch size as training
        )
        callbacks.append(entropy_eval_callback)
    
   

    trainer = pl.Trainer(
        max_steps=args.steps,
        reload_dataloaders_every_epoch=False,
        strategy="ddp",
        accelerator="auto",
        devices=4,
        callbacks=callbacks,
        gradient_clip_val=args.optim.clip,
        accumulate_grad_batches=args.grad_acc_steps,
        precision="bf16",
    )
    

    trainer.fit(model, datamodule=data_module)
    
    gc.collect()

if __name__ == "__main__":
    # FLOPs,Patch size,Tokens,Encoder parameters,Global transformer parameters,
    # Decoder parameters,Global transformer dimension,Encoder/Decoder dimension,
    # Global transformer layers,Decoder layers
    #s,2,8977160953.457802,789760,100769792,3948800,1024,256,8,5 
    train_args = TrainArgs()
    train(train_args)
