import os
import math
import random
import gc
from dataclasses import asdict
from typing import Any, Optional, Union, List, Dict
from bytelatent.data.ngram_processor import NgramProcessor, parse_ngram_to_size
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

import wandb
from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from args import TrainArgs
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from arrow_iterator import ArrowFileIterator

from bytelatent.model.blt import ByteLatentTransformer
# make sure to import the BLT here
from optim import build_optimizer

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
        self.tokenizer = args.data.tokenizer_args.build() 
        
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
        
    def on_after_backward(self):
        optimizer = self.optimizers()
        # Get Lightning’s GradScaler if it exists (only in 16‑mixed)
        scaler = getattr(self.trainer, "scaler", None)
        if scaler is not None:
            # unscale exactly once before clipping
            scaler.unscale_(optimizer)

        # now clip the (unscaled) gradients of your model
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            max_norm=self.args.optim.clip
        )

    def forward(self, x, patch_lengths=None, ngram_ids=None):
        if self.args.train_entropy_model:
            return self.model(x)
        else:
            return self.model(x, patch_lengths=patch_lengths, ngram_ids=ngram_ids)
    
    def training_step(self, batch, batch_idx):
        # TODO: Profile code. Data is slow, but is it *that* slow? How much time spent on forward/backward pass, is it as expected?
        batch_x = batch.x
        batch_y = batch.y
        batch_patch_lengths = batch.patch_lengths  # may be None
        mask = batch.mask  # may be None
        ngram_ids = batch.ngram_ids  # may be None


        if self.args.model.encoder_enable_byte_ngrams and ngram_ids is None:
        # Ensure that you have a valid directory for the ngram tables in your args.
            if not hasattr(self.args.model, "encoder_ngram_table_dir") or self.args.model.encoder_ngram_table_dir is None:
                raise ValueError("encoder_ngram_table_dir must be provided in the model args if using ngram embeddings.")

        # Parse ngram sizes from your configuration string.
            ngram_to_size = parse_ngram_to_size(self.args.model.encoder_ngram_to_size_str)
        # Initialize the NgramProcessor using the table directory and ngram sizes.
            ngram_processor = NgramProcessor(
                ngram_table_dir=self.args.model.encoder_ngram_table_dir,
                ngram_to_size=ngram_to_size
        )
        # Convert batch_x to a numpy array.
            raw_tokens = batch_x.cpu().numpy()
        # Compute ngram IDs. The output is a list (one per ngram size).
            ngram_ids_list = ngram_processor.encode_token_ngrams(raw_tokens)
        # Stack them along a new axis to form a single numpy array.
            ngram_ids_np = np.stack(ngram_ids_list, axis=0)
        # Convert to torch tensor and move to the same device as batch_x.
         
            ngram_ids = torch.tensor(ngram_ids_np, dtype=torch.int64, device=batch_x.device)
            self.print(f"Computed ngram_ids with shape: {ngram_ids.shape}")


        # if batch_patch_lengths is not None and isinstance(batch_patch_lengths, np.ndarray):
        #      batch_patch_lengths = torch.tensor(batch_patch_lengths, device=batch_x.device) 
        # # Update byte count for metrics based on tokenizer type
        # elif self.args.data.tokenizer_args.name in ["sp", "tiktoken"]:
        #     for example in batch.y:
        #         target_tokens = self.tokenizer.decode(example.tolist())
        #         self.n_bytes += (
        #             len(target_tokens.encode("utf-8")) +
        #             (example == self.tokenizer.eos_id).sum().item() +
        #             (example == self.tokenizer.bos_id).sum().item()
        #         )
        # else:
        #     raise ValueError(f"Unexpected tokenizer: {self.args.data.tokenizer_args.name}")
        
        # Forward pass and loss computation
        pred = self.forward(batch_x, batch_patch_lengths, ngram_ids)
        loss, tok_loss = compute_loss(pred, batch_y, mask, scale=1.0)
        self.log("train_entropy_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log("train_perplexity", torch.exp(loss), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # FIXME: Validation step did not run in wandb log? Nor did checkpointing?
        batch_x = batch.x
        batch_y = batch.y
        batch_patch_lengths = batch.patch_lengths  # may be None
        mask = batch.mask  # may be None
        ngram_ids = batch.ngram_ids  # may be None
        # if batch_patch_lengths is not None and isinstance(batch_patch_lengths, np.ndarray):
        #      batch_patch_lengths = torch.tensor(batch_patch_lengths, device=batch_x.device)
        if self.args.model.encoder_enable_byte_ngrams and ngram_ids is None:
        # Ensure that you have a valid directory for the ngram tables in your args.
            if not hasattr(self.args.model, "encoder_ngram_table_dir") or self.args.model.encoder_ngram_table_dir is None:
                raise ValueError("encoder_ngram_table_dir must be provided in the model args if using ngram embeddings.")
        
        # Parse ngram sizes from your configuration string.
            ngram_to_size = parse_ngram_to_size(self.args.model.encoder_ngram_to_size_str)
        # Initialize the NgramProcessor using the table directory and ngram sizes.
            ngram_processor = NgramProcessor(
                ngram_table_dir=self.args.model.encoder_ngram_table_dir,
                ngram_to_size=ngram_to_size
            )   
        # Convert batch_x to a numpy array.
            raw_tokens = batch_x.cpu().numpy()
        # Compute ngram IDs. The output is a list (one per ngram size).
            ngram_ids_list = ngram_processor.encode_token_ngrams(raw_tokens)
        # Stack them along a new axis to form a single numpy array.
            ngram_ids_np = np.stack(ngram_ids_list, axis=0)
        # Convert to torch tensor and move to the same device as batch_x.
            ngram_ids = torch.tensor(ngram_ids_np, dtype=torch.int64, device=batch_x.device)
        # Optionally log that you computed ngram_ids.
            self.print(f"Computed ngram_ids with shape: {ngram_ids.shape}")




        predictions = self.forward(batch_x, batch_patch_lengths, ngram_ids)
        loss, _ = compute_loss(predictions, batch_y, mask, scale=1.0)

        # Might want to play around with logging some of this
        # if mask is None:
        #     num_tokens = batch_y.numel()
        # else:
        #     num_tokens = mask.sum().item()

        # Log validation metrics every step. Use prog_bar=True when on Chimera (won't work on UHN)
        self.log("val_entropy_loss", loss, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log("val_perplexity", torch.exp(loss), on_step=True, on_epoch=False, sync_dist=True)

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
        rank = self.trainer.global_rank if self.trainer is not None else 0
        world_size = self.trainer.world_size if self.trainer is not None else 1
        self.train_data_loader = self.args.data.build_from_rank(rank=rank, world_size=world_size, mode="train")
    
    def train_dataloader(self):
        return self.train_data_loader
    
    def val_dataloader(self):
        rank = self.trainer.global_rank if self.trainer is not None else 0
        world_size = self.trainer.world_size if self.trainer is not None else 1
        return self.args.data.build_from_rank(rank=rank, world_size=world_size, mode="validation")
    
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Override LightningModule to move all batch elements to the correct device.
        """
        batch_type = type(batch)
        # Move x
        if not torch.is_tensor(batch.x):
            x = torch.tensor(batch.x, device=device)
        else:
            x = batch.x.to(device)
        # Move y
        if not torch.is_tensor(batch.y):
            y = torch.tensor(batch.y, device=device)
        else:
            y = batch.y.to(device)
        # Move patch_lengths
        patch_lengths = batch.patch_lengths
        if not torch.is_tensor(patch_lengths):
            patch_lengths = torch.tensor(patch_lengths, device=device)
        else:
            patch_lengths = patch_lengths.to(device)
        # Move mask
        mask = batch.mask
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=device)
        else:
            mask = mask.to(device)
        # Move ngram_ids
        ngram_ids = batch.ngram_ids
        if ngram_ids is not None:
            if not torch.is_tensor(ngram_ids):
                ngram_ids = torch.tensor(ngram_ids, device=device)
            else:
                ngram_ids = ngram_ids.to(device)
        # Reconstruct batch with all tensors on the correct device
        return batch_type(x=x, y=y, patch_lengths=patch_lengths, mask=mask, ngram_ids=ngram_ids)


###############################################
# Training Function and Main Entrypoint
###############################################

def train(args: TrainArgs, test_mode = False):


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch._dynamo.config.optimize_ddp = False # we may want to remove flex attention to use optimize_ddp
    
    # Initialize the Lightning module and datamodule.
    model = ByteLatentLightningModule(args)
    
    # Use test_mode parameter
    data_module = ByteLatentDataModule(args, test_mode=test_mode)

    # Set up Weights & Biases logger
    wandb_logger = WandbLogger(project="byte-latent")

    checkpoint_callback = ModelCheckpoint(
        # dirpath=args.checkpoint.path or os.path.join(args.dump_dir, "checkpoints"),
        filename="{step}-{val_entropy_loss:.2f}",
        # save_top_k=args.checkpoint.dump.keep if args.checkpoint.dump.keep > 0 else -1,
        monitor="val_entropy_loss",
        mode="min",
        save_top_k=1,
        every_n_train_steps=args.checkpoint.dump.every,
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        max_steps=args.steps,
        strategy="ddp",
        accelerator="auto",
        devices=1,
        callbacks=checkpoint_callback,
        gradient_clip_val=None, # must be none for fused adam
        accumulate_grad_batches=args.grad_acc_steps,
        precision="bf16-mixed",
        val_check_interval=args.checkpoint.dump.every,
        logger=wandb_logger,
        enable_progress_bar=False,
        log_every_n_steps=50
    )
    

    trainer.fit(model, datamodule=data_module)
    
    gc.collect()

if __name__ == "__main__":
    """

        tokens=18_000_000_000,
        seq_len=8192,
        patch_size=2,
        hidden_state_g=512,
        layers_g=9,
        hidden_state_e=256,
        layers_e=1,
        window_e=512,
        hidden_state_d=256,
        layers_d=5,
        window_d=512,
        ratio_patchdim2bytedim=2,
        vocab=4,
        feed_forward_mult=2.5,
    
    vs.

        tokens=12_400_000_000,
        seq_len=8192,
        patch_size=1,
        hidden_state_g=448,
        layers_g=7,
        hidden_state_e=192,
        layers_e=1,
        window_e=512,
        hidden_state_d=192,
        layers_d=5,
        window_d=512,
        ratio_patchdim2bytedim=2,
        vocab=4,
        feed_forward_mult=2.5,

    """
    train_args = TrainArgs()
    train(train_args)
