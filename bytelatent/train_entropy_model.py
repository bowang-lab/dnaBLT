"""
Train a byte-level entropy model using a transformer architecture.

This script trains a transformer model to predict the next byte in a sequence,
which can be used as an entropy model for compression. It uses PyTorch Lightning
for training and supports distributed training, mixed precision, etc.

The model is trained on raw bytes (values 0-255) and uses causal self-attention
to predict the next byte in the sequence.
"""

import os
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.nn.attention.flex_attention import BlockMask
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR

import numpy as np
from datasets import load_dataset

from bytelatent.transformer import LMTransformer, LMTransformerArgs
from xformers.ops import AttentionBias


class DNAByteDataset(Dataset):
    """Dataset for byte-level processing."""

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        stage: str,
        split: str = "train",
    ):
        """Initialize dataset.

        Args:
            data_path: Path/name of HuggingFace dataset
            seq_length: Sequence length for model input
            stage: Stage of the dataset
            split: Dataset split to use
        """
        self.seq_length = seq_length
        self.dataset = load_dataset(data_path, stage)[split]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        bytes_data = self.dataset[idx]["text"].encode("utf-8")
        bytes_array = np.frombuffer(bytes_data, dtype=np.uint8)

        # Truncate or pad to the sequence length
        if len(bytes_array) > self.seq_length:
            bytes_array = bytes_array[: self.seq_length]
        else:
            bytes_array = np.pad(bytes_array, (0, self.seq_length - len(bytes_array)))

        return torch.from_numpy(bytes_array.copy()).long()


class DNAByteDataModule:
    """Data module for byte data."""

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        batch_size: int,
        num_workers: int,
    ):
        self.data_path = data_path
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = "stage1"):
        self.train_dataset = DNAByteDataset(
            self.data_path, self.seq_length, stage, "validation"
        )
        self.val_dataset = DNAByteDataset(
            self.data_path, self.seq_length, stage, "validation"
        )
        self.test_dataset = DNAByteDataset(
            self.data_path, self.seq_length, stage, "validation"
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class EntropyModelTrainer(pl.LightningModule):
    """PyTorch Lightning module for training the entropy model."""

    def __init__(self, args: argparse.Namespace):
        """Initialize the trainer.

        Args:
            args: Command line arguments containing model and training hyperparameters
        """
        super().__init__()
        self.save_hyperparameters(args)

        # Initialize the transformer model
        model_args = LMTransformerArgs(
            dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            max_seqlen=args.seq_length,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            vocab_size=256,  # byte-level, so vocab size is 256
            sliding_window=args.sliding_window,
            seed=args.seed,
            norm_eps=args.norm_eps,
            return_dict=True,
        )
        self.model = LMTransformer(model_args)

    def forward(
        self,
        tokens: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            tokens: Input tensor of shape [batch_size, seq_len]
            target: Optional target tensor for computing loss
            tok_idx: Optional token indices
            mask: Optional attention mask
            attn_impl: Attention implementation to use

        Returns:
            Model output logits or loss if target is provided
        """
        return self.model(tokens, target, tok_idx, mask, attn_impl)

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        """Common step logic for training and validation.

        Args:
            batch: Input tensor of shape [batch_size, seq_len]
            stage: Either 'train' or 'val'

        Returns:
            Loss value
        """
        target = torch.roll(batch, shifts=-1, dims=-1)
        target[:, -1] = 0

        outputs = self.forward(batch, target=target, attn_impl="sdpa")

        # Calculate and log entropy
        entropy = self.compute_entropy(outputs["logits"])
        self.log(f"{stage}_entropy", entropy.mean(), prog_bar=True, sync_dist=True)
        self.log(f"{stage}_loss", outputs["loss"], prog_bar=True, sync_dist=True)

        return outputs["loss"]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self._step(batch, "test")

    def configure_optimizers(
        self,
    ) -> Tuple[list[Any], list[dict[str, SequentialLR | str]]]:
        """Configure optimizers and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=n_warmup_steps,
        )
        decay = CosineAnnealingLR(
            optimizer,
            T_max=n_decay_steps,
            eta_min=self.hparams.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, decay],
            milestones=[n_warmup_steps],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy from logits."""
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy


def main(args: argparse.Namespace):
    """Main training function.

    Sets up and runs the training loop for the entropy model using PyTorch Lightning.

    Args:
        args: Command line arguments containing model and training hyperparameters
            including data paths, model architecture, optimization settings, etc.
    """
    pl.seed_everything(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("medium")

    model = EntropyModelTrainer(args)
    data_module = DNAByteDataModule(
        data_path=args.data_path,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_module.setup(stage=args.stage)
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="entropy-{epoch:02d}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = WandbLogger(project="entropy-model", name=args.run_name)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda",
        devices=args.devices,
        callbacks=callbacks,
        logger=logger,
        precision="bf16-mixed",
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.grad_accum,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=14)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=8192)
    parser.add_argument("--ffn_dim_multiplier", type=int, default=4)
    parser.add_argument("--sliding_window", type=int, default=512)
    parser.add_argument("--weight_tying", type=bool, default=True)
    parser.add_argument("--norm_eps", type=float, default=1e-5)

    # Training arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="ddp")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="entropy-model")
    parser.add_argument("--stage", type=str, default="stage1")

    args = parser.parse_args()
    main(args)
