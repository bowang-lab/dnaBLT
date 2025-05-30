import random
import gc
from typing import Any, Union, Dict
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from pytorch_lightning.loggers import WandbLogger

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import torch._dynamo

torch._dynamo.config.suppress_errors = True


from bytelatent.model.blt import ByteLatentTransformer

# make sure to import the BLT here
from optim import build_optimizer
from v_args import DataloaderArgs, TrainArgs, OptimArgs
from patching import PatcherArgs
from blt import ByteLatentTransformerArgs

###############################################
# Helper Functions
###############################################

class StatefulDataloaderCheckpoint(pl.Callback):
    """
    Gather the StatefulDataLoader.state_dict from each rank and save
    a dictionary {rank: state_dict} inside the checkpoint.
    """
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Save a full `StatefulDataLoader.state_dict()` for every rank so that the
        training job can be resumed bit‑for‑bit.  We use
        ``torch.distributed.all_gather_object`` to collect the state dictionaries
        from all ranks onto rank 0 and then stash them under the
        ``"dataloader_state"`` key in the checkpoint.
        """
        # Full state for *this* rank’s DataLoader
        local_state = trainer.datamodule.train_dataloader().state_dict()

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            # Gather the dictionaries onto rank 0
            world_size = dist.get_world_size()
            gathered_states = [None] * world_size
            dist.all_gather_object(gathered_states, local_state)

            if trainer.global_rank == 0:
                checkpoint["dataloader_state"] = {
                    rank: state for rank, state in enumerate(gathered_states)
                }
        else:
            # Single‑process training – just save our own state.
            checkpoint["dataloader_state"] = {0: local_state}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        rank_state = checkpoint.get("dataloader_state", {}).get(trainer.global_rank)
        if rank_state is not None:
            trainer.datamodule._loaded_train_dataloader_state = rank_state


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
class PackedBatchDataset(IterableDataset):
    """
    IterableDataset that simply *delegates* to the existing iterator
    chain (Arrow → Preprocess → Sequence → Packing).

    Each element yielded is already a fully-formed `Batch` coming from
    `PackingIterator`, so no extra collation is required.
    """

    def __init__(self, dl_args: DataloaderArgs, dataset_key: str):
        super().__init__()
        self.dl_args = dl_args
        self.dataset_key = dataset_key
        worker_info = get_worker_info()
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Ensure _iterator is created fresh or if it's None
        self._iterator = self.dl_args.build_from_rank(
            ddp_rank=local_rank,
            ddp_world_size=world_size,
            worker_id=worker_info.id if worker_info else 0,
            num_workers=worker_info.num_workers if worker_info else 1,
            mode=self.dataset_key,
        )
    
    def state_dict(self):
        if self.dataset_key == "train":
            return {"i": self._iterator.sequence_iterator.source_to_iterator['16b*']._src_iter.arrow_batch_iterator.current_batch_idx}
        else:
            return {"i": self._iterator.sequence_iterator.source_to_iterator['entropies_validation.arrow']._src_iter.arrow_batch_iterator.current_batch_idx}

    def load_state_dict(self, state_dict):
        if self.dataset_key == "train":
            # Load state for training dataset
            self._iterator.sequence_iterator.source_to_iterator['16b*']._src_iter.arrow_batch_iterator.current_batch_idx = state_dict["i"]
        else:
            self._iterator.sequence_iterator.source_to_iterator['entropies_validation.arrow']._src_iter.arrow_batch_iterator.current_batch_idx = state_dict["i"]

    def __iter__(self):
        return iter(self._iterator)



def to_device_async(batch, device):
    """
    Override LightningModule to move all batch elements to the correct device.
    """
    batch_type = type(batch)
    # Move x
    if not torch.is_tensor(batch.x):
        x = torch.tensor(batch.x, device=device)
    else:
        x = batch.x.to(device, non_blocking=True)
    # Move y
    if not torch.is_tensor(batch.y):
        y = torch.tensor(batch.y, device=device)
    else:
        y = batch.y.to(device, non_blocking=True)
    # Move patch_lengths
    patch_lengths = batch.patch_lengths
    if not torch.is_tensor(patch_lengths):
        patch_lengths = torch.tensor(patch_lengths, device=device)
    else:
        patch_lengths = patch_lengths.to(device, non_blocking=True)
    # Move mask
    mask = batch.mask
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, device=device)
    else:
        mask = mask.to(device, non_blocking=True)
    # Move ngram_ids
    ngram_ids = batch.ngram_ids
    if ngram_ids is not None:
        if not torch.is_tensor(ngram_ids):
            ngram_ids = torch.tensor(ngram_ids, device=device)
        else:
            ngram_ids = ngram_ids.to(device)
    # Reconstruct batch with all tensors on the correct device
    return batch_type(
        x=x, y=y, patch_lengths=patch_lengths, mask=mask, ngram_ids=ngram_ids
    )


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
        self.model = ByteLatentTransformer(
            args.model
        )  # Or whatever values make sense for your test
        self.model_args = args.model

        ## change this so we call the BLT here and not the dummy model

        # Initialize model weights
        self.model.init_weights()
        self._prefetch_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self._next_on_device = None
        self._first_batch = True

    def on_after_backward(self):
        optimizer = self.optimizers()
        # Get Lightning’s GradScaler if it exists (only in 16‑mixed)
        scaler = getattr(self.trainer, "scaler", None)
        if scaler is not None:
            # unscale exactly once before clipping
            scaler.unscale_(optimizer)

        # now clip the (unscaled) gradients of your model
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.args.optim.clip)

    def forward(self, x, patch_lengths=None, ngram_ids=None):
        if self.args.train_entropy_model:
            return self.model(x)
        else:
            return self.model(x, patch_lengths=patch_lengths, ngram_ids=ngram_ids)

    def training_step(self, batch_cpu, batch_idx):
        if self._first_batch:
            batch = to_device_async(batch_cpu, self.device)
            self._first_batch = False
        else:
            torch.cuda.current_stream().wait_stream(self._prefetch_stream)
            batch = self._next_on_device

        with torch.cuda.stream(self._prefetch_stream):
            self._next_on_device = to_device_async(batch_cpu, self.device)

        # Forward pass and loss computation
        pred = self.forward(batch.x, batch.patch_lengths, None)
        loss, tok_loss = compute_loss(pred, batch.y, batch.mask, scale=1.0)
        self.log(
            "train_entropy_loss", loss, on_step=True, on_epoch=False, prog_bar=False
        )
        self.log("train_perplexity", torch.exp(loss), on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # FIXME: Validation step did not run in wandb log? Nor did checkpointing?
        batch = to_device_async(batch, self.device)
        batch_x = batch.x
        batch_y = batch.y
        batch_patch_lengths = batch.patch_lengths  # may be None
        mask = batch.mask  # may be None
        ngram_ids = batch.ngram_ids  # may be None

        predictions = self.forward(batch_x, batch_patch_lengths, ngram_ids)
        loss, _ = compute_loss(predictions, batch_y, mask, scale=1.0)

        # Log validation metrics every step. Use prog_bar=True when on Chimera (won't work on UHN)
        self.log(
            "val_entropy_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "val_perplexity",
            torch.exp(loss),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer, scheduler = build_optimizer(
            self.model, self.args.optim, self.args.steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


###############################################
# DataModule and Iterator Wrapper (No IterableDataset)
###############################################
class ByteLatentDataModule(pl.LightningDataModule):
    def __init__(self, dl_args: DataloaderArgs):
        super().__init__()
        self.dl_args = dl_args
        
        self.train_dataset: PackedBatchDataset | None = None
        self.val_dataset: PackedBatchDataset | None = None
        
        self._train_dataloader_instance: StatefulDataLoader | None = None
        self._val_dataloader_instance: StatefulDataLoader | None = None
        
        self._loaded_train_dataloader_state = None

    def setup(self, stage: str | None = None):
        # setup is called before train_dataloader, val_dataloader, etc.
        # We create datasets here. DataLoaders are created on-demand by their respective methods.
        self.train_dataset = PackedBatchDataset(self.dl_args, dataset_key="train")
        self.val_dataset = PackedBatchDataset(self.dl_args, dataset_key="validation")

    def train_dataloader(self) -> StatefulDataLoader:
        if self._train_dataloader_instance is None:
            self._train_dataloader_instance = StatefulDataLoader(
                self.train_dataset,
                batch_size=None,  # Dataset yields pre-formed batches
                num_workers=4, 
                pin_memory=True,
                persistent_workers=True # Crucial for stateful iteration
            )
            if self._loaded_train_dataloader_state is not None:
                self._train_dataloader_instance.load_state_dict(self._loaded_train_dataloader_state)
                self._loaded_train_dataloader_state = None # Clear after applying
        return self._train_dataloader_instance

    def val_dataloader(self) -> StatefulDataLoader:
        if self._val_dataloader_instance is None:
            self._val_dataloader_instance = StatefulDataLoader(
                self.val_dataset,
                batch_size=None,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
        return self._val_dataloader_instance

    # transfer_batch_to_device remains the same


###############################################
# Training Function and Main Entrypoint
###############################################


def train(args: TrainArgs, num_gpus):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("medium")
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch._dynamo.config.optimize_ddp = (
        False  # we may want to remove flex attention to use optimize_ddp
    )

    model = ByteLatentLightningModule(args)

    # Use test_mode parameter
    data_module = ByteLatentDataModule(args.data)

    # Set up Weights & Biases logger
    wandb_logger = WandbLogger(project="byte-latent")

    checkpoint_callback = ModelCheckpoint(
        filename="{step}-{val_entropy_loss:.2f}",
        monitor="val_entropy_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        max_steps=args.steps,
        strategy="ddp",
        accelerator="auto",
        devices=num_gpus,
        callbacks=[checkpoint_callback, StatefulDataloaderCheckpoint()],
        gradient_clip_val=None,  # must be none for fused adam
        accumulate_grad_batches=args.grad_acc_steps,
        precision="bf16-mixed",
        logger=wandb_logger,
        enable_progress_bar=False,
        log_every_n_steps=10,
        val_check_interval=200,
    )

    trainer.fit(model, datamodule=data_module)

    gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a ByteLatent model.")
    parser.add_argument(
        "--tokens",
        type=int,
        default=18_000_000_000,
        help="Number of tokens to train on.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--patch_size", type=int, default=2, choices=[2, 4], help="Patch size."
    )
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    parser.add_argument(
        "--dim_global", type=int, default=512, help="Global transformer dimension"
    )
    parser.add_argument(
        "--dim_local", type=int, default=256, help="Local transformers dimension"
    )
    parser.add_argument(
        "--global_layers", type=int, default=9, help="Global transformer layers."
    )
    parser.add_argument(
        "--decoder_layers", type=int, default=5, help="Decoder transformer layers."
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    args = parser.parse_args()

    grad_accum_size = 2 ** 21 // (args.batch_size * 8192 * args.num_gpus)

    steps = int(args.tokens) // (2 ** 21)
    seq_len = 8192 // args.patch_size

    if args.patch_size == 2:
        threshold = 1.1
        max_patch_length = 250
    elif args.patch_size == 4:
        threshold = 1.268
        max_patch_length = 952
    else:
        raise ValueError(f"Invalid patch size: {args.patch_size}. Must be 2 or 4.")

    train_args = TrainArgs(
        grad_acc_steps=grad_accum_size,
        steps=steps,
        max_steps=steps,
        data=DataloaderArgs(
            batch_size=args.batch_size,
            patcher_args=PatcherArgs(
                threshold=threshold,
                max_patch_length=max_patch_length,
            ),
            seq_len=seq_len,
            buffer_size=args.batch_size,
        ),
        optim=OptimArgs(
            lr=args.lr,
            warmup=int(steps * 0.05),
        ),
        model=ByteLatentTransformerArgs(
            dim_global=args.dim_global,
            dim_local_decoder=args.dim_local,
            dim_local_encoder=args.dim_local,
            n_layers_global=args.global_layers,
            n_layers_local_decoder=args.decoder_layers,
            n_heads_global=args.dim_global // 64,
            n_heads_local_decoder=args.dim_local // 64,
            n_heads_local_encoder=args.dim_local // 64,
            cross_attn_nheads=args.dim_local // 64,
            max_seqlen=seq_len,
        ),
    )
    train(train_args, args.num_gpus)
