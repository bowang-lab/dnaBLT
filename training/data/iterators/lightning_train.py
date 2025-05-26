import random
import gc
from typing import Any, Union, Dict
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info, DataLoader

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

    def __init__(self, dl_args: DataloaderArgs, mode: str = "train"):
        super().__init__()
        self.dl_args = dl_args
        self.mode = mode
        self._iterator = None
        self._loaded_state_to_apply = None # Stores state loaded from checkpoint until __iter__ applies it

    def __iter__(self):
        winfo = get_worker_info()  # None when num_workers = 0
        local_worker_id = 0 if winfo is None else winfo.id
        local_num_workers = 1 if winfo is None else winfo.num_workers

        if torch.distributed.is_initialized():
            ddp_rank = torch.distributed.get_rank()  # 0‥world-1
            ddp_world_size = torch.distributed.get_world_size()  # world
        else:  # single-GPU / CPU debug runs
            ddp_rank, ddp_world_size = 0, 1

        # Create the iterator
        self._iterator = self.dl_args.build_from_rank(
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            worker_id=local_worker_id,
            num_workers=local_num_workers,
            mode=self.mode,  # "train" / "validation"
        )
        
        # Apply loaded state if it exists, after iterator is created
        if self._loaded_state_to_apply is not None:
            state_to_apply = self._loaded_state_to_apply
            self._loaded_state_to_apply = None # Clear it after fetching to prevent re-application

            if 'current_batch_idx' in state_to_apply and state_to_apply['current_batch_idx'] is not None:
                try:
                    target_idx = state_to_apply['current_batch_idx']
                    self._iterator.sequence_iterator.source_to_iterator['16b*']._src_iter.arrow_batch_iterator.current_batch_idx = target_idx
                    # print(f"Worker {local_worker_id} restored iterator to batch_idx: {target_idx}") # For debugging
                except Exception as e:
                    print(f"Worker {local_worker_id} ERROR applying state in __iter__: {e}. State was: {state_to_apply}")
            
        return iter(self._iterator)

    def get_state(self):
        """Get the current iterator state for checkpointing."""
        state = {}
        # If state was loaded but not yet applied by __iter__, that's the most current view for saving.
        if self._loaded_state_to_apply is not None and 'current_batch_idx' in self._loaded_state_to_apply:
            state['current_batch_idx'] = self._loaded_state_to_apply['current_batch_idx']
            return state

        # Otherwise, try to get state from the live iterator if it exists.
        if self._iterator is not None:
            try:
                current_batch_idx = (
                    self._iterator.sequence_iterator
                    .source_to_iterator['16b*']._src_iter
                    .arrow_batch_iterator.current_batch_idx
                )
                state['current_batch_idx'] = current_batch_idx
            except Exception:
                state['current_batch_idx'] = None # Or an initial default like 0 if preferred
        else:
            # No iterator created yet and no pending state to apply.
            state['current_batch_idx'] = None # Or an initial default like 0 if preferred
        return state

    def _set_state(self, state):
        """Store the iterator state. It will be applied when __iter__ is called."""
        # This method is called by ByteLatentDataModule.setup() before __iter__ is called.
        # It should store the state, not try to apply it immediately, as self._iterator is not yet created.
        self._loaded_state_to_apply = state
        # print(f"PackedBatchDataset._set_state called, storing: {state}") # For debugging


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
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        self.train_data_loader = None
        self.val_data_loader = None
        self.train_iterator_state = None
        self.current_batch_idx = 0

    def _build_dataloader(self, mode: str = "train"):
        """
        Returns a torch.utils.data.DataLoader whose `dataset` streams the same
        `Batch` objects you used to obtain from directly looping over
        `args.build_from_rank(...)`.

        • `batch_size` **must** stay `None` because the dataset already
          emits *batched* samples.
        • No `collate_fn` is needed: each iterator element is forwarded
          unchanged.
        """
        dataset = PackedBatchDataset(self.args.data, mode)
        return DataLoader(
            dataset,
            batch_size=None,  # one Batch per iteration
            num_workers=4,  # Using 4 workers by default
            pin_memory=True,
            persistent_workers=True,
        )

    def setup(self, stage=None):
        self.train_data_loader = self._build_dataloader(mode="train")
        self.val_data_loader = self._build_dataloader(mode="validation")
        # Restore iterator state if available
        if self.train_iterator_state is not None and hasattr(self.train_data_loader.dataset, '_set_state'):
            self.train_data_loader.dataset._set_state(self.train_iterator_state)

    # Modern PyTorch Lightning checkpoint hooks
    def state_dict(self):
        state_payload = {}
        if self.train_data_loader is not None and hasattr(self.train_data_loader.dataset, 'get_state'):
            local_iterator_state = self.train_data_loader.dataset.get_state()
        else:
            local_iterator_state = None # Or an empty dict, depending on what get_state might return

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            # Everyone needs to participate in all_gather_object
            gathered_states = [None] * world_size
            dist.all_gather_object(gathered_states, local_iterator_state)
            # The actual content of the checkpoint is typically written by rank 0.
            # For DataModule state, it's good practice for rank 0 to save all gathered states.
            # Other ranks will also have `gathered_states` but their return value from state_dict might be ignored by the saver.
            state_payload['per_rank_train_iterator_states'] = gathered_states
        else:
            # Non-DDP mode or DDP not initialized
            state_payload['per_rank_train_iterator_states'] = [local_iterator_state]
        
        return state_payload

    def load_state_dict(self, state_dict_from_ckpt):
        all_rank_states = state_dict_from_ckpt.get('per_rank_train_iterator_states')

        if all_rank_states:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                if rank < len(all_rank_states):
                    self.train_iterator_state = all_rank_states[rank]
                else:
                    # Current rank is out of bounds for the loaded states (e.g., trained on N GPUs, now running on M > N GPUs)
                    self.train_iterator_state = None # Start fresh for this rank
            else:
                # Non-DDP mode or DDP not initialized
                self.train_iterator_state = all_rank_states[0]
        else:
            self.train_iterator_state = None


    def train_dataloader(self):
        return self.train_data_loader

    def val_dataloader(self):
        return self.val_data_loader

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Override LightningModule to move all batch elements to the correct device.
        """
        return batch


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
    data_module = ByteLatentDataModule(args)

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
        callbacks=checkpoint_callback,
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
        "--grad_accum_size", type=int, default=8, help="Gradient accumulation size."
    )
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

    steps = int(args.tokens) // (
        args.batch_size * args.grad_accum_size * 8192 * args.num_gpus
    )  # guard against tokens float
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
        grad_acc_steps=args.grad_accum_size,
        steps=steps,
        max_steps=steps,
        data=DataloaderArgs(
            batch_size=args.batch_size,
            patcher_args=PatcherArgs(
                threshold=threshold,
                max_patch_length=max_patch_length,
            ),
            seq_len=seq_len,
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
