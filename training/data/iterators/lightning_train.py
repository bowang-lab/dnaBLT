import random
import gc
from typing import Any, Union, Dict
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info, DataLoader

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import AdvancedProfiler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

import torch._dynamo
torch._dynamo.config.suppress_errors = True


from bytelatent.model.blt import ByteLatentTransformer
# make sure to import the BLT here
from optim import build_optimizer
from v_args import DataloaderArgs, TrainArgs

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

    def _make_iterator_for_worker(self):
        winfo = get_worker_info()  # None when num_workers = 0
        local_worker_id = 0 if winfo is None else winfo.id
        local_num_workers = 1 if winfo is None else winfo.num_workers

        if torch.distributed.is_initialized():
            ddp_rank       = torch.distributed.get_rank()        # 0‥world-1
            ddp_world_size = torch.distributed.get_world_size()  # world
        else:                       # single-GPU / CPU debug runs
            ddp_rank, ddp_world_size = 0, 1

        # “Global” worker ids that cover *all* DDP ranks × DataLoader workers. Injective function.
        # global_worker_id = ddp_rank * local_num_workers + local_worker_id
        # global_num_workers = ddp_world_size * local_num_workers

        return self.dl_args.build_from_rank(
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            worker_id=local_worker_id,
            num_workers=local_num_workers,
            mode=self.mode,  # "train" / "validation"
        )

    def __iter__(self):
        return iter(self._make_iterator_for_worker())


def build_dataloader(
    dl_args: DataloaderArgs,
    mode: str = "train",
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Returns a torch.utils.data.DataLoader whose `dataset` streams the same
    `Batch` objects you used to obtain from directly looping over
    `args.build_from_rank(...)`.

    • `batch_size` **must** stay `None` because the dataset already
      emits *batched* samples.
    • No `collate_fn` is needed: each iterator element is forwarded
      unchanged.
    """
    dataset = PackedBatchDataset(dl_args, mode)
    return DataLoader(
        dataset,
        batch_size=None,  # one Batch per iteration
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

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
    return batch_type(x=x, y=y, patch_lengths=patch_lengths, mask=mask, ngram_ids=ngram_ids)

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
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            max_norm=self.args.optim.clip
        )

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
        self.log("train_entropy_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
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
        self.train_data_loader = build_dataloader(self.args.data, mode="train", num_workers=0, pin_memory=True) # change on gpu
        self.val_data_loader = build_dataloader(self.args.data, mode="validation", num_workers=0, pin_memory=True) # change on gpu
    
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

    # AdvancedProfiler will write a human-readable summary to profile.txt after training.
    # To view: simply open profile.txt after training completes.
    profiler = AdvancedProfiler(
        dirpath=".",              # current directory
        filename="profile.txt"    # human-readable profile output
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
        log_every_n_steps=50,
        profiler=profiler,  # <-- Add this line
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
    train_args.steps = 50 # for profiling purposes.
    train(train_args)
