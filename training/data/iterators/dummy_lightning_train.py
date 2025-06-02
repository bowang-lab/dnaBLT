import random
import gc
from typing import Any, Union, Dict
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from pytorch_lightning.strategies import DDPStrategy

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from v_args import DataloaderArgs, TrainArgs
from patching import PatcherArgs

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

###############################################
# Lightning Module (Minimal for Data Loading Test)
###############################################
class DummyLightningModule(pl.LightningModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        # We still need to save hyperparameters for the Trainer to work correctly,
        # even if the model itself is dummy.
        # Create a dummy model_dump if args.model is None
        # self.automatic_optimization = False
        # Add a dummy trainable parameter so DistributedDataParallel
        # has at least one parameter that requires gradients.
        self._dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def training_step(self, batch, batch_idx):
        # Log batch shapes to verify data loading
        print(f"Batch {batch_idx}: {batch.x}")
        # Return a dummy loss. Lightning expects a tensor.
        loss = self._dummy_param.sum()  # or just self._dummy_param.sum()
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad(): # Disable gradient computation for data loading test
            # Log batch shapes to verify data loading
            # Return a dummy loss. Lightning expects a tensor.
            loss = self._dummy_param.sum() * 0  # or just self._dummy_param.sum()
            return loss

    def configure_optimizers(self):
        # Dummy optimizer, not actually used for training.
        # Lightning requires an optimizer to be configured.
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

###############################################
# DataModule and Iterator Wrapper
###############################################
class PackedBatchDataset(IterableDataset):
    """
    IterableDataset that simply *delegates* to the existing iterator
    chain (Arrow → Preprocess → Sequence → Packing).

    Each element yielded is already a fully-formed `Batch` coming from
    `PackingIterator`, so no extra collation is required.
    """

    def __init__(
        self, dl_args: DataloaderArgs, dataset_key: str, shuffle: bool = False
    ):
        """
        Build an iterator constructor once at init.

        • For **training** we immediately materialise the long‑running iterator
          so its state can be checkpointed.

        • For **validation / test** we keep only the *factory*; a fresh
          iterator is produced on every call to ``__iter__``.
        """
        super().__init__()
        self.dl_args = dl_args
        self.dataset_key = dataset_key
        self.shuffle = shuffle

    def _make_iterator_for_worker(self):
        winfo = get_worker_info()  # None when num_workers = 0
        local_worker_id = 0 if winfo is None else winfo.id
        local_num_workers = 1 if winfo is None else winfo.num_workers

        # ------------------------------------------------------------------ #
        # Rank/world‑size: Lightning already set these env vars (`torchrun`)
        # so we can simply read them.  Do **not** create another PG here.
        # ------------------------------------------------------------------ #
        ddp_rank       = int(os.environ.get("RANK", 0))
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))

        # “Global” worker ids that cover *all* DDP ranks × DataLoader workers. Injective function.
        global_worker_id = ddp_rank * local_num_workers + local_worker_id
        global_num_workers = ddp_world_size * local_num_workers

        return self.dl_args.build_from_rank(
            worker_id=global_worker_id,
            num_workers=global_num_workers,
            mode=self.dataset_key,  # "train" / "validation"
            shuffle=self.shuffle,  # Shuffle for training, no shuffle for validation
        )
    # --------------------------------------------------------------------- #
    # Actual iteration
    # --------------------------------------------------------------------- #
    def __iter__(self):
        """
        Training  → keep yielding batches forever so that every DDP rank
        stays in lock‑step.  When the underlying iterator is exhausted we
        simply create a new one, effectively starting the next epoch
        without ever raising StopIteration on the training ranks.

        Validation/Test → keep the original single‑pass behaviour.
        """
        # Finite pass for validation / test
        if self.dataset_key != "train":
            return iter(self._make_iterator_for_worker())

        # Infinite stream for training
        worker_iter = iter(self._make_iterator_for_worker())
        while True:
            try:
                yield next(worker_iter)
            except StopIteration:
                # Iterator exhausted – start a new epoch
                worker_iter = iter(self._make_iterator_for_worker())

def build_dataloader(
    dl_args: DataloaderArgs,
    mode: str = "train",
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle: bool = False,
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
    dataset = PackedBatchDataset(dl_args, mode, shuffle)
    return DataLoader(
        dataset,
        batch_size=None,  # one Batch per iteration
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


class ByteLatentDataModule(pl.LightningDataModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        self.data_loader = None

    def setup(self, stage=None):
        self.train_data_loader = build_dataloader(self.args.data, mode="train", num_workers=4, pin_memory=True) # shuffle optional
        self.val_data_loader = build_dataloader(self.args.data, mode="validation", num_workers=4, pin_memory=True) # shuffle optional

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


def train_data_loader_test(args: TrainArgs, num_gpus):
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("medium")
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch._dynamo.config.optimize_ddp = False

    model = DummyLightningModule(args)
    data_module = ByteLatentDataModule(args)

    # ------------------------------------------------------------------ #
    # DDPStrategy (Gloo backend) – Lightning initialises the PG once.
    # No extra init in the dataset or workers.
    # ------------------------------------------------------------------ #
    strategy = DDPStrategy(process_group_backend="gloo", find_unused_parameters=False)

    trainer = pl.Trainer(
        max_steps=15,               # quick data‑loader test
        strategy=strategy,
        enable_progress_bar=False,
        devices=num_gpus,           # must match --nproc-per-node when using torchrun
        accelerator="cpu",
    )

    trainer.fit(model, datamodule=data_module)

    gc.collect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test the ByteLatent data loader with Lightning Trainer.")
    parser.add_argument(
        "--tokens",
        type=int,
        default=18_000_000_000,
        help="Number of tokens to train on.",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--patch_size", type=int, default=2, choices=[2, 4], help="Patch size."
    )
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs")
    args = parser.parse_args()

    grad_accum_size = 2**21 // (args.batch_size * 8192 * args.num_gpus)

    steps = int(args.tokens) // (2**21)
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
    )
    train_args.model = None # Set model to None for the dummy module
    train_data_loader_test(train_args, args.num_gpus)