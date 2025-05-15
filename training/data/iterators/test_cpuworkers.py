# tests/peek_first_batch_dataloader.py
"""
Show the very first model-ready Batch object that *each* DDP rank receives
from YOUR build_dataloader(...) helper – including multi-worker logic.

Run, for example:
    torchrun --standalone --nproc_per_node 4 tests/peek_first_batch_dataloader.py

If you want fewer/more workers, use:
    torchrun ... tests/peek_first_batch_dataloader.py --num_workers 4
"""
from __future__ import annotations
import argparse, json, torch, torch.distributed as dist
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info, DataLoader

# ---- import your code -------------------------------------------------
from v_args import TrainArgs, DataloaderArgs  # or however you load cfg
import time


# ----------------------------------------------------------------------
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

        if dist.is_available() and dist.is_initialized():
            ddp_rank = dist.get_rank()
            ddp_world_size = dist.get_world_size()
        else:
            ddp_rank, ddp_world_size = 0, 1

        # “Global” worker ids that cover *all* DDP ranks × DataLoader workers. Injective function.
        global_worker_id = ddp_rank * local_num_workers + local_worker_id
        global_num_workers = ddp_world_size * local_num_workers

        return self.dl_args.build_from_rank(
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            worker_id=global_worker_id,
            num_workers=global_num_workers,
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


# -------------------- pretty-print helper ------------------------------
def summarize(batch):
    """
    Turn an arbitrary Batch object (attrs or dict) into a JSON-serialisable
    summary: dtype + shape for tensors, str(type) otherwise.
    """
    items = batch.__dict__.items() if hasattr(batch, "__dict__") else batch.items()
    out = {}
    for k, v in items:
        if torch.is_tensor(v):
            out[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "first_vals": v.flatten()[:5].tolist(),
            }
        else:
            out[k] = str(type(v))
    return out


# -------------------- main ---------------------------------------------
def main(num_workers: int):
    # 1) initialise PG if it isn't already (torchrun does this for us)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    
    dist.barrier()
    t0 = time.time()

    # 2) build your config  (replace with YAML load if you prefer)
    cfg = TrainArgs()

    # 3) build the **real DataLoader** with the requested worker count
    loader = build_dataloader(
        cfg.data,
        mode="train",
        num_workers=num_workers,
        pin_memory=True,
    )

    # Iterate through dataset
    for _ in iter(loader):
        pass

    t1 = time.time()
    print(f"[RANK {dist.get_rank()}] Total wall time: {t1-t0:.3f} s")

    # optional: shut down workers cleanly (nice for CI runs)
    loader._iterator._shutdown_workers() if hasattr(loader, "_iterator") else None
    dist.barrier()
    dist.destroy_process_group()


# -------------------- CLI wrapper --------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="workers PER rank to spawn in DataLoader",
    )
    args = ap.parse_args()
    main(args.num_workers)
