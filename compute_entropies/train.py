import gc
import logging
import math
import os
import sys
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass
from timeit import default_timer as timer
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed
import torch.nn.functional
import torch.nn.functional as F
import wandb
import xformers.profiler
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim import lr_scheduler

from bytelatent.args import TrainArgs
from bytelatent.checkpoint import CheckpointManager, load_from_checkpoint
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.iterators.abstract_iterator import get_state_and_refresh
from bytelatent.data.iterators.multiprocess_iterator import (
    MultiprocessIterator,
    MultiprocessIteratorState,
)
from bytelatent.data.iterators.packing_iterator import PackingIteratorState
from bytelatent.distributed import (
    check_model_value_range,
    clean_env,
    dist_mean,
    dist_sum,
    get_device_mesh,
    get_is_master,
    get_world_size,
    init_signal_handler,
    parallelize_model,
    requeue_slurm_job,
    setup_env,
    setup_torch_distributed,
)
from bytelatent.eval import EVAL_FOLDER_NAME, launch_eval
from bytelatent.logger import init_logger
from bytelatent.metrics import GPUMemoryMonitor, MetricLogger, get_num_params
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.norms import fixed_clip_grad_norm_
from bytelatent.optim import build_optimizer
from bytelatent.probe import AutoProbeD
from bytelatent.profiling import maybe_run_profiler
from bytelatent.stool import StoolArgs, launch_job
from bytelatent.transformer import (
    LMTransformer,
    build_fsdp_grouping_plan,
    get_no_recompute_ops,
    get_num_flop_per_token,
    tp_parallelize,
)

logger = logging.getLogger()

def get_iterator_state_name(iterator_state):
    if isinstance(iterator_state, MultiprocessIteratorState):
        return "multiprocess"
    elif isinstance(iterator_state, PackingIteratorState):
        return "packing"
    else:
        raise ValueError(f"Unsupported iterator to get name from: {iterator_state}")


@dataclass
class TrainState(Stateful):
    step: int  # Nb of steps taken by the optimizer
    acc_step: int  # Nb of accumulation steps done since last optimizer step
    scheduler: lr_scheduler.LambdaLR
    data_loader_state: MultiprocessIteratorState | PackingIteratorState
    scale: float = 1.0
    data_loader_class: str | None = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "acc_step": self.acc_step,
            "data_loader_state": self.data_loader_state.model_dump(),
            "data_loader_class": get_iterator_state_name(self.data_loader_state),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.acc_step = state_dict["acc_step"]
        self.data_loader_class = state_dict["data_loader_class"]
        if self.data_loader_class == "multiprocess":
            self.data_loader_state = MultiprocessIteratorState(
                **state_dict["data_loader_state"]
            )
        elif self.data_loader_class == "packing":
            self.data_loader_state = PackingIteratorState(
                **state_dict["data_loader_state"]
            )
        else:
            raise ValueError(f"invalid data loader class: {self.data_loader_class}")
        self.scheduler.load_state_dict(state_dict["scheduler"])

def validate_train_args(args: TrainArgs, output_size: int):
    assert args.model is not None or args.entropy_model is not None
    if args.model is not None:
        logger.info(f"Setting model output size to {args.model.vocab_size}")
        args.model.vocab_size = output_size
        assert (
            args.model.max_encoder_seq_length == args.data.max_encoder_seq_length
        ), "max_encoder_seq_length for model and data should match"

    if args.entropy_model is not None:
        logger.info(f"Setting model output size to {args.entropy_model.vocab_size}")
        args.entropy_model.vocab_size = output_size

    assert args.dump_dir, "Dump dir not set"

    if args.checkpoint.path is None:
        logger.info(f"Setting checkpoint path to {args.checkpoint.path}")
        args.checkpoint.path = os.path.join(args.dump_dir, "checkpoints")

    data_fs = get_fs(args.data.root_dir, s3_profile=args.data.s3_profile)
    for source in args.data.sources:
        data_path = os.path.join(args.data.root_dir, source)
        assert data_fs.exists(data_path), f"{data_path} doesn't exist"

    if (
        args.distributed.dp_replicate
        * args.distributed.dp_shard
        * args.distributed.tp_size
        != get_world_size()
    ):
        logging.info("Modifying TrainArgs distributed config")
        assert get_world_size() % args.distributed.dp_shard == 0
        logging.info("World size: %s", get_world_size())
        logging.info(
            "Existing setting: train_args.distributed.dp_shard=%s",
            args.distributed.dp_shard,
        )
        logging.info(
            "Setting train_args.distributed.dp_replicate=%s, was dp_replicate=%s",
            get_world_size() // args.distributed.dp_shard,
            args.distributed.dp_replicate,
        )
        args.distributed.dp_replicate = get_world_size() // args.distributed.dp_shard

        logging.info(
            "Changing dp_replicate from %s to %s, to account for tp_size=%s",
            args.distributed.dp_replicate,
            args.distributed.dp_replicate // args.distributed.tp_size,
            args.distributed.tp_size,
        )
        assert args.distributed.dp_replicate % args.distributed.tp_size == 0
        args.distributed.dp_replicate = (
            args.distributed.dp_replicate // args.distributed.tp_size
        )

        logger.warning(
            f"Setting Data Parallel size to {args.distributed.dp_replicate * args.distributed.dp_shard}"
        )
        assert (
            args.distributed.dp_replicate
            * args.distributed.dp_shard
            * args.distributed.tp_size
            == get_world_size()
        )

        if args.distributed.fsdp_type == "no_shard":
            assert (
                args.distributed.dp_shard == 1
                and args.distributed.dp_replicate == get_world_size()
            )

    if args.model is not None:
        args.model.max_seqlen = args.data.seq_len
    if args.entropy_model is not None:
        args.entropy_model.max_seqlen = args.data.seq_len

    if args.distributed.tp_size == 1:
        logger.warning(
            "Tensor parallelism has not been tested for a while, use at your own risk"
        )

    assert (
        args.probe_freq != args.profiling.mem_steps
    ), "Don't profile during probe step"
    assert (
        args.probe_freq != args.profiling.profile_steps
    ), "Don't profile during probe step"
    if args.logging.wandb is not None:
        args.logging.wandb.name = args.name

    if args.probe_freq is not None:
        assert (
            args.distributed.tp_size == 1
        ), "Probing not supported with tensor parallelism"
        assert (
            args.distributed.selective_activation_checkpointing is False
        ), "Probing not supported with selective activation checkpointing"


def train(args: TrainArgs):
    # ----- Setup: Initialize tokenizer, validate args, seed, and build model -----
    tokenizer = args.data.tokenizer_args.build()
    validate_train_args(args, tokenizer.n_words)
    torch.manual_seed(args.seed)
    
    # Build the model on a meta device to allow building huge models
    with torch.device("meta"):
        if args.train_entropy_model:
            model = LMTransformer(args.entropy_model)
            model_args = args.entropy_model
        else:
            model = ByteLatentTransformer(args.model)
            model_args = args.model

    # Parallelize and prepare model (abstracted distributed logic)
    world_mesh = get_device_mesh(args.distributed)
    model = parallelize_model(
        model,
        world_mesh,
        model_args,
        args.distributed,
        fsdp_grouping_plan=build_fsdp_grouping_plan(model_args),
        tp_parallelize=tp_parallelize,
        no_recompute_ops=get_no_recompute_ops(),
    )
    model = model.to_empty(device="cuda")
    
    # Initialize model weights: either from a checkpoint or freshly
    if args.checkpoint.init_ckpt_path:
        load_from_checkpoint(
            get_fs(args.checkpoint.init_ckpt_path, s3_profile=args.checkpoint.s3_profile),
            args.checkpoint.init_ckpt_path,
            model,
            model_key="model"
        )
        model.rope_embeddings.reset_parameters()  # Reset RoPE buffers if needed
    else:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(model_args.seed)
            model.init_weights()
    check_model_value_range(model, range=10.0, std=1.0)
    
    # ----- Setup: Create optimizer, scheduler, and data loader -----
    optimizer, scheduler = build_optimizer(model, args.optim, args.steps)
    # TODO: need to define the DP rank and degree stuff based on our implementation in the auxialiry_entropy.py
    data_loader = args.data.build_from_rank(dp_rank, dp_degree)
    batch_iterator = data_loader.create_iter()
    train_state = TrainState(
        step=0,
        acc_step=0,
        data_loader_state=data_loader.get_state(),
        scheduler=scheduler,
        scale=1.0,
    )

    checkpoint = CheckpointManager.instantiate_and_make_dir(args.checkpoint)
    checkpoint.load(model, optimizer, train_state, world_mesh)

    # ----- Main Training Loop -----
    model.train()
    while train_state.step < args.steps and (args.max_steps is None or train_state.step < args.max_steps):
        # Update gradient accumulation counter (wraps around at grad_acc_steps)
        train_state.acc_step = (train_state.acc_step + 1) % args.grad_acc_steps

        # Load batch and move data to GPU
        batch = next(batch_iterator)
        batch_x = torch.from_numpy(batch.x).cuda()
        batch_y = torch.from_numpy(batch.y).cuda()
        # (Optional: process additional inputs like patch_lengths, mask, ngram_ids if available)

        # Forward pass: model prediction
        if args.train_entropy_model:
            pred = model(batch_x)
        else:
            pred = model(batch_x)
        
        # Compute loss (and optionally token-level loss) and scale for gradient accumulation
        loss, _ = compute_loss(pred, batch_y, mask=None, scale=train_state.scale)
        loss = loss / args.grad_acc_steps

        # Backward pass on the scaled loss
        loss.backward()

        # Clip gradients to avoid explosion
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.optim.clip)

        # When accumulated gradients are ready, update weights and reset gradients
        if train_state.acc_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_state.step += 1

            # Simple logging of step and loss
            logger.info(f"Step {train_state.step}: Loss = {loss.item()}, Grad Norm = {grad_norm}")

    # ----- Final Cleanup: Optionally save checkpoint -----
    checkpoint.save(model, optimizer, train_state, args)



import torch
import pytorch_lightning as pl

class MyLightningModule(pl.LightningModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        # --- Setup: Tokenizer and argument validation ---
        tokenizer = args.data.tokenizer_args.build()
        validate_train_args(args, tokenizer.n_words)
        torch.manual_seed(args.seed)
        
        # --- Build model on meta device ---
        with torch.device("meta"):
            if args.train_entropy_model:
                self.model = LMTransformer(args.entropy_model)
                self.model_args = args.entropy_model
            else:
                self.model = ByteLatentTransformer(args.model)
                self.model_args = args.model

        # --- Parallelize model ---
        world_mesh = get_device_mesh(args.distributed)
        self.model = parallelize_model(
            self.model,
            world_mesh,
            self.model_args,
            args.distributed,
            fsdp_grouping_plan=build_fsdp_grouping_plan(self.model_args),
            tp_parallelize=tp_parallelize,
            no_recompute_ops=get_no_recompute_ops(),
        )
        self.model = self.model.to_empty(device="cuda")
        
        # --- Initialize weights or load checkpoint ---
        if args.checkpoint.init_ckpt_path:
            fs = get_fs(args.checkpoint.init_ckpt_path, s3_profile=args.checkpoint.s3_profile)
            load_from_checkpoint(fs, args.checkpoint.init_ckpt_path, self.model, model_key="model")
            self.model.rope_embeddings.reset_parameters()
        else:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(self.model_args.seed)
                self.model.init_weights()
        check_model_value_range(self.model, range=10.0, std=1.0)
        
        # --- Create optimizer and scheduler ---
        self.optimizer, self.scheduler = build_optimizer(self.model, args.optim, args.steps)
        
        # --- Prepare DataLoader ---
        # It is recommended to use a LightningDataModule. Here we assume that your data
        # API already returns a DataLoader. Note: dp_rank and dp_degree would be handled by Lightning.
        self.train_dataloader_obj = args.data.build_from_rank(dp_rank=0, dp_degree=1)

    def forward(self, x, patch_lengths=None, ngram_ids=None):
        # Forward pass (adapt kwargs as needed)
        if self.args.train_entropy_model:
            return self.model(x)
        else:
            return self.model(x, patch_lengths=patch_lengths, ngram_ids=ngram_ids)
        
    def training_step(self, batch, batch_idx):
        # Convert numpy arrays to torch tensors and move to GPU if needed
        batch_x = torch.from_numpy(batch.x).cuda()
        batch_y = torch.from_numpy(batch.y).cuda()
        mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda() ## same logic from meta train loop
        
        # Process additional inputs if they exist in the batch
        if batch.patch_lengths is None:
            patch_lengths = None
        else:
            patch_lengths = torch.from_numpy(batch.patch_lengths).cuda() ## same logic (unchanged)
        

        ngram_ids = (
                None
                if batch.ngram_ids is None
                else torch.from_numpy(batch.ngram_ids).cuda()
        ) ## same logic
        
        # Forward pass with appropriate parameters based on the model type
        if self.args.train_entropy_model:
            pred = self.forward(batch_x)
        else:
            pred = self.forward(batch_x, patch_lengths=patch_lengths, ngram_ids=ngram_ids)
        
        # we compute loss here -- 
        loss, _ = compute_loss(pred, batch_y, mask, scale=1.0)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        # Return both optimizer and scheduler.
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",  # or "epoch" based on your scheduler
            },
        }
    
    def train_dataloader(self):
        # Lightning will use this DataLoader for training.
        return self.train_dataloader_obj

# --- Usage with PyTorch Lightning Trainer ---
from pytorch_lightning import Trainer

def main(args: TrainArgs):

    model = MyLightningModule(args)

    trainer = Trainer(
        max_steps=args.steps,
        accumulate_grad_batches=args.grad_acc_steps,
        gpus=-1,  
        accelerator="input whatever",  
    )

    trainer.fit(model)
