# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
import os
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from bytelatent.checkpoint import CheckpointArgs
from bytelatent.data.data_types import Batch
from bytelatent.data.iterators.abstract_iterator import StatefulIterator
from bytelatent.data.iterators.arrow_iterator import (
    ArrowFileIterator,
    find_and_sanitize_chunks,
)
from bytelatent.data.iterators.looping_iterator import LoopingIterator
from bytelatent.data.iterators.multiprocess_iterator import MultiprocessIterator
from bytelatent.data.iterators.packing_iterator import PackingArgs, PackingIterator
from bytelatent.data.iterators.preprocess_iterator import PreprocessIterator
from bytelatent.data.iterators.sampling_iterator import SamplingIterator
from bytelatent.data.iterators.sequence_iterator import (
    SequenceIterator,
    SequencePackingArgs,
)
from bytelatent.data.patcher import PatcherArgs
from bytelatent.distributed import DistributedArgs, EnvironmentArgs
from bytelatent.metrics import LoggingArgs
from bytelatent.model.blt import ByteLatentTransformerArgs
from bytelatent.optim import OptimArgs
from bytelatent.profiling import ProfilerArgs
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.transformer import LMTransformerArgs

logger = logging.getLogger()


def get_rng_state(seed: int, rank: int, world_size: int) -> dict[str, Any]:
    return np.random.default_rng((seed, rank, world_size)).bit_generator.state


def distribute_data_to_rank(
    *,
    dataset_path: str,
    preprocess_dir: str,
    entropy_model_name: str | None,
    arrow_batch_size: int,
    rank: int,
    world_size: int,
    s3_profile: str | None = None,
) -> ArrowFileIterator:
    dataset_chunks = find_and_sanitize_chunks(
        dataset_path, world_size, s3_profile=s3_profile
    )
    n_workers_per_chunk = world_size // len(dataset_chunks)
    rank_to_arrow_iterator_params = []
    for chunk_path in dataset_chunks:
        for worker_id in range(n_workers_per_chunk):
            rank_to_arrow_iterator_params.append(
                ArrowFileIterator(
                    file_path=chunk_path,
                    worker_id=worker_id,
                    num_workers=n_workers_per_chunk,
                    preprocess_dir=preprocess_dir,
                    dataset_files=None,
                    entropy_model_name=entropy_model_name,
                    arrow_batch_size=arrow_batch_size,
                    s3_profile=s3_profile,
                )
            )
    return rank_to_arrow_iterator_params[rank]


class DataloaderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    s3_profile: str | None = None
    root_dir: str | None = None
    sources: dict[str, float] = {}
    batch_size: int = 2
    seq_len: int = 2048
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 64
    preprocess_dir: str | None = None
    dataset_files: list[str] | None = None
    entropy_model_name: str | None = "transformer_100m"
    arrow_batch_size: int = 100
    buffer_size: int = 64

    pad_to_max_length: bool = True
    max_encoder_seq_length: int = 12288
    enable_byte_ngrams: bool = False

    tokenizer_args: TokenizerArgs = TokenizerArgs()
    patcher_args: PatcherArgs = PatcherArgs()

    def _create_sequence_iterators(
        self, rank: int, world_size: int
    ) -> dict[str, SequenceIterator]:
        sequence_packing_args = SequencePackingArgs(
            output_seq_len=self.seq_len,
            buffer_size=self.buffer_size,
        )
        source_to_sequence_iterator: dict[str, SequenceIterator] = {}
        for dataset_path in self.sources:
            shuffle_rng_state = get_rng_state(self.seed + 1, rank, world_size)
            arrow_iterator = distribute_data_to_rank(
                dataset_path=os.path.join(self.root_dir, dataset_path),
                preprocess_dir=self.preprocess_dir,
                entropy_model_name=self.entropy_model_name,
                arrow_batch_size=self.arrow_batch_size,
                rank=rank,
                world_size=world_size,
                s3_profile=self.s3_profile,
            )
            looping_iterator = LoopingIterator(arrow_iterator)
            preprocess_iterator = PreprocessIterator(
                looping_iterator,
                patcher_args=self.patcher_args,
                tokenizer_args=self.tokenizer_args,
            )
            sequence_iterator = SequenceIterator(
                preprocess_iterator,
                sequence_packing_args=sequence_packing_args,
                rng_state=shuffle_rng_state,
            )

            source_to_sequence_iterator[dataset_path] = sequence_iterator
        return source_to_sequence_iterator

    def build_from_rank(
        self, rank: int, world_size: int
    ) -> StatefulIterator[Batch, Any]:
        source_to_sequence_iterators = self._create_sequence_iterators(rank, world_size)
        weight_rng_state = get_rng_state(self.seed + 1, rank, world_size)
        sampling_iterator = SamplingIterator(
            rng_state=weight_rng_state,
            source_to_weight=self.sources,
            source_to_iterator=source_to_sequence_iterators,
        )
        tokenizer = self.tokenizer_args.build()
        packing_args = PackingArgs(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            pad_id=tokenizer.boe_id,
            max_length=self.max_encoder_seq_length,
            pad_to_max_length=self.pad_to_max_length,
            enable_byte_ngrams=self.enable_byte_ngrams,
        )
        packing_iterator = PackingIterator(sampling_iterator, packing_args=packing_args)
        mp_iterator = MultiprocessIterator(
            packing_iterator, n_batches_to_prefetch=self.prefetch_size
        )

        return mp_iterator


class TrainArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    debug_dynamo: bool = False

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 1

    gc_collect_freq: int = 1000
    probe_freq: int | None = None

    # Nb optimizer steps to take
    steps: int = 1000

    data: DataloaderArgs = DataloaderArgs()
    optim: OptimArgs = OptimArgs()
    model: ByteLatentTransformerArgs = ByteLatentTransformerArgs()
    # This is only needed for training the entropy model
    entropy_model: LMTransformerArgs | None = None
    # Instead of training main model, train entropy model
    train_entropy_model: bool = False
    distributed: DistributedArgs = DistributedArgs()
    env: EnvironmentArgs = EnvironmentArgs()

    checkpoint: CheckpointArgs = CheckpointArgs()
    profiling: ProfilerArgs = ProfilerArgs()
    logging: LoggingArgs = LoggingArgs()

    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: int | None = None
    eval: Any | None = None
    eval_on_gpus: int | None = None

    def dump_to_yaml_file(
        self, path: str, log_config: bool = True, sort_keys: bool = True
    ):
        model_dict = self.model_dump(mode="json")
        yaml_str = yaml.dump(
            model_dict,
            allow_unicode=True,
            sort_keys=sort_keys,
            default_flow_style=False,
        )
        with open(path, "w") as f:
            if log_config:
                logger.info("Using the following config for this run:")
                logger.info(yaml_str)
            f.write(yaml_str)
