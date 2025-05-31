import logging
import os
import math
from typing import Any
from enum import Enum

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict
import fsspec

from sampling_iterator import SamplingIterator
from v_preprocess_iterator import PreprocessIterator
from v_arrow_iterator import ArrowFileIterator
from v_sequence_iterator import SequenceIterator, PackedSequence
from v_packing_iterator import PackingArgs, PackingIterator, PackingMode

from blt import ByteLatentTransformerArgs
from optim import OptimArgs
from build_tokenizer import TokenizerArgs
from patching import PatcherArgs, PatchingModeEnum

logger = logging.getLogger()

class SaveEvery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    every: int = 500
    keep: int = 0

class CheckpointArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dump: SaveEvery = SaveEvery()
    eval: SaveEvery = SaveEvery()
    path: str | None = None
    init_ckpt_path: str | None = None
    continue_training_from_init: bool = False
    s3_profile: str | None = None


TRAIN_DATA_FILE_PATTERN = "*.arrow"



def get_rng_state(seed: int, rank: int, world_size: int) -> dict[str, Any]:
    return np.random.default_rng((seed, rank, world_size)).bit_generator.state

def get_fs(path: str, s3_profile: str | None = None) -> fsspec.AbstractFileSystem:
    if path.startswith("s3://"):
        if s3_profile is None:
            return fsspec.filesystem("s3")
        else:
            return fsspec.filesystem("s3", profile=s3_profile)
    else:
        return fsspec.filesystem("file")

def find_and_sanitize_chunks(
    dataset_path: str,
    world_size: int,
    file_pattern: str,
    s3_profile: str | None = None,
):
    fs = get_fs(dataset_path, s3_profile=s3_profile)
    path_with_glob = os.path.join(dataset_path, file_pattern)
    dataset_chunks = fs.glob(path_with_glob)
    n_chunks = len(dataset_chunks)

    if n_chunks > world_size:
        n_discard = n_chunks - world_size
        dataset_chunks = dataset_chunks[:world_size]
    else:
        assert (
            world_size % n_chunks == 0
        ), "World size should be a multiple of number of chunks"

    assert n_chunks > 0, f"No valid chunks in {dataset_path}"

    return dataset_chunks


def distribute_data_to_rank(
    *,
    dataset_path: str,
    entropy_files: str,
    preprocess_dir: str,
    entropy_model_name: str | None,
    arrow_batch_size: int,
    ddp_rank: int,
    ddp_world_size: int,
    worker_id: int,
    num_workers: int,
    file_format: str,
    s3_profile: str | None = None,
    file_pattern: str = TRAIN_DATA_FILE_PATTERN,
    shuffle: bool = False,
) -> ArrowFileIterator:
    """
    Build an `ArrowFileIterator` that is aware of the global DDP rank.

    * The global `rank` is used directly as `worker_id`, and `world_size`
      is passed as `num_workers`.  This allows the iterator to deterministically
      pick a disjoint subset of files using the modulo logic added in
      `_select_shard_files`.
    * If the number of dataset shards is smaller than `world_size`, replicate
      the list so every rank still receives at least one file.  This prevents
      the empty‑shard error raised inside `ArrowFileIterator`.
    """
    dataset_chunks = find_and_sanitize_chunks(
        dataset_path, ddp_world_size, entropy_files, s3_profile=s3_profile
    )

    # Ensure there are at least as many shards as ranks; replicate if needed.
    # HACK:
    # files_per_rank = math.ceil(len(dataset_files) / ddp_world)
    # start = ddp_rank * files_per_rank
    # end   = start + files_per_rank
    # selected_files = dataset_files[start:end]

    if len(dataset_chunks) < ddp_world_size:
        reps = math.ceil(ddp_world_size / len(dataset_chunks))
        dataset_chunks = (dataset_chunks * reps)[:ddp_world_size]

    return ArrowFileIterator(
        file_path=None,
        file_format=file_format,
        ddp_rank=ddp_rank,          # global DDP rank
        ddp_world=ddp_world_size,  # global DDP world size
        worker_id=worker_id,        # local DDP rank
        num_workers=num_workers,  # total ranks
        preprocess_dir=preprocess_dir,
        dataset_files=dataset_chunks,
        entropy_model_name=entropy_model_name,
        arrow_batch_size=arrow_batch_size,
        shuffle=shuffle
    )


class PackedCausalTransformerGeneratorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float = 0.0
    top_p: float | None = None
    top_k: float | None = None
    max_gen_len: int = 512  # Maximum number of tokens to generate
    max_tokens: int = 1024  # Maximum number of tokens that can go through the model
    max_prompt_len: int | None = None
    until: list[str] = []
    compile_prefilling: bool = False
    reduce_generation_overhead: bool = False
    show_progress: bool = False
    dtype: str | None = "bf16"
    device: str | None = "cuda"





class DataloaderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    s3_profile: str | None = None
    # root_dir: str | None = "/large_storage/goodarzilab/ashah"
    root_dir: str | None = "/Users/arnavshah/Documents"
    # ingenious hack: use 16b[34].arrow glob match for new data if crash (pre StatefulDataLoader)
    sources: dict[str, dict[str, float]] = {"train": {"16b*": 1}, "validation": {"entropies_validation.arrow": 1}}
    batch_size: int = 16
    seq_len: int = 4096
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 64
    preprocess_dir: str | None = None
    dataset_files: list[str] | None = None
    # entropy_model_name: str | None = "transformer_100m"
    entropy_model_name: str | None = None
    arrow_batch_size: int = 128 # can't be larger unless we want to rewrite the entropy tensors.
    buffer_size: int = 16
    file_format: str = "arrow"

    pad_to_max_length: bool = True
    max_encoder_seq_length: int = 8192
    enable_byte_ngrams: bool = False

    add_patches: bool = True

    tokenizer_args: TokenizerArgs = TokenizerArgs()
    patcher_args: PatcherArgs = PatcherArgs()

    def _create_sequence_iterators(
        self, ddp_rank: int, ddp_world_size: int, worker_id: int, num_workers: int, mode: str = "train", shuffle=False,
    ) -> dict[str, SequenceIterator]:
        source_to_sequence_iterator: dict[str, SequenceIterator] = {}
        for dataset_path in self.sources[mode]:
            shuffle_rng_state = get_rng_state(self.seed + 1, worker_id, num_workers)
            arrow_iterator = distribute_data_to_rank(
                file_format=self.file_format,
                dataset_path=self.root_dir,
                entropy_files=dataset_path,
                preprocess_dir=self.preprocess_dir,
                entropy_model_name=self.entropy_model_name,
                arrow_batch_size=self.arrow_batch_size,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                worker_id=worker_id,
                num_workers=num_workers,
                s3_profile=self.s3_profile,
                shuffle=shuffle,
            )
            looping_iterator = arrow_iterator
            preprocess_iterator = PreprocessIterator(
                looping_iterator,
                patcher_args=self.patcher_args,
                # tokenizer_args=self.tokenizer_args,
                add_patches=self.add_patches,
            )
            sequence_iterator = SequenceIterator(
                preprocess_iterator,
                max_seq_patches=(self.seq_len, self.buffer_size),
                rng_state=shuffle_rng_state,
            )

            source_to_sequence_iterator[dataset_path] = sequence_iterator
        return source_to_sequence_iterator

    def build_from_rank(
        self, ddp_rank: int, ddp_world_size: int, worker_id: int, num_workers: int, mode: str = "train", shuffle=False
    ):
        source_to_sequence_iterators = self._create_sequence_iterators(ddp_rank, ddp_world_size, worker_id, num_workers, mode, shuffle)
        weight_rng_state = get_rng_state(self.seed + 1, worker_id, num_workers)
        sampling_iterator = SamplingIterator(
            rng_state=weight_rng_state,
            source_to_weight=self.sources[mode],
            source_to_iterator=source_to_sequence_iterators,
        )
        tokenizer = self.tokenizer_args.build()
        if self.tokenizer_args.name == "bytes":
            pad_id = 0
        else:
            pad_id = tokenizer.boe_id
        packing_args = PackingArgs(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            pad_id=pad_id,
            max_length=self.max_encoder_seq_length,
            pad_to_max_length=self.pad_to_max_length,
            enable_byte_ngrams=self.enable_byte_ngrams,
            packing_mode=(
                PackingMode.BYTES
                if self.patcher_args.patching_mode == PatchingModeEnum.byte
                else PackingMode.PATCHING
            ),
        )
        packing_iterator = PackingIterator(sampling_iterator, packing_args=packing_args)
  
        return packing_iterator


class LMHarnessArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[Any] | None = None
    num_fewshot: int | None = None
    device: str | None = None
    use_cache: str | None = None
    cache_requests: bool = False
    rewrite_requests_cache: bool = False
    delete_requests_cache: bool = False
    limit: int | float | None = None
    bootstrap_iters: int = 100000
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = True
    system_instruction: str | None = None
    apply_chat_template: bool | str = False
    fewshot_as_multiturn: bool = False
    gen_kwargs: str | None = None
    verbosity: str = "INFO"
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234


class ValidationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_n_docs: int | None = (
        None  # If None the whole validation file is used -> /!\ This number of steps is gpu dependent (100 max steps on 8 gpus = 800 steps on 1 gpu)
    )
    use_val_from_train_src: bool = True  # Use the validation set from training sources
    root_dir: str = ""
    sources: list[str] = []  # Other sources to eval on


class EvalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dump_dir: str | None = None
    ckpt_dir: str | None = None
    metric_log_dir: str | None = None
    generator: PackedCausalTransformerGeneratorArgs = (
        PackedCausalTransformerGeneratorArgs()
    )

    harness: LMHarnessArgs | None = LMHarnessArgs()
    validation: ValidationArgs | None = ValidationArgs()

    global_step: int | None = None  # for in-training evaluation
    s3_profile: str | None = None


class TrainArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "lingua"
    dump_dir: str = ""

    seed: int = 42

    debug_dynamo: bool = False

    # Number of gradient accumulation steps
    # Total batch size is batch_size*grad_acc_steps
    grad_acc_steps: int = 8

    gc_collect_freq: int = 1000
    probe_freq: int | None = None

    # Nb optimizer steps to take | 9B token run / 2M token batch = 4500 steps.
    steps: int = 17166 
    # If not None, halt training after this many steps,
    # useful for debugging
    max_steps: int | None = 17166

    data: DataloaderArgs = DataloaderArgs()
    optim: OptimArgs = OptimArgs()
    model: ByteLatentTransformerArgs | None = ByteLatentTransformerArgs()
    # This is only needed for training the entropy model
    entropy_model: None = None
    # Instead of training main model, train entropy model
    train_entropy_model: bool = False

    checkpoint: CheckpointArgs = CheckpointArgs()


    # If set to None, eval is run locally otherwise it launches a new job with the given number of gpus
    async_eval_gpus: int | None = None
    eval: EvalArgs | None = None
    eval_on_gpus: int | None = None

    def dump_to_yaml_file(
        self, path: str, log_config: bool = True, sort_keys: bool = True
    ):
        yaml_str = self.dump_to_yaml_str(sort_keys=sort_keys)
        with open(path, "w") as f:
            if log_config:
                logger.info("Using the following config for this run:")
                logger.info(yaml_str)
            f.write(yaml_str)

    def dump_to_yaml_str(self, sort_keys: bool = True):
        model_dict = self.model_dump(mode="json")
        yaml_str = yaml.dump(
            model_dict,
            allow_unicode=True,
            sort_keys=sort_keys,
            default_flow_style=False,
        )
        return yaml_str
