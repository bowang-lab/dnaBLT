import os
import re
import random
from logging import getLogger
from typing import Generator, Any, Optional

import pyarrow as pa
import pyarrow.dataset as dataset
from blterror import ByteLatentError
from bltexample import BltExample
# from bytelatent.preprocess.preprocess_entropies import get_id_key, get_text

# NOTE:
#   * `rank` / `world_size` partition **files** across DDP processes.
#   * `worker_id` / `num_workers` partition **batches** across DataLoader
#     worker subprocesses *within* a single rank.
#   This lets us run multiple workers per rank without duplicating data.

logger = getLogger(__name__)

def get_text(doc: dict):
    if "text" in doc:
        text = doc["text"]
    elif "content" in doc:
        text = doc["content"]
    else:
        raise ValueError(f"Could not find a text key from: {doc.keys()}")
    return text


def get_id_key(doc: dict) -> int:
    """
    We need a reliable way to ensure that samples from jsonl
    and arrow are the same, but there is no unique id field,
    so derive the best possible
    """
    if "sample_id" in doc:
        return "sample_id"
    elif "title" in doc:
        return "title"
    elif "qid" in doc:
        return "qid"
    elif "paper_id" in doc:
        return "paper_id"
    elif "path" in doc:
        return "path"
    elif "url" in doc:
        return "url"
    elif "id" in doc:
        return "id"
    else:
        raise ValueError(f"Could not find a id key from: {doc.keys()}")


def shard_sort_key(file: str):
    match = re.search(r".+\.shard_([0-9]+)\.arrow", file)
    return int(match.group(1))


def _select_shard_file(dataset_files: list[str], index: int, modulo: int) -> list[str]:
    """
    Deterministically assign a disjoint subset of `dataset_files` to the
    **rank** identified by `index`, assuming `modulo` total ranks.
    """
    if modulo == 1:
        return dataset_files[0]
    for idx, f in enumerate(dataset_files):
        if idx % modulo == index:
            return f


def maybe_truncate_string(text: str, max_length: int):
    return text if len(text) <= max_length else text[:max_length] + "..."


class ArrowFileIterator:
    def __init__(
        self,
        *,
        file_path: Optional[str] = None,
        ddp_rank: int = 0, ddp_world: int = 1,
        worker_id: int = 0,
        num_workers: int = 1,
        preprocess_dir: Optional[str] = None,
        entropy_model_name: Optional[str] = None,
        arrow_batch_size: int = 100,
        dataset_files: Optional[list[str]] = None,
        file_format: str = "arrow",
        shuffle: bool = False, # false for apples-to-apples runs. unless resuming from crashed run (pre-StatefulDataIterator)
        seed: int = 42,
    ):
        assert 0 <= worker_id < num_workers, (worker_id, num_workers)
        self.rank = ddp_rank
        self.world_size = ddp_world
        if file_path is None and dataset_files is None:
            raise ByteLatentError("file_path and dataset_files cannot both be None")

        self.row_num = 0
        self.batch_iterator = None
        self.batch_to_consume = None
        self.dataset = None

        self.file_path = file_path
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.preprocess_dir = preprocess_dir
        self.entropy_model_name = entropy_model_name
        self.arrow_batch_size = arrow_batch_size
        self.file_format = file_format
        self.shuffle = shuffle
        self.seed = seed

        self.dataset_files = self._initialize_dataset_files(file_path, dataset_files, file_format)

        # Split files across ranks (DDP processes), not DataLoader workers
        self.shard_file = _select_shard_file(self.dataset_files, self.rank, self.world_size)
        self.reader = pa.ipc.open_file(self.shard_file)
        
        # Initialize RNG for shuffling if needed
        self.current_batch_idx = 0
        if self.shuffle:
            # Create a unique RNG for this worker to ensure deterministic shuffling
            self.rng = random.Random(self.seed + self.rank * 1000 + self.worker_id)
            # Pre-compute all batch indices for this worker
            self.batch_indices = list(range(self.worker_id, self.reader.num_record_batches, self.num_workers))
            self.rng.shuffle(self.batch_indices)

    def _initialize_dataset_files(self, file_path, dataset_files, file_format):
        if dataset_files is not None:
            return dataset_files

        assert file_path is not None, "Must specify file_path if dataset_files is None"

        if file_format == "json":
            return [file_path]
        else:
            return sorted(
                [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith(".arrow")],
                key=shard_sort_key,
            )

    def get_state(self):
        return {
            "file_path": self.file_path,
            "row_num": self.row_num,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "preprocess_dir": self.preprocess_dir,
            "entropy_model_name": self.entropy_model_name,
            "arrow_batch_size": self.arrow_batch_size,
            "dataset_files": self.dataset_files,
            "file_format": self.file_format,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
        }

    @classmethod
    def from_state(cls, state):
        iterator = cls(
            file_path=state["file_path"],
            rank=state.get("rank", 0),
            world_size=state.get("world_size", 1),
            worker_id=state["worker_id"],
            num_workers=state["num_workers"],
            preprocess_dir=state["preprocess_dir"],
            entropy_model_name=state["entropy_model_name"],
            arrow_batch_size=state["arrow_batch_size"],
            dataset_files=state["dataset_files"],
            file_format=state["file_format"],
            shuffle=state.get("shuffle", False),
            seed=state.get("seed", 42),
        )
        if state["row_num"] != 0:
            iterator.set_position(state["row_num"])
        return iterator

    def create_iter(self) -> Generator[BltExample, Any, None]:
        if self.batch_to_consume is not None:
            yield self.batch_to_consume
            self.batch_to_consume = None
            self.current_row_in_batch = 0
        
        if self.shuffle:
            # Use pre-shuffled batch indices for this worker
            for batch_idx in self.batch_indices[self.current_batch_idx:]:
                self.current_batch_idx += 1
                yield self.reader.get_batch(batch_idx)
        else:
            # Original behavior: each worker gets every num_workers-th batch
            for rg in range(self.worker_id, self.reader.num_record_batches, self.num_workers):
                self.current_batch_idx += 1
                yield self.reader.get_batch(rg)

            # --- Tokenization (

    def set_position(self, target_row_num: int):
        # Not implemented for shuffle mode as it would break the shuffle order
        if self.shuffle:
            raise NotImplementedError("set_position is not supported when shuffle=True")
            
        data_str = maybe_truncate_string(str(self.shard_files), 200)
        logger.info(f"Setting arrow position to {target_row_num} for {data_str}")

        if target_row_num == 0:
            self.row_num = 0
            self.dataset = None
            self.batch_iterator = None
            self.batch_to_consume = None
            return

        self.dataset = pa.dataset.dataset(
            self.shard_files, format=self.file_format
        )
        self.batch_iterator = self.dataset.to_batches(batch_size=self.arrow_batch_size)

        curr_remaining = target_row_num
        for batch in self.batch_iterator:
            if len(batch) > curr_remaining:
                batch_columns = batch.to_pydict()
                if self.file_format == "arrow":
                    leftover_sample_ids = batch_columns["sample_id"][curr_remaining:]
                    leftover_entropies = batch_columns["entropies"][curr_remaining:]
                    leftover_texts = batch_columns["text"][curr_remaining:]
                elif self.file_format == "json":
                    leftover_sample_ids = batch_columns[get_id_key(batch_columns)][curr_remaining:]
                    leftover_entropies = None
                    leftover_texts = get_text(batch_columns)[curr_remaining:]
                else:
                    raise ValueError(f"Unknown file format: {self.file_format}")

                self.batch_to_consume = {
                    "sample_id": leftover_sample_ids,
                    "entropies": leftover_entropies,
                    "text": leftover_texts
                }
                break
            elif len(batch) == curr_remaining:
                break
            else:
                curr_remaining -= len(batch)

        self.row_num = target_row_num
        logger.info(f"Finished setting arrow position to {target_row_num} for {data_str}")

    def __iter__(self):
        return self.create_iter()
