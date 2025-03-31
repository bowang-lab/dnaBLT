import os
import re
from logging import getLogger
from typing import Generator, Any, Optional

import fsspec
import pyarrow as pa
import pyarrow.dataset
import s3fs
from bytelatent import ByteLatentError
from bytelatent.data.data_types import BltExample
from bytelatent.data.file_util import get_fs
from bytelatent.preprocess.preprocess_entropies import get_id_key, get_text

logger = getLogger(__name__)


def shard_sort_key(file: str):
    """Extract shard number from arrow file name for sorting."""
    assert isinstance(file, str)
    match = re.search(r".+\.shard_([0-9]+)\.arrow", file)
    shard_number = int(match.group(1))
    return shard_number


def maybe_truncate_string(text: str, max_length: int):
    """Truncate string if it exceeds max_length."""
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length] + "..."


class ArrowFileIterator:
    def __init__(
        self,
        *,
        file_path: Optional[str] = None,
        worker_id: int = 0,
        num_workers: int = 1,
        preprocess_dir: Optional[str] = None,
        entropy_model_name: Optional[str] = None,
        arrow_batch_size: int = 100,
        dataset_files: Optional[list[str]] = None,
        s3_profile: Optional[str] = None,
        filesystem_type: Optional[str] = None,
        file_format: str = "arrow",
    ):
        """
        Initialize an iterator for Arrow files.
        
        Args:
            file_path: Path to the source file
            worker_id: ID of the current worker for parallelization
            num_workers: Total number of workers for parallelization
            preprocess_dir: Directory containing preprocessed data
            entropy_model_name: Name of the entropy model used
            arrow_batch_size: Number of rows to read per batch
            dataset_files: List of dataset files (alternative to file_path)
            s3_profile: S3 profile name for S3 access
            filesystem_type: Type of filesystem ("file" or "s3")
            file_format: Format of the files ("arrow" or "json")
        """
        assert 0 <= worker_id < num_workers, (worker_id, num_workers)
        if file_path is None and dataset_files is None:
            raise ByteLatentError("file_path and dataset_files cannot both be None")
            
        # Initialize instance variables
        self.row_num = 0
        self.batch_iterator = None
        self.batch_to_consume = None
        self.dataset = None
        
        # Store constructor parameters
        self.file_path = file_path
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.preprocess_dir = preprocess_dir
        self.entropy_model_name = entropy_model_name
        self.arrow_batch_size = arrow_batch_size
        self.s3_profile = s3_profile
        self.filesystem_type = filesystem_type
        self.file_format = file_format
        
        # Initialize filesystem
        self.fs = self._initialize_filesystem(filesystem_type, s3_profile)
        
        # Initialize dataset files
        self.dataset_files = self._initialize_dataset_files(
            file_path, dataset_files, file_format, preprocess_dir, entropy_model_name, s3_profile
        )

    def _initialize_filesystem(self, filesystem_type, s3_profile):
        """Initialize the appropriate filesystem."""
        if filesystem_type is not None:
            if filesystem_type == "file":
                fs = fsspec.filesystem("file")
            elif filesystem_type == "s3":
                fs = fsspec.filesystem("s3", profile=s3_profile)
            else:
                raise ValueError(f"Unknown filesystem type: {filesystem_type}")
            logger.info("Arrow iterator using fs=%s", fs)
            return fs
        return None
    # this was implemented to make it easier to get the dataset files (avoid reusing code)
    def _initialize_dataset_files(
        self, file_path, dataset_files, file_format, preprocess_dir, entropy_model_name, s3_profile
    ):
        """Initialize the list of dataset files."""
        if dataset_files is not None:
            # If dataset_files is provided, use them directly
            if dataset_files[0].startswith("s3://"):
                for f in dataset_files:
                    assert f.startswith("s3://")
            
            if self.fs is None:
                self.fs = get_fs(dataset_files[0], s3_profile=s3_profile)
                if isinstance(self.fs, s3fs.S3FileSystem):
                    self.filesystem_type = "s3"
                else:
                    self.filesystem_type = "file"
            
            return dataset_files
        
        # Otherwise, derive dataset files from file_path
        assert file_path is not None, "Must specify file_path if dataset_files is None"
        
        if file_format == "json":
            # For JSON files, just use the file_path directly
            if self.fs is None:
                self.fs = get_fs(file_path, s3_profile=s3_profile)
                if isinstance(self.fs, s3fs.S3FileSystem):
                    self.filesystem_type = "s3"
                else:
                    self.filesystem_type = "file"
            return [file_path]
        else:
            # For arrow files, find the shards
            return self._find_arrow_shards(file_path, preprocess_dir, entropy_model_name, s3_profile)

   

    def get_state(self):
        """Get the current state of the iterator for checkpointing."""
        return {
            "file_path": self.file_path,
            "row_num": self.row_num,
            "worker_id": self.worker_id,
            "num_workers": self.num_workers,
            "preprocess_dir": self.preprocess_dir,
            "entropy_model_name": self.entropy_model_name,
            "arrow_batch_size": self.arrow_batch_size,
            "dataset_files": self.dataset_files,
            "s3_profile": self.s3_profile,
            "filesystem_type": self.filesystem_type,
            "file_format": self.file_format,
        }

    @classmethod
    def from_state(cls, state):
        """Reconstruct an iterator from a checkpoint state."""
        iterator = cls(
            file_path=state["file_path"],
            worker_id=state["worker_id"],
            num_workers=state["num_workers"],
            preprocess_dir=state["preprocess_dir"],
            entropy_model_name=state["entropy_model_name"],
            arrow_batch_size=state["arrow_batch_size"],
            dataset_files=state["dataset_files"],
            s3_profile=state["s3_profile"],
            filesystem_type=state["filesystem_type"],
            file_format=state["file_format"],
        )
        if state["row_num"] != 0:
            iterator.set_position(state["row_num"])
        return iterator

    def create_iter(self) -> Generator[BltExample, Any, None]:
        """Create an iterator over the dataset."""
        # Initialize dataset if not already done
        if self.dataset is None:
            self._initialize_dataset()
        
        # First yield any rows from a partially consumed batch
        if self.batch_to_consume is not None:
            yield from self._process_batch(self.batch_to_consume)
            self.batch_to_consume = None

        # Then iterate through all remaining batches
        for batch in self.batch_iterator:
            batch_columns = batch.to_pydict()
            yield from self._process_batch(batch_columns)

    def _initialize_dataset(self):
        """Initialize the PyArrow dataset and batch iterator."""
        filesystem = self.fs if isinstance(self.fs, s3fs.core.S3FileSystem) else None
        self.dataset = pa.dataset.dataset(
            self.dataset_files, format=self.file_format, filesystem=filesystem
        )
        self.batch_iterator = self.dataset.to_batches(batch_size=self.arrow_batch_size)

    def _process_batch(self, batch_columns):
        """Process a batch of data and yield examples for this worker."""
        if self.file_format == "arrow":
            sample_ids = batch_columns["sample_id"]
            texts = batch_columns["text"]
            entropies = batch_columns["entropies"]
        elif self.file_format == "json":
            # Handle raw JSON data
            sample_ids = batch_columns[get_id_key(batch_columns)]
            texts = get_text(batch_columns)
            entropies = None
        else:
            raise ValueError(f"Unknown file format: {self.file_format}")
            
        for i in range(len(sample_ids)):
            self.row_num += 1
            # Only process rows assigned to this worker
            if (self.row_num - 1) % self.num_workers == self.worker_id:
                yield BltExample(
                    sample_id=sample_ids[i],
                    entropies=entropies[i] if entropies is not None else None,
                    text=texts[i],
                    tokens=None,
                    mask=None,
                    patch_lengths=None,
                )

    def set_position(self, target_row_num: int):
        """Set the iterator to a specific row position (for resuming)."""
        data_str = maybe_truncate_string(str(self.dataset_files), 200)
        logger.info(f"Setting arrow position to {target_row_num} for {data_str}")
        
        if target_row_num == 0:
            # Reset to beginning
            self.row_num = 0
            self.dataset = None
            self.batch_iterator = None
            self.batch_to_consume = None
            return
            
        # Initialize dataset
        filesystem = self.fs if isinstance(self.fs, s3fs.core.S3FileSystem) else None
        self.dataset = pa.dataset.dataset(
            self.dataset_files, format=self.file_format, filesystem=filesystem
        )
        self.batch_iterator = self.dataset.to_batches(batch_size=self.arrow_batch_size)
        
        # Skip through batches until we reach the target row
        curr_remaining = target_row_num
        for batch in self.batch_iterator:
            if len(batch) > curr_remaining:
                # Target is in the middle of this batch
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
                # We are exactly at the end of the batch
                break
            else:
                # Keep going to the next batch
                curr_remaining -= len(batch)
                
        self.row_num = target_row_num
        logger.info(f"Finished setting arrow position to {target_row_num} for {data_str}")

    def __iter__(self):
        """Make this class directly iterable."""
        return self.create_iter()