# Proposed change for /Users/arnavshah/Code/dnaBLT/training/data/iterators/sequence_iterator.py

import numpy as np
from typing import Generator, List, Dict, Any # Added Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import logging

# Keep SequenceIteratorOutput definition
class SequenceIteratorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    tokens: List[np.ndarray] = Field(..., description="List of token sequences")
    mask: List[np.ndarray] = Field(..., description="List of mask sequences")
    patch_lengths_unpacked: List[np.ndarray] = Field(
        ..., description="List of unpacked patch length sequences"
    )

log = logging.getLogger(__name__)

class SequenceIterator:
    """
    Iterator that sequences variable-length examples from a base iterator
    into fixed-length sequences based on patch lengths. Consumes dictionaries
    yielded by PreprocessIterator.
    """
    def __init__(
        self,
        preprocess_iterator: Generator[Dict[str, List[Any]], Any, None], # Updated type hint
        output_seq_len: int,
        buffer_size: int = 1, # Keep defaults
        add_patches: bool = True, # Pass this explicitly now
    ):
        """
        Args:
            preprocess_iterator: Iterator yielding dictionaries like
                                 {"tokens": List[List[int]], "mask": List[List[bool]],
                                  "patch_lengths": List[List[int]]}
            output_seq_len: The desired fixed sequence length (in patches).
            buffer_size: Number of sequences to buffer before yielding a batch.
            add_patches: Whether the input data includes patch lengths.
        """
        self.preprocess_iterator = preprocess_iterator
        self.output_seq_len = output_seq_len
        self.buffer_size = buffer_size
        self.add_patches = add_patches # Store this

        # Internal buffers remain the same
        self.tokens: List[int] = []
        self.mask: List[bool] = []
        self.patch_lengths: List[int] = []

    def __iter__(self) -> Generator[SequenceIteratorOutput, Any, None]:
        """
        Yields batches of fixed-length sequences.
        """
        example_iter = self.preprocess_iterator
        n_buffer_patches = self.buffer_size * self.output_seq_len

        # Local buffers for accumulating before extending self buffers
        # (Can help slightly with performance by reducing list extend calls)
        tokens_buffer: List[int] = []
        mask_buffer: List[bool] = []
        patch_lengths_buffer: List[int] = []

        # --- Main Loop Consuming Batch Dictionaries ---
        for batch_dict in example_iter:
            batch_tokens = batch_dict.get("tokens", [])
            batch_mask = batch_dict.get("mask", [])
            batch_patch_lengths = batch_dict.get("patch_lengths", [])

            # Iterate through samples within the batch dictionary
            num_samples_in_batch = len(batch_tokens)
            for i in range(num_samples_in_batch):
                sample_tokens = batch_tokens[i]
                sample_mask = batch_mask[i]
                sample_patch_lengths = batch_patch_lengths[i]

                # --- Basic Checks (Optional, as PreprocessIterator validates) ---
                if not sample_tokens: continue # Skip empty samples

                # --- Extend local buffers ---
                tokens_buffer.extend(sample_tokens)
                mask_buffer.extend(sample_mask)
                patch_lengths_buffer.extend(sample_patch_lengths)


            # --- Process Buffers when Full ---
            # Move data from local buffers to main buffers
            self.tokens.extend(tokens_buffer)
            self.mask.extend(mask_buffer)
            self.patch_lengths.extend(patch_lengths_buffer)
            tokens_buffer.clear()
            mask_buffer.clear()
            patch_lengths_buffer.clear()

            # --- Chunking Logic (Remains the same) ---
            while len(self.patch_lengths) >= n_buffer_patches:
                # Reshape patch lengths
                try:
                    x_patches = np.array(self.patch_lengths[:n_buffer_patches]).reshape(
                        self.buffer_size, self.output_seq_len
                    )
                except ValueError as e:
                     log.error(f"Error reshaping patch lengths: {e}. Buffer size: {self.buffer_size}, Seq Len: {self.output_seq_len}, Patches available: {len(self.patch_lengths)}. Skipping this chunk.")
                     # Consume the problematic chunk to potentially recover
                     problematic_len = min(n_buffer_patches, len(self.patch_lengths))
                     num_tokens_to_discard = sum(self.patch_lengths[:problematic_len])
                     self.tokens = self.tokens[num_tokens_to_discard:]
                     self.mask = self.mask[num_tokens_to_discard:]
                     self.patch_lengths = self.patch_lengths[problematic_len:]
                     continue


                # Extract token sequences based on patch lengths
                seq_tokens = []
                seq_mask = []
                current_token_idx = 0
                for b in range(self.buffer_size):
                    num_tokens_in_seq = x_patches[b].sum()
                    seq_end_idx = current_token_idx + num_tokens_in_seq
                    seq_tokens.append(np.array(self.tokens[current_token_idx:seq_end_idx]))
                    seq_mask.append(np.array(self.mask[current_token_idx:seq_end_idx]))
                    current_token_idx = seq_end_idx


                # Yield the batch
                yield SequenceIteratorOutput(
                    tokens=seq_tokens,
                    mask=seq_mask,
                    patch_lengths_unpacked=list(x_patches), # Convert reshaped numpy array rows to list of arrays
                )

                # Update main buffers by removing processed data
                self.tokens = self.tokens[current_token_idx:]
                self.mask = self.mask[current_token_idx:]
                self.patch_lengths = self.patch_lengths[n_buffer_patches:]

        # Log remaining data if any (optional)
        if self.tokens:
            log.debug(f"SequenceIterator finished. Discarding {len(self.tokens)} remaining tokens.")


    def get_state(self):
        # TODO: Implement state saving/loading if checkpointing is needed
        # Needs to save internal buffers (tokens, mask, patch_lengths)
        # and potentially the state of the underlying preprocess_iterator if it's stateful.
        raise NotImplementedError("State saving/loading not implemented for SequenceIterator")

    @classmethod
    def from_state(cls, state):
        raise NotImplementedError("State saving/loading not implemented for SequenceIterator")
