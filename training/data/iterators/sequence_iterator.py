from typing import Generator, List, Optional, Any
import numpy as np
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict
from preprocess_iterator import PreprocessIterator

class BltSequence(BaseModel):
    tokens: list[int]
    mask: list[bool]
    patch_lengths: list[int] | None

    
class SequencePackingArgs:
    """Configuration for sequence packing parameters."""
    def __init__(self, output_seq_len: int, buffer_size: int):
        self.output_seq_len = output_seq_len
        self.buffer_size = buffer_size




class SequenceIterator:
    """Iterator that packs sequences into fixed-length batches."""
    
    def __init__(
        self,
        preprocess_iterator: PreprocessIterator,
        *,
        sequence_packing_config: SequencePackingArgs,
        seed: Optional[int] = None,
    ):
        """Initialize the sequence iterator.
        
        Args:
            preprocess_iterator: Iterator providing preprocessed examples
            sequence_packing_config: Configuration for sequence packing
            seed: Random seed for shuffling, or None to disable shuffling
        """
        self.preprocess_iterator = preprocess_iterator
        self.config = sequence_packing_config
        self.output_seq_len = sequence_packing_config.output_seq_len
        self.buffer_size = sequence_packing_config.buffer_size
        
        # Initialize random number generator if seed is provided
        self.rng = np.random.default_rng(seed) if seed is not None else None

    def create_iter(self) -> Generator[BltSequence, None, None]:
        """Create an iterator that yields packed sequences."""
        example_iter = self.preprocess_iterator.create_iter()
        n_buffer_patches = self.buffer_size * self.output_seq_len

        patch_lengths: List[int] = []
        tokens: List[int] = []
        mask: List[bool] = []
        
        for example in example_iter:
            # Validate example
            assert example.tokens is not None and len(example.tokens) > 0
            assert example.mask is not None and len(example.mask) > 0
            assert len(example.tokens) == len(example.mask)
            
            if self.preprocess_iterator.add_patches:
                assert example.patch_lengths is not None
                assert len(example.tokens) == sum(example.patch_lengths)
            
            # Add example data to buffers
            tokens.extend(example.tokens)
            mask.extend(example.mask)
            
            if self.preprocess_iterator.add_patches:
                patch_lengths.extend(example.patch_lengths)
            else:
                # Use uniform patch length of 1 if no patches provided
                patch_lengths.extend([1] * len(example.tokens))

            # Process full buffers
            while len(patch_lengths) >= n_buffer_patches:
                # Reshape patch lengths to form batches
                x_patches = np.array(patch_lengths[:n_buffer_patches]).reshape(
                    self.buffer_size, self.output_seq_len
                )
                
                # Extract token sequences based on patch lengths
                seq_tokens = []
                seq_mask = []
                start_id = 0
                
                for num_tokens in x_patches.sum(axis=-1):
                    seq_tokens.append(tokens[start_id : start_id + num_tokens])
                    seq_mask.append(mask[start_id : start_id + num_tokens])
                    start_id += num_tokens

                # Verify we used the expected number of tokens
                assert start_id == x_patches.sum()

                # Remove processed items from buffers
                patch_lengths = patch_lengths[n_buffer_patches:]
                tokens = tokens[start_id:]
                mask = mask[start_id:]

                # Convert to list for easier processing
                seq_patch_lengths: List[List[int]] = x_patches.tolist()
                
                # Determine sequence order (random or sequential)
                if self.rng is None:
                    permutations = list(range(len(seq_patch_lengths)))
                else:
                    permutations = self.rng.permutation(len(seq_patch_lengths))

                # Yield sequences in determined order
                for idx in permutations:
                    # Verify sequence integrity
                    assert len(seq_patch_lengths[idx]) == self.output_seq_len
                    assert sum(seq_patch_lengths[idx]) == len(seq_tokens[idx]) == len(seq_mask[idx])
                    assert seq_patch_lengths[idx][0] > 0
                    
                    # Yield sequence with or without patch lengths
                    if self.preprocess_iterator.add_patches:
                        yield BltSequence(
                            tokens=seq_tokens[idx],
                            mask=seq_mask[idx],
                            patch_lengths=seq_patch_lengths[idx],
                        )
                    else:
                        yield BltSequence(
                            tokens=seq_tokens[idx],
                            mask=seq_mask[idx],
                            patch_lengths=None,
                        )