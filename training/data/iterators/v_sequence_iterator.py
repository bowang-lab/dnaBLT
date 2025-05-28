from __future__ import annotations

import torch
from typing import Generator, Iterable, List
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass
import numpy as np

# --------------------------------------------------------------------------- #
#                              Output dataclass                               #
# --------------------------------------------------------------------------- #

@dataclass
class PackedSequence():
    """A single packed sequence carrying exactly `MAX_SEQLEN` patches."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    tokens: torch.Tensor
    patch_lengths: torch.Tensor

# --------------------------------------------------------------------------- #
#                               Main iterator                                 #
# --------------------------------------------------------------------------- #
class SequenceIterator:
    """
    Consumes `BltExample` batches and emits `PackedSequence`s with *exactly*
    `MAX_SEQLEN` patches each, suitable for efficient model training.
    """

    def __init__(
        self,
        preprocess_iterator: Iterable,  # Generator[BltExample]
        max_seq_patches: tuple[int, int],
        rng_state = None,
    ) -> None:
        self._src_iter = preprocess_iterator
        # max_seq_pathches[0] is seq_len, max_seq_pathches[1] is buffer_size
        self._max_seq = int(max_seq_patches[0] * max_seq_patches[1])
        self.seqlen = max_seq_patches[0]

        # Use plain lists and a cursor index instead of deques
        self._token_buffer: torch.Tensor | None = None
        self._patch_lengths: List[int] = []

        if rng_state is None:
            self.rng = None
        else:
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = rng_state

    # --------------------------------------------------------------------- #
    #                           Public iterator                              #
    # --------------------------------------------------------------------- #
    def create_iter(self) -> Generator[PackedSequence, None, None]:
        max_seq  = self._max_seq
        token_buf = self._token_buffer
        lens_buf = self._patch_lengths
        for example in self._src_iter.create_iter():
            tk_batch: torch.Tensor = example.tokens                  # [B, T]
            pl_batch: torch.Tensor = example.patch_lengths           # [B, P]

            # -------- Vectorised flatten of the whole batch -------- #
            # 1. Optional row‑level permutation for better mixing.
            if self.rng is not None:
                perm = torch.as_tensor(
                    self.rng.permutation(len(tk_batch)),
                    dtype=torch.long,
                )
                tk_batch = tk_batch.index_select(0, perm)
                pl_batch = pl_batch.index_select(0, perm)

            # 2. Extract patch lengths, discarding any zero‑length slots in a single call.
            valid_mask   = pl_batch.ne(0)
            flat_lengths = pl_batch[valid_mask]          # 1‑D tensor of all patch lengths
            if flat_lengths.numel() == 0:
                continue  # the whole batch was padding

            # 3. Trim away right‑padding tokens *vectorially*.
            row_token_totals = pl_batch.sum(dim=1)       # total real tokens per row
            T = tk_batch.size(1)
            arange_t = torch.arange(T).expand_as(tk_batch)
            keep_mask = arange_t < row_token_totals.unsqueeze(1)
            flat_tokens = tk_batch[keep_mask]            # flattened stream of real tokens

            # 4. Split the flattened token stream into individual patches. Each split
            #    is executed in highly‑optimised C++.
            if token_buf is None:
                token_buf = flat_tokens.clone()
            else:
                token_buf = torch.cat((token_buf, flat_tokens), dim=0)

            lens_buf.extend(flat_lengths.tolist())

            while len(lens_buf) >= max_seq:
                seq_lens = lens_buf[:max_seq]
                seq_len_sum = int(sum(seq_lens))
                seq_toks = token_buf[:seq_len_sum]

                yield PackedSequence(
                    tokens=seq_toks,
                    patch_lengths=torch.tensor(seq_lens, dtype=torch.long),
                )

                lens_buf = lens_buf[max_seq:]
                token_buf = token_buf[seq_len_sum:]

        # -------- Final flush so *no tokens are ever dropped* -------- #
        remaining = len(lens_buf) // self.seqlen
        if remaining > 0:
            # Copy remaining patch‑lengths and right‑pad with zeros to MAX_SEQLEN
            seq_count = remaining * self.seqlen
            seq_lens = lens_buf[:seq_count]
            seq_len_sum = int(sum(seq_lens))
            seq_toks = token_buf[:seq_len_sum]

            yield PackedSequence(
                tokens=seq_toks,
                patch_lengths=torch.tensor(seq_lens, dtype=torch.long),
            )
        
        self._token_buffer = token_buf
        self._patch_lengths = lens_buf

    # ------------------------------ Utilities ---------------------------- #
    def __len__(self) -> int:  # optional, not strictly required
        raise TypeError("SequenceIterator does not support len().")
    
    def __iter__(self):
        """Iterate over the packed sequences."""
        return self.create_iter()
