from __future__ import annotations

import torch
from typing import Generator, Iterable, List
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass
import numpy as np

MAX_SEQLEN = 4096  # number of *patches* per packed example


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
        rng_state = None,
        max_seq_patches: int = MAX_SEQLEN,
    ) -> None:
        self._src_iter = preprocess_iterator
        self._max_seq = int(max_seq_patches)

        # Use plain lists and a cursor index instead of deques
        self._patch_tokens: List[torch.Tensor] = []
        self._patch_lengths: List[int] = []
        self._cursor: int = 0  # index of the first un‑consumed patch

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
        toks_buf = self._patch_tokens
        lens_buf = self._patch_lengths
        cursor   = self._cursor
        device = torch.device("cpu")   # will be updated on the first real example

        for example in self._src_iter.create_iter():
            tk_batch: torch.Tensor = example.tokens                  # [B, T]
            pl_batch: torch.Tensor = example.patch_lengths           # [B, P]
            device = tk_batch.device

            # -------- Vectorised flatten of the whole batch -------- #
            # 1. Optional row‑level permutation for better mixing.
            if self.rng is not None:
                perm = torch.as_tensor(
                    self.rng.permutation(len(tk_batch)),
                    dtype=torch.long,
                    device=tk_batch.device,
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
            arange_t = torch.arange(T, device=device).expand_as(tk_batch)
            keep_mask = arange_t < row_token_totals.unsqueeze(1)
            flat_tokens = tk_batch[keep_mask]            # flattened stream of real tokens

            # 4. Split the flattened token stream into individual patches. Each split
            #    is executed in highly‑optimised C++.
            patches = torch.split(flat_tokens, flat_lengths.tolist())

            # 5. Append to running buffers that maintain *patch‑level* alignment.
            toks_buf.extend(patches)                     # one tensor per patch
            lens_buf.extend(flat_lengths.tolist())       # matching scalar lengths
            
            while cursor + max_seq <= len(lens_buf):
                end = cursor + max_seq
                seq_lens = lens_buf[cursor:end]            # Python list slice (cheap)
                seq_toks = torch.cat(toks_buf[cursor:end])  # Concatenate once
                print(len(seq_lens) / 4096)

                yield PackedSequence(
                    tokens=seq_toks,
                    patch_lengths=torch.tensor(seq_lens, device=device, dtype=torch.long),
                )
                cursor = end

                # Periodically drop consumed prefix to keep memory bounded
                if cursor > 4 * max_seq and cursor % (4 * max_seq) == 0:
                    del toks_buf[:cursor]
                    del lens_buf[:cursor]
                    cursor = 0

            self._cursor = cursor
            self._compact_buffers()

        # -------- Final flush so *no tokens are ever dropped* -------- #
        # NOTE: This cooks the x_patch_lengths guard in the PackingIterator logic.
        # remaining = len(lens_buf) - cursor
        # if remaining > 0:
        #     seq_lens = lens_buf[cursor:]                 # leftover patch lengths
        #     pad = max_seq - remaining               # how many zero‑length slots
        #     seq_lens += [0] * pad                        # right‑pad to MAX_SEQLEN

        #     seq_toks = torch.cat(toks_buf[cursor:]) if remaining else torch.empty(0, device=device)

        #     yield PackedSequence(
        #         tokens=seq_toks,
        #         patch_lengths=torch.tensor(seq_lens, device=device, dtype=torch.long),
        #     )

    def _compact_buffers(self) -> None:
        """Drop consumed patches when cursor grows large to keep memory bounded."""
        if self._cursor > 4 * self._max_seq:
            # remove consumed prefix
            del self._patch_tokens[: self._cursor]
            del self._patch_lengths[: self._cursor]
            self._cursor = 0

    # ------------------------------ Utilities ---------------------------- #
    def __len__(self) -> int:  # optional, not strictly required
        raise TypeError("SequenceIterator does not support len().")
    
    def __iter__(self):
        """Iterate over the packed sequences."""
        return self.create_iter()
