from __future__ import annotations

import torch
from typing import Generator, Iterable, List
from pydantic import BaseModel, ConfigDict, Field
from dataclasses import dataclass

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
        max_seq_patches: int = MAX_SEQLEN,
    ) -> None:
        self._src_iter = preprocess_iterator
        self._max_seq = int(max_seq_patches)

        # Use plain lists and a cursor index instead of deques
        self._patch_tokens: List[torch.Tensor] = []
        self._patch_lengths: List[int] = []
        self._cursor: int = 0  # index of the first un‑consumed patch

    # --------------------------------------------------------------------- #
    #                           Public iterator                              #
    # --------------------------------------------------------------------- #
    def __iter__(self) -> Generator[PackedSequence, None, None]:
        max_seq  = self._max_seq
        toks_buf = self._patch_tokens
        lens_buf = self._patch_lengths
        cursor   = self._cursor
        device = torch.device("cpu")   # will be updated on the first real example

        for example in self._src_iter:
            tk_batch: torch.Tensor = example.tokens                  # [B, T]
            pl_batch: torch.Tensor = example.patch_lengths           # [B, P]
            device = tk_batch.device

            # -------- Flatten the batch into a stream of patches -------- #
            for row_tokens, row_pl in zip(tk_batch, pl_batch):
                # Keep patch‑lengths on the **CPU** to avoid a host⇄device sync
                # when we materialise them as Python ints.
                lengths = row_pl.cpu()
                lengths = lengths[lengths.ne(0)]           # drop zero‑length slots
                if lengths.numel() == 0:
                    continue

                # Trim away padding once, then cut into patches in C++
                total_tokens = int(lengths.sum())
                row_tokens = row_tokens[: total_tokens]

                # torch.split_with_sizes does the heavy lifting in a single call
                patches = torch.split(row_tokens, lengths.tolist())

                # Running buffers keep a *patch‑level* alignment
                toks_buf.extend(patches)          # one tensor per patch
                lens_buf.extend(lengths.tolist()) # matching scalar lengths
            
            # -------- Emit full packed sequences as soon as possible ----- #
            while cursor + max_seq <= len(lens_buf):
                end = cursor + max_seq
                seq_lens = lens_buf[cursor:end]            # Python list slice (cheap)
                seq_toks = torch.cat(toks_buf[cursor:end])  # Concatenate once

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
        remaining = len(lens_buf) - cursor
        if remaining:
            seq_lens = lens_buf[cursor:]                 # leftover patch lengths
            pad      = max_seq - remaining               # how many zero‑length slots
            seq_lens += [0] * pad                        # right‑pad to MAX_SEQLEN

            seq_toks = torch.cat(toks_buf[cursor:]) if remaining else torch.empty(0, device=device)

            yield PackedSequence(
                tokens=seq_toks,
                patch_lengths=torch.tensor(seq_lens, device=device, dtype=torch.long),
            )

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
