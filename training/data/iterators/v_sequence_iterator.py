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

        for example in self._src_iter:
            tk_batch: torch.Tensor = example.tokens                  # [B, T]
            pl_batch: torch.Tensor = example.patch_lengths           # [B, P]
            device = tk_batch.device

            # -------- Flatten the batch into a stream of patches -------- #
            for row_tokens, row_pl in zip(tk_batch, pl_batch):
                nz_mask = row_pl != 0
                if not torch.any(nz_mask):
                    continue                                          # empty row

                lengths = row_pl[nz_mask].tolist()                    # List[int]
                n_tok_row = sum(lengths)
                row_tokens = row_tokens[: n_tok_row]                 # trim padding

                # Split tokens by patch boundaries (torch.split is C‑backed)
                for length, patch in zip(lengths, torch.split(row_tokens, lengths)):
                    toks_buf.append(patch)
                    lens_buf.append(length)

            # -------- Emit full packed sequences as soon as possible ----- #
            while cursor + max_seq <= len(lens_buf):
                seq_lens = lens_buf[cursor : cursor + max_seq]
                seq_toks = torch.cat(toks_buf[cursor : cursor + max_seq])

                yield PackedSequence(
                    tokens=seq_toks,
                    patch_lengths=torch.tensor(seq_lens, device=device, dtype=torch.long),
                )
                cursor += max_seq
                self._compact_buffers()

            self._cursor = cursor

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
