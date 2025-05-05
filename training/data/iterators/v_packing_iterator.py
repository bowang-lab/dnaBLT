# Copyright (c) Meta Platforms, Inc. and affiliates.
from enum import Enum
from dataclasses import dataclass
from typing import Any, Generator, List, Optional
from pydantic import BaseModel
from sequence_iterator import SequenceIterator
import torch

import numpy as np


class PackingMode(str, Enum):
    BYTES = "bytes"
    PATCHING = "patching"


class PackingArgs:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        pad_id: int,
        max_length: Optional[int],
        pad_to_max_length: bool,
        enable_byte_ngrams: bool,
        packing_mode: PackingMode,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.enable_byte_ngrams = enable_byte_ngrams
        self.packing_mode = packing_mode


class BltSequence(BaseModel):
    tokens: list[int]
    mask: list[bool]
    patch_lengths: list[int] | None


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray | None = None
    mask: np.ndarray | None = None
    patch_lengths: np.ndarray | None = None
    ngram_ids: np.ndarray | None = None
    is_final: bool = False

    def to_python_dict(self) -> dict:
        x = self.x.tolist()
        y = self.y.tolist()
        if self.mask is None:
            mask = None
        else:
            mask = self.mask.tolist()
        if self.patch_lengths is None:
            patch_lengths = None
        else:
            patch_lengths = self.patch_lengths.tolist()
        if self.ngram_ids is None:
            ngram_ids = None
        else:
            ngram_ids = self.ngram_ids.tolist()
        return {
            "x": x,
            "y": y,
            "mask": mask,
            "patch_lengths": patch_lengths,
            "ngram_ids": ngram_ids,
            "is_final": self.is_final,
        }

    @classmethod
    def from_python_dict(cls, data: dict) -> "Batch":
        x = np.array(data["x"])
        y = np.array(data["y"])
        if data["mask"] is None:
            mask = None
        else:
            mask = np.array(data["mask"])
        if data["patch_lengths"] is None:
            patch_lengths = None
        else:
            patch_lengths = np.array(data["patch_lengths"])
        if data["ngram_ids"] is None:
            ngram_ids = None
        else:
            ngram_ids = np.array(data["ngram_ids"])
        return Batch(
            x=x,
            y=y,
            mask=mask,
            patch_lengths=patch_lengths,
            ngram_ids=ngram_ids,
            is_final=data["is_final"],
        )


def _merge_patch_seq_masks(bs: int, slen: int, mask_seqs: List[List[bool]]):
    assert len(mask_seqs) == bs
    lens = [len(m) for m in mask_seqs]
    if all(all(m) for m in mask_seqs) and all(lens[0] == l for l in lens):
        return np.ones((bs, slen), dtype=bool)
    assert slen == max(lens) - 1, f"slen={slen} != max(lens)-1={max(lens) - 1}"
    mask = np.zeros((bs, slen), dtype=bool)
    for i, m in enumerate(mask_seqs):
        if m is None:
            print(
                "Did not implement None mask, the mask should be True for all toks, so we need to pass that to this function."
            )
            raise NotImplementedError
        mask[i][: len(mask_seqs[i]) - 1] = mask_seqs[i][1:]
    return mask


class PackingIterator:
    def __init__(
        self,
        sequence_iterator: SequenceIterator,
        packing_args: PackingArgs,
    ):
        """
        Initialize a packing iterator that batches sequences according to packing mode.

        Args:
            sequences: List of sequences to be packed
            packing_args: Configuration for packing
        """
        self.sequence_iterator = sequence_iterator
        self.packing_args = packing_args
        self.current_idx = 0

    def __iter__(self) -> Generator[Batch, Any, None]:
        """Return an iterator over batches"""
        if self.packing_args.packing_mode == PackingMode.BYTES:
            yield from self._create_iter_from_bytes()
        elif self.packing_args.packing_mode == PackingMode.PATCHING:
            yield from self._create_iter_from_patch_lengths()
        else:
            raise ValueError(f"Invalid patching mode: {self.packing_args.packing_mode}")

    """
    For the next couple iterations, in the original implementation, they used an Iterator that was defined, but we want to make this simpler. As a result, the work around for this was to 
    replicate the same functionality via a more specific while loop condition. 

    In the original code, there was a While True loop, but in this one, what we are doing is manually checking if the index has surpassed the length of the sequence.

    """

    def _create_iter_from_bytes(self):
        sequence_iter = self.sequence_iterator
        batch_size = self.packing_args.batch_size
        pad_id = self.packing_args.pad_id
        seq_len = self.packing_args.seq_len

        while True:
            tokens: List[List[int]] = []
            masks: List[List[bool]] = []
            stop_iteration = False
            try:
                # Collect sequences for the batch
                for _ in range(batch_size):
                    sequence = next(sequence_iter)

                    _tokens = sequence.tokens
                    _mask = sequence.mask
                    assert (
                        sequence.patch_lengths is None
                    ), "patch_lengths should not be used in byte packing"

                    tokens.append(_tokens)
                    masks.append(_mask)
            except StopIteration:
                stop_iteration = True

            # Create batch arrays with appropriate padding
            x = np.full((batch_size, seq_len), fill_value=pad_id)
            y = np.full((batch_size, seq_len), fill_value=pad_id)
            m = np.zeros((batch_size, seq_len), dtype=np.bool_)

            for i, tok_seq in enumerate(tokens):
                x[i, : len(tok_seq)] = tok_seq
                y[i, : len(tok_seq) - 1] = tok_seq[1:]
                m[i, : len(tok_seq)] = masks[i]

            batch = Batch(x=x, y=y, mask=m)
            assert (
                batch.mask is None or np.sum(x != pad_id) == batch.mask.sum()
            ), f"{np.sum(x != pad_id)} != {batch.mask.sum()}"

            yield batch

            if stop_iteration:
                break

    def _create_iter_from_patch_lengths(self):
        sequence_iter = self.sequence_iterator
        batch_size = self.packing_args.batch_size
        pad_id = self.packing_args.pad_id
        seq_len = self.packing_args.seq_len
        pad_to_max_length = self.packing_args.pad_to_max_length
        max_length = self.packing_args.max_length
        assert (
            max_length is not None
        ), "max_length must be provided for patch-based packing"
        running_toks = torch.tensor([])
        running_patch_lengths = torch.tensor([])

        for batch in sequence_iter:
            tokens, patch_lengths = truncate_and_rectangularise(
                batch,
                max_length=max_length,
                pad_id=pad_id,
                patches_per_seq=seq_len+1,
                pad_to_max_length=pad_to_max_length,
            )

            tokens = torch.cat((running_toks, tokens), dim=0)
            patch_lengths = torch.cat((running_patch_lengths, patch_lengths), dim=0)

            while tokens.shape[0] >= batch_size: # assumes fetch size geq batch size (valid)
                yield Batch(
                    x=tokens[:batch_size],
                    patch_lengths=patch_lengths[:batch_size]
                )  # y and mask
                tokens = tokens[batch_size:]
                patch_lengths = patch_lengths[batch_size:]
            
            running_toks = tokens
            running_patch_lengths = patch_lengths
        
        if running_toks.shape[0] > 0:
            yield Batch(x=running_toks, patch_lengths=running_patch_lengths)


def truncate_and_rectangularise(
    batch,
    *,
    max_length: int,  # tokens, *not* counting EOS
    pad_id: int,
    patches_per_seq: int,  # fixed header size  (M)
    pad_to_max_length: bool = False,
):
    """
    • Edits `batch.patch_lengths` *in-place* so each header row sums to
      `max_length+1` (EOS still counted).
    • Returns new 2-D tensors (`tokens2d`, plus `y2d` / `mask2d` if present),
      padded on the right with `pad_id` (and mask=False) as needed.
    """
    dev = batch.tokens.device
    pl = batch.patch_lengths.view(-1, patches_per_seq)  # [B, M]
    B, M = pl.shape

    # ---------- per-sequence lengths ---------------------------------
    row_len_hdr = pl.sum(1)  # [B]  incl. EOS
    row_len_tok = row_len_hdr - 1  # [B]  excl. EOS
    # Preserve the original (pre‑truncation / pre‑padding) token counts
    # so we can later split the flat `batch.tokens` buffer safely.
    row_len_tok_orig = row_len_tok.clone()
    max_hdr = max_length + 1
    rows_exceed = row_len_hdr > max_hdr

    # ---------- (1) fix headers, vectorised --------------------------
    if rows_exceed.any():
        cumsum = pl.cumsum(1)  # [B, M]
        idx_ex = (cumsum > max_hdr).int().argmax(1)
        idx_prev = torch.clamp(idx_ex - 1, min=0)
        count_prev = cumsum.gather(1, idx_prev[:, None]).squeeze(1)
        count_prev = torch.where(idx_ex == 0, torch.zeros_like(count_prev), count_prev)
        new_len = max_hdr - count_prev

        m_axis = torch.arange(M, device=dev)
        kill_mask = (m_axis[None, :] > idx_ex[:, None]) & rows_exceed[:, None]
        pl.masked_fill_(kill_mask, 0)
        pl[rows_exceed, idx_ex[rows_exceed]] = new_len[rows_exceed]
        row_len_tok = torch.where(
            rows_exceed, torch.full_like(row_len_tok, max_length), row_len_tok
        )

    # ---------- (2) optional right-pad to max_length -----------------
    if pad_to_max_length:
        need_pad = row_len_tok < max_length
        if need_pad.any():
            diff = max_length - row_len_tok  # [B]
            last_idx = (pl != 0).sum(1) - 1
            pl[torch.arange(B, device=dev), last_idx] += diff
            row_len_tok = torch.full_like(row_len_tok, max_length)

    # ---------- (3) build rectangular tensors ------------------------
    width = int(row_len_tok.max().item())  # ≤ max_length
    tokens2d = batch.tokens.new_full((B, width), pad_id)

    # Pre-compute prefix offsets into the *flat* tensors
    src_off = torch.cat(
        [torch.tensor([0], device=dev), (row_len_hdr - 1).cumsum(0)[:-1]]
    )

        # -------- vectorised copy (no Python loop) ----------------------------
    # Build a (B, width) matrix of column indices: 0 .. width‑1  for each row
    idx_mat = torch.arange(width, device=dev).expand(B, width)          # [B, W]

    # Effective number of source tokens to copy per row (may be < row_len_tok
    # when padding, or == row_len_tok when truncation occurred)
    copy_len = torch.minimum(row_len_tok, row_len_tok_orig)             # [B]

    # Boolean mask indicating positions fed by *real* tokens
    valid_mask = idx_mat < copy_len.unsqueeze(1)                         # [B, W]

    # Convert (row, col) → flat offset into the *input* 1‑D `batch.tokens`
    src_flat_idx = src_off.unsqueeze(1) + idx_mat                        # [B, W]

    # To stay in‑bounds for rows that are shorter, clamp indices that would
    # read past the end of the original sequence.  They will be ignored by
    # the `valid_mask` anyway, but clamping avoids undefined behaviour.
    src_flat_idx = torch.where(
        valid_mask, src_flat_idx, torch.zeros_like(src_flat_idx)
    )

    # Gather all source tokens in one go and drop them into `tokens2d`
    gathered = batch.tokens.gather(0, src_flat_idx.view(-1)).view(B, width)
    tokens2d[valid_mask] = gathered[valid_mask]
    # positions where `valid_mask` is False already contain `pad_id`


    # ---------- copy optional companions the same way ----------------
    # y2d, mask2d = None, None
    # if batch.y is not None:
    #     y2d = batch.y.new_full((B, width), pad_id)
    #     for b in range(B):
    #         n = row_len_tok[b].item()
    #         y2d[b, :n] = batch.y[src_off[b]: src_off[b] + n]

    # if batch.mask is not None:
    #     mask2d = batch.mask.new_full((B, width), False)
    #     for b in range(B):
    #         n = row_len_tok[b].item()
    #         mask2d[b, :n] = batch.mask[src_off[b]: src_off[b] + n]

    # ---------- sanity checks ----------------------------------------
    assert torch.all(max_hdr - pl.sum(1) == 0)
    assert width <= max_length
    if pad_to_max_length:
        assert width == max_length

    # ---------- return everything ------------------------------------
    # return tokens2d, pl, y2d, mask2d
    return tokens2d, pl
