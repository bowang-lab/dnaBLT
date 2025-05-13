import pyarrow as pa
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Generator, Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
import logging # Use logging for warnings

from patching import PatcherArgs, Patcher, PatchingModeEnum
from dataclasses import dataclass

log = logging.getLogger(__name__)
EOS_ID = 2

@dataclass
class BltExample:
    model_config = ConfigDict(extra="forbid")
    # torch tensor type not defined for pydantic config
    tokens: Any | None
    patch_lengths: Any | None
    # mask: list[bool] | None

class PreprocessIterator:
    """
    Takes an iterator yielding pyarrow.RecordBatch, tokenizes, patches (if enabled),
    and yields batches of processed data as dictionaries containing lists of
    variable-length sequences.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        arrow_batch_iterator: Generator[pa.RecordBatch, Any, None],
        patcher_args: PatcherArgs,
        # tokenizer_args: BltTokenizerArgs,
        add_patches: bool = True,
    ):
        self.arrow_batch_iterator = arrow_batch_iterator
        self.add_patches = add_patches
        # self.tokenizer = tokenizer_args.build()

        if self.add_patches:
            self.patcher = patcher_args.build()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if hasattr(self.patcher, 'to'):
                 self.patcher.to(self.device)
            # Basic check for entropy mode if entropies are expected
            # assert self.patcher.patching_mode == PatchingModeEnum.entropy
        else:
             self.patcher = None
             self.device = torch.device("cpu")


    def create_iter(self) -> Generator[Dict[str, Any], Any, None]:
        for arrow_batch in self.arrow_batch_iterator.create_iter():
            texts = arrow_batch.column("text").to_pylist()
            entropies_list = arrow_batch.column("entropies").to_pylist()
            batch_size = len(texts)
            if batch_size == 0:
                continue

            # --- Tokenization (Common Step) ---
            tokenized = [torch.cat((torch.frombuffer(bytearray(s.encode("utf-8", errors="ignore")), dtype=torch.uint8).long() + 4, torch.tensor([EOS_ID]))) for s in texts]
            tokens = pad_sequence(tokenized, batch_first=True, padding_value=0)
            # True (non‑padding) length of each sequence
            seq_lengths = (tokens != 0).sum(dim=1)
            entropies = pad_sequence([torch.tensor(e, dtype=torch.float32) for e in entropies_list], batch_first=True, padding_value=0)
            # --- Prepare Output Lists ---
            include_next_token = False
            patch_start_ids = find_entropy_patch_start_ids(
                entropies, include_next_token=include_next_token,
                threshold=self.patcher.threshold, seq_lengths=seq_lengths,
            )
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_lengths
            )

            patch_lengths = split_large_numbers_tensor(
                patch_lengths, max_patch=self.patcher.max_patch_length
            )

            last_non_zero_col_reversed = ((patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min())
            patch_lengths = patch_lengths[:, : patch_lengths.shape[1] - last_non_zero_col_reversed]


            # --- Yield Batch Dictionary ---
            # Only yield if there's valid processed data
            yield BltExample(
                tokens=tokens,
                patch_lengths=patch_lengths
            )

            # TODO: Make patch packing + batch truncation work with new BltExample object


    def __iter__(self):
        return self.create_iter()


# Vectorized patching

def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full((bs, trunc_seq_len), trunc_seq_len, dtype=torch.long, device=patch_start_mask.device)
    else:
        patch_ids = torch.arange(trunc_seq_len, device=patch_start_mask.device).unsqueeze(0).repeat(bs, 1)
        extra_patch_ids = torch.full((bs, trunc_seq_len), trunc_seq_len + 1, dtype=torch.long, device=patch_start_mask.device)
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat((patch_start_mask, ~patch_start_mask), dim=1)
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(bs, trunc_seq_len)[:, :max_patches]
    return patch_start_ids

def find_entropy_patch_start_ids(entropies, seq_lengths, threshold=None, include_next_token=True):
    """
    Uses entropies to compute patch start IDs. If threshold is provided, patches are defined incrementally.
    Otherwise, a fixed number of patches (derived from patch_size) is used.
    """
    bs, seq_len = entropies.shape[:2]
    first_ids = torch.tensor([0, 1], dtype=torch.long, device=entropies.device).unsqueeze(0).repeat(bs, 1)
    preds_truncation_len = first_ids.shape[1]
    entropies = entropies[:, 1:]
    patch_start_mask = entropies > threshold
    if not include_next_token:
        arange = torch.arange(seq_len - 1).expand(bs, seq_len - 1)
        mask = arange != (seq_lengths - 3).unsqueeze(1)
        patch_start_mask = patch_start_mask[mask].reshape(bs, seq_len - 2)
        # patch_start_mask = patch_start_mask[:, :-1]
    patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
    patch_start_ids = torch.cat((first_ids, patch_start_ids + preds_truncation_len), dim=1)
    return patch_start_ids

# ---------------------------------------------------------------------
# 2. Patch-start indices → patch lengths
# ---------------------------------------------------------------------
def patch_lengths_from_start_ids(patch_start_ids, seq_lens):
    """
    Compute patch lengths given patch start IDs, ensuring that each row’s
    lengths sum to the number of non-padding tokens and never go negative.
    """
    # Ensure everything is on the same device
    seq_lens = seq_lens.to(patch_start_ids.device)              # [B]
    seq_lens_row = seq_lens.unsqueeze(1)                        # [B,1]

    # Next patch start (or seq_len if this is the last patch)
    next_start = torch.cat((patch_start_ids[:, 1:], seq_lens_row), dim=1)

    # End idx is min(next_start-1, last real token)
    end_ids = torch.minimum(next_start - 1, seq_lens_row - 1)

    # Raw lengths
    lengths = end_ids - patch_start_ids + 1

    # Zero-out any positions where the start is already past the sequence end
    valid_mask = patch_start_ids < seq_lens_row
    lengths = torch.where(valid_mask, lengths, torch.zeros_like(lengths))

    # Extra safety in case clipping still left negatives
    # lengths.clamp_min_(0)

    return lengths


# ---------------------------------------------------------------------
# 3. Split every “long” patch into chunks ≤ max_patch
# ---------------------------------------------------------------------
def split_large_numbers_tensor(
        lengths:   torch.Tensor,   # [B, N], 0-padded
        max_patch: int
) -> torch.Tensor:                 # [B, N′], 0-padded
    """
    Vectorised “split large numbers” without any invalid reshapes.

    Any element > max_patch is expanded into ⌈len/max_patch⌉ chunks of
    size max_patch, with the last chunk carrying the remainder.
    """
    B, N   = lengths.shape
    device = lengths.device
    dtype  = lengths.dtype

    # ---------------------------------------------------------------
    # (1) flatten, but keep the *row* each element belongs to
    flat      = lengths.view(-1)                          # [B*N]
    keep_mask = flat > 0
    flat_pos  = flat[keep_mask]                           # only >0 values

    row_idx = torch.arange(B, device=device).repeat_interleave(N)
    row_idx = row_idx[keep_mask]                          # same length as flat_pos

    # ---------------------------------------------------------------
    # (2) how many chunks will each positive value expand to?
    n_chunks = torch.div(flat_pos + max_patch - 1,
                         max_patch,
                         rounding_mode='floor')            # ceil(len/max_patch)

    # total chunks per *row*
    per_row_chunks = torch.zeros(B, dtype=torch.long, device=device)
    per_row_chunks.index_add_(0, row_idx, n_chunks)       # sum by row → [B]
    max_chunks = int(per_row_chunks.max().item())

    # ---------------------------------------------------------------
    # (3) build one long ‘expanded’ vector of actual chunk sizes
    total_chunks = int(n_chunks.sum().item())              # scalar

    expanded = torch.full((total_chunks,),
                          max_patch,
                          dtype=dtype,
                          device=device)                  # start with all 16s

    # last element of every group might be a remainder
    remainder = flat_pos % max_patch
    has_rem   = remainder != 0
    last_idx  = torch.cumsum(n_chunks, 0) - 1             # idx of last-chunk per group
    expanded[last_idx[has_rem]] = remainder[has_rem]

    # ---------------------------------------------------------------
    # (4) scatter ‘expanded’ back into a [B, max_chunks] tensor
    out = torch.zeros(B, max_chunks, dtype=dtype, device=device)

    cursor = 0
    for b in range(B):
        cnt = per_row_chunks[b].item()
        if cnt:                                           # skip empty rows
            out[b, :cnt] = expanded[cursor: cursor + cnt]
            cursor += cnt

    return out