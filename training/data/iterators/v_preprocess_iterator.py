# Proposed change for /Users/arnavshah/Code/dnaBLT/training/data/iterators/preprocess_iterator.py
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
        for arrow_batch in self.arrow_batch_iterator:
            texts = arrow_batch.column("text").to_pylist()
            entropies_list = arrow_batch.column("entropies").to_pylist()
            batch_size = len(texts)
            if batch_size == 0:
                continue

            # --- Tokenization (Common Step) ---
            tokenized = [torch.cat((torch.frombuffer(bytearray(s.encode("utf-8", errors="ignore")), dtype=torch.uint8).long() + 4, torch.tensor([EOS_ID]))) for s in texts]
            tokens = pad_sequence(tokenized, batch_first=True, padding_value=0)
            entropies = pad_sequence([torch.tensor(e, dtype=torch.float32) for e in entropies_list], batch_first=True, padding_value=0)
            # --- Prepare Output Lists ---
            include_next_token = False
            bs, seq_len = tokens.shape
            seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
            patch_start_ids = find_entropy_patch_start_ids(
                entropies, include_next_tok=include_next_token,
                threshold=self.patcher.threshold
            )
            patch_lengths = patch_lengths_from_start_ids(
                patch_start_ids, seq_len_next_tok
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


    def __iter__(self):
        return self.create_iter()


# Vectorized patching

# return patch_lengths[0], scores

def patch_start_ids_from_patch_start_mask(mask: torch.Tensor) -> torch.LongTensor:
    """
    mask : [B, L′] boolean – True where a patch should start
    returns a tensor [B, P_max] containing the start indices
            (filled with L′ where a row has fewer than P_max patches)
    """
    B, L = mask.shape
    max_patches = int(mask.sum(1).max().item())

    if max_patches == 0:                                  # no patches at all
        return torch.full((B, 1), L, dtype=torch.long, device=mask.device)

    patch_ids = torch.arange(L, device=mask.device).expand(B, -1)     # [B, L]
    sentinel  = torch.full_like(patch_ids, L)                         # pad value
    # boolean concat trick from the original code
    all_ids   = torch.cat([patch_ids, sentinel], dim=1)               # [B, 2L]
    padded_m  = torch.cat([mask, ~mask], dim=1)                       # [B, 2L]

    out = all_ids[padded_m].view(B, L)[:, :max_patches]               # [B, P_max]
    return out

def find_entropy_patch_start_ids(
    entropies:        torch.Tensor,        # [B, L]
    threshold:        float | None = None,
    include_next_tok: bool  = True,
) -> torch.LongTensor:
    """
    Exact batched analogue of your original single-example routine.
    Returns a LongTensor [B, 2+P] whose first two columns are 0 and 1.
    """
    B, L = entropies.shape
    dev   = entropies.device

    # -----------------------------------------------------------------
    # the “always present” first two tokens
    first_two = torch.tensor([0, 1], device=dev).expand(B, 2)  # [B, 2]
    preds_trunc_len = 2

    # all work happens on the truncated view (drop token-0)
    ent = entropies[:, 1:]                                     # [B, L-1]
    L_trunc = L - 1

    patch_mask = ent > threshold                            # [B, L-1]

    if not include_next_tok:
        patch_mask = patch_mask[:, :-1]                         # drop last column

    patch_start_ids = patch_start_ids_from_patch_start_mask(patch_mask)

    # -----------------------------------------------------------------
    # re-add the 0/1 prefix and offset by +2  -------------------------
    patch_start_ids = torch.cat([first_two,
                                 patch_start_ids + preds_trunc_len], dim=1)
    return patch_start_ids  

# ---------------------------------------------------------------------
# 2. Patch-start indices → patch lengths
# ---------------------------------------------------------------------
def patch_lengths_from_start_ids(
        start_ids: torch.Tensor,
        seq_len: int
) -> torch.LongTensor:
    """
    start_ids  : [B, N]  (padded with seq_len)
    returns    : [B, N]  (0-padded)
    """
    # NB: start_ids == seq_len is *padding*, not a real patch
    last_tok = torch.full_like(start_ids[:, :1], seq_len - 1)
    end_ids  = torch.cat([start_ids[:, 1:] - 1, last_tok], dim=1)

    lengths = (end_ids - start_ids + 1).clamp_min(0)
    lengths.masked_fill_(start_ids == seq_len, 0)        # wipe padding rows

    return lengths                                       # [B, N]


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