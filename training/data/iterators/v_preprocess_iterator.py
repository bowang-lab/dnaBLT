# Proposed change for /Users/arnavshah/Code/dnaBLT/training/data/iterators/preprocess_iterator.py
import pyarrow as pa
import torch
from typing import Generator, Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
import numpy as np
import logging # Use logging for warnings

# Assuming these are defined correctly elsewhere
from bytelatent.datasets import BltExample
from bytelatent.models.tokenizers import BltTokenizerArgs
from .patching import PatcherArgs, Patcher, PatchingModeEnum

log = logging.getLogger(__name__)

# Remove pad_sequences helper as padding is deferred

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
        tokenizer_args: BltTokenizerArgs,
        add_patches: bool = True,
    ):
        self.arrow_batch_iterator = arrow_batch_iterator
        self.add_patches = add_patches
        self.tokenizer = tokenizer_args.build()
        self.pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0) # Still potentially useful info

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
            sample_ids = arrow_batch.column("sample_id").to_pylist()
            texts = arrow_batch.column("text").to_pylist()
            entropies_list = arrow_batch.column("entropies").to_pylist()
            batch_size = len(texts)
            if batch_size == 0:
                continue

            # --- Tokenization (Common Step) ---
            tokenized_batch = [np.frombuffer(s.encode("utf-8", errors="ignore"), dtype=np.uint8) + 4 for s in texts]

            # --- Prepare Output Lists ---
            processed_sample_ids = []
            processed_tokens = []
            processed_masks = []
            processed_patch_lengths = []
            # Process sample by sample to apply patcher correctly without premature padding
            for i in range(batch_size):
                tokens = tokenized_batch[i]
                sample_id = sample_ids[i]
                if not tokens: continue # Skip empty sequences

                entropies = entropies_list[i]

                tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
                entropies_tensor = torch.tensor([entropies], dtype=torch.float32, device=self.device)

                # Call patcher
                # include_next_token=False matches SequenceIterator expectation? Verify.
                patch_output = self.patcher.patch(
                    tokens=tokens_tensor,
                    entropies=entropies_tensor,
                    include_next_token=False
                )
                # patch_output['patch_lengths'] shape is likely (1, num_patches)
                patch_lengths = patch_output['patch_lengths'].squeeze(0).tolist()


                # Append valid processed data
                processed_sample_ids.append(sample_id)
                processed_tokens.append(tokens)
                processed_masks.append([True] * len(tokens))
                processed_patch_lengths.append(patch_lengths)


            # --- Yield Batch Dictionary ---
            # Only yield if there's valid processed data
            if processed_tokens:
                yield {
                    "sample_id": processed_sample_ids,
                    "tokens": processed_tokens,
                    "mask": processed_masks,
                    "patch_lengths": processed_patch_lengths
                }


    def __iter__(self):
        return self.create_iter()


# Vectorized patching

# return patch_lengths[0], scores

def find_entropy_patch_start_ids(entropies, threshold=None, include_next_token=False):
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
        patch_start_mask = patch_start_mask[:, :-1]
    patch_start_ids = patch_start_ids_from_patch_start_mask(patch_start_mask)
    patch_start_ids = torch.cat((first_ids, patch_start_ids + preds_truncation_len), dim=1)
    return patch_start_ids

def patch_start_ids_from_patch_start_mask(patch_start_mask):
    bs, trunc_seq_len = patch_start_mask.shape
    max_patches = patch_start_mask.sum(dim=1).max()
    if max_patches == 0:
        patch_start_ids = torch.full((bs, trunc_seq_len), trunc_seq_len, dtype=torch.long, device=patch_start_mask.device)
    else:
        patch_ids = torch.arange(trunc_seq_len, device=patch_start_mask.device).unsqueeze(0).repeat(bs, 1)
        extra_patch_ids = torch.full((bs, trunc_seq_len), trunc_seq_len, dtype=torch.long, device=patch_start_mask.device)
        all_patch_ids = torch.cat((patch_ids, extra_patch_ids), dim=1)
        patch_start_mask_padded = torch.cat((patch_start_mask, ~patch_start_mask), dim=1)
        patch_start_ids = all_patch_ids[patch_start_mask_padded].reshape(bs, trunc_seq_len)[:, :max_patches]
    return patch_start_ids

def patch_lengths_from_start_ids(patch_start_ids, seq_len):
    """
    Given patch start IDs (with extra padding), compute the patch lengths.
    """
    last_ids = torch.full_like(patch_start_ids[:, :1], seq_len - 1)
    patch_end_ids = torch.cat((patch_start_ids[:, 1:] - 1, last_ids), dim=1)
    patch_lengths = patch_end_ids - patch_start_ids + 1
    assert torch.all(patch_lengths >= 0), f"{patch_lengths}"
    assert not check_non_zero_after_zero(patch_lengths), f"{patch_lengths}"
    return patch_lengths

def split_large_numbers(lst, m):
    new_lst = []
    for i in lst:
        if i > m:
            while i > m:
                new_lst.append(m)
                i -= m
            new_lst.append(i)
        else:
            new_lst.append(i)
    assert sum(new_lst) == sum(lst), f"{sum(new_lst)} != {sum(lst)}"
    return new_lst

def check_non_zero_after_zero(tensor):
    zero_mask = tensor == 0
    shifted_mask = torch.cat([torch.zeros(tensor.shape[0], 1, dtype=torch.bool, device=tensor.device),
                               zero_mask[:, :-1]], dim=1)
    non_zero_after_zero = (tensor != 0) & shifted_mask
    return non_zero_after_zero.any()

def rightpad(seq, pad_id, max_len):
    return seq + [pad_id] * (max_len - len(seq))

include_next_token = False
bs, seq_len = tokens.shape
seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
scores = entropies.to(dtype=torch.float32)
patch_start_ids = find_entropy_patch_start_ids(
    scores, include_next_token=include_next_token,
    threshold=threshold if threshold is not None else self.threshold,
)
patch_lengths = patch_lengths_from_start_ids(patch_start_ids, seq_len_next_tok)
# enforce max patch size 16
patch_lengths = [split_large_numbers(pl, self.max_patch_length) for pl in patch_lengths.tolist()]
max_len = max(len(pl) for pl in patch_lengths)
patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
patch_lengths = torch.tensor(patch_lengths, dtype=tokens.dtype, device=tokens.device)

assert not check_non_zero_after_zero(patch_lengths)
last_non_zero_col_reversed = ((patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min())
patch_lengths = patch_lengths[:, : patch_lengths.shape[1] - last_non_zero_col_reversed]
expected_total = tokens.numel() + include_next_token * tokens.shape[0]
assert torch.sum(patch_lengths) == expected_total, f"{torch.sum(patch_lengths)} != {expected_total}"


# Vectorized chatgpt output
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 1. Entropy → patch-start indices
# ---------------------------------------------------------------------
def find_entropy_patch_start_ids(
        entropies: torch.Tensor,
        threshold: float,
        include_next_token: bool = False
) -> torch.LongTensor:
    """
    entropies        : [B, L]  — per-token entropy-scores
    returns LongTensor [B, 2+P]  (padded with seq_len; first two cols are 0, 1)
    """
    B, L = entropies.shape
    device = entropies.device

    # we always keep the first two positions (0, 1)
    first_two = torch.tensor([0, 1], device=device).expand(B, 2)

    ent = entropies[:, 1:]                       # drop the first token (handled by 'first_two')
    L_trunc = L - 1

    # where does entropy exceed threshold?
    start_mask = ent > threshold                 # [B, L-1]
    if not include_next_token:                   # optionally suppress the last element
        start_mask[:, -1] = False

    # gather the indices that were flagged in each row
    row_idx, col_idx = torch.where(start_mask)   # 1-D tensors
    counts_per_row = start_mask.sum(1)           # [B]
    max_patches = int(counts_per_row.max())

    pad_val = L                                  # sentinel “padding” index
    out = torch.full((B, 2 + max_patches), pad_val, dtype=torch.long, device=device)
    out[:, :2] = first_two

    if max_patches:                              # nothing to do if every row is empty
        flat = col_idx + 1                       # +1 because we removed the very first token
        chunks = torch.split(flat, counts_per_row.tolist())
        for b, chunk in enumerate(chunks):
            if chunk.numel():
                out[b, 2:2 + chunk.numel()] = chunk

    return out                                   # [B, 2+max_patches]


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
        lengths: torch.Tensor,
        max_patch: int
) -> torch.LongTensor:
    """
    Vectorised replacement of the old Python-loop ‘split_large_numbers’.

    lengths   : [B, N]  — zero-padded patch lengths
    returns   : [B, N′] — zero-padded, every entry ≤ max_patch
    """
    B, N = lengths.shape
    device, dtype = lengths.device, lengths.dtype

    # Flatten & discard zeros (already padding)
    flat        = lengths.view(-1)
    keep_mask   = flat > 0
    flat_keep   = flat[keep_mask]                         # (=all true patch lengths)

    # how many chunks does each real length expand to?
    n_chunks = torch.div(flat_keep + max_patch - 1, max_patch, rounding_mode='floor')
    # total expanded length per batch sample
    per_row_chunks = n_chunks.view(B, -1).sum(1)          # [B]
    max_chunks     = int(per_row_chunks.max().item())

    # ---- build the expanded (≤ max_patch) list ----------------------
    # 1) repeat ‘max_patch’ n_chunks-times
    expanded = torch.repeat_interleave(
        flat_keep.new_full((1,), max_patch), n_chunks, dim=0
    )
    # 2) fix the *last* element in every group if there is a remainder
    rem        = flat_keep % max_patch
    has_rem    = rem != 0
    last_idx   = torch.cumsum(n_chunks, 0) - 1            # idx of *last* chunk per group
    expanded[last_idx[has_rem]] = rem[has_rem]

    # -----------------------------------------------------------------
    # Re-assemble into [B, max_chunks] (still 0-padded)
    out = torch.zeros(B, max_chunks, dtype=dtype, device=device)
    p   = 0
    for b, cnt in enumerate(per_row_chunks.tolist()):
        if cnt:
            out[b, :cnt] = expanded[p : p + cnt]
            p += cnt
    return out                                            # [B, max_chunks]

include_next_token = False
bs, seq_len = tokens.shape
seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
scores = entropies.to(dtype=torch.float32)
patch_start_ids = find_entropy_patch_start_ids(
    scores, include_next_token=include_next_token,
    threshold=threshold if threshold is not None else self.threshold,
)
patch_lengths = patch_lengths_from_start_ids(
    patch_start_ids, seq_len_next_tok
)

patch_lengths = split_large_numbers_tensor(
    patch_lengths, max_patch=self.max_patch_length
)
max_len = max(len(pl) for pl in patch_lengths)
patch_lengths = [rightpad(pl, 0, max_len=max_len) for pl in patch_lengths]
patch_lengths = torch.tensor(patch_lengths, dtype=tokens.dtype, device=tokens.device)

assert not check_non_zero_after_zero(patch_lengths)
last_non_zero_col_reversed = ((patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min())
patch_lengths = patch_lengths[:, : patch_lengths.shape[1] - last_non_zero_col_reversed]
expected_total = tokens.numel() + include_next_token * tokens.shape[0]
assert torch.sum(patch_lengths) == expected_total, f"{torch.sum(patch_lengths)} != {expected_total}"
