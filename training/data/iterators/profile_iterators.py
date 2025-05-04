import time 
from args import DataloaderArgs, find_and_sanitize_chunks
from arrow_iterator import ArrowFileIterator
import torch
from torch.nn.utils.rnn import pad_sequence
import pyarrow as pa
import numpy as np

# begin = time.time()

# for _ in range(759):
#     print(type(next(iterator)))

# end = time.time()
# print(f"Time taken: {end - begin}")

args = DataloaderArgs()

# testing purposes; debugging iterators implementation
def get_batch():
    train_loader = iter(args.build_from_rank(0, 1))
    token_sums = []
    patch_sums = []
    batch = next(train_loader)

def series_iterator_patch_lengths():
    file_format = args.file_format
    dataset_path = args.root_dir
    entropy_files = "entropies_validation_cloned.arrow" 
    preprocess_dir = args.preprocess_dir
    entropy_model_name = args.entropy_model_name
    arrow_batch_size = args.arrow_batch_size
    rank = 0 
    world_size = 1 
    s3_profile = args.s3_profile

    dataset_chunks = find_and_sanitize_chunks(
        dataset_path=dataset_path,
        world_size=world_size,
        file_pattern=entropy_files,
        s3_profile=s3_profile
    )

    iterator = iter(ArrowFileIterator(
        file_path=None, 
        file_format=file_format,
        worker_id=rank, 
        num_workers=world_size, 
        preprocess_dir=preprocess_dir,
        dataset_files=dataset_chunks,
        entropy_model_name=entropy_model_name,
        arrow_batch_size=arrow_batch_size
    ))

    def tokenize(text):
        tokens = bytes(text, encoding="utf-8", errors="ignore")
        tokens = [int(unit) + 4 for unit in tokens] # 4 is offset
        tokens.append(2) # 2 is EOS token
        return tokens

    patch_len_list = []

    for _ in range(16):
        batch = next(iterator)
        entropies_tensor = torch.tensor(batch.entropies).unsqueeze(0)
        # Calculate patch lengths
        tokens_tensor = torch.tensor(tokenize(batch.text)).unsqueeze(0)
        patcher = args.patcher_args.build()
        patch_lengths = patcher.patch(
            tokens_tensor,
            include_next_token=False,
            entropies=entropies_tensor,
        )[0][0]
        patch_len_list.append(patch_lengths)
    
    return patch_len_list

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



def vectorized_iterator_patch_lengths():
    file_format = args.file_format
    dataset_path = args.root_dir
    entropy_files = "entropies_validation_cloned.arrow" 
    preprocess_dir = args.preprocess_dir
    entropy_model_name = args.entropy_model_name
    arrow_batch_size = args.arrow_batch_size
    rank = 0 
    world_size = 1 
    s3_profile = args.s3_profile

    dataset_chunks = find_and_sanitize_chunks(
        dataset_path=dataset_path,
        world_size=world_size,
        file_pattern=entropy_files,
        s3_profile=s3_profile
    )

    dataset = pa.dataset.dataset(
        dataset_chunks[0], format=file_format
    )
    batch_iterator = dataset.to_batches()
    batch = next(batch_iterator)
    sample_ids = batch.column("sample_id").to_pylist()
    texts = batch.column("text").to_pylist()
    entropies_list = batch.column("entropies").to_pylist()
    batch_size = len(texts)
    # --- Tokenization (Common Step) ---
    # padding id is 0
    # TODO: make patch lengths work correctly with EOS token.
    tokens = pad_sequence([torch.frombuffer(bytearray(s.encode("utf-8", errors="ignore")), dtype=torch.uint8).long() + 4 for s in texts], batch_first=True, padding_value=0)
    entropies = pad_sequence([torch.tensor(e, dtype=torch.float32) for e in entropies_list], batch_first=True, padding_value=0)
    include_next_token = False
    bs, seq_len = tokens.shape
    seq_len_next_tok = seq_len + 1 if include_next_token else seq_len
    patch_start_ids = find_entropy_patch_start_ids(
        entropies, include_next_tok=include_next_token,
        threshold=args.patcher_args.threshold
    )
    patch_lengths = patch_lengths_from_start_ids(
        patch_start_ids, seq_len_next_tok
    )

    patch_lengths = split_large_numbers_tensor(
        patch_lengths, max_patch=args.patcher_args.max_patch_length
    )

    from patching import check_non_zero_after_zero
    assert not check_non_zero_after_zero(patch_lengths)
    last_non_zero_col_reversed = ((patch_lengths != 0).flip(dims=[1]).int().argmax(dim=1).min())
    patch_lengths = patch_lengths[:, : patch_lengths.shape[1] - last_non_zero_col_reversed]
    expected_total = tokens.numel() + include_next_token * tokens.shape[0]
    assert torch.sum(patch_lengths) == expected_total, f"{torch.sum(patch_lengths)} != {expected_total}"

# token_sums.append(batch.mask.sum().item())
# patch_sums.append(batch.patch_lengths.flatten().nonzero()[0].shape[0])
# cProfile.runctx("next(train_loader)", globals(), locals(), sort="tottime")

# import time
# begin = time.time()
# for i, v in enumerate(train_loader):
#     print(i)
# end = time.time()
# print(f"Time taken: {end - begin}")

# print("Real token ratio:", sum(token_sums) / (16 * 8192 * batches))
# print("Real patch ratio:", sum(patch_sums) / (16 * 4096 * batches))

if __name__ == "__main__":
    # get_batch()
    vectorized_iterator_patch_lengths()
    # series_iterator_patch_lengths()
