import time 
from args import DataloaderArgs, find_and_sanitize_chunks
from arrow_iterator import ArrowFileIterator
import torch
import pyarrow as pa

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
