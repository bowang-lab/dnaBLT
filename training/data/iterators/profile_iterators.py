import time
from args import DataloaderArgs, find_and_sanitize_chunks
from v_preprocess_iterator import PreprocessIterator
from v_arrow_iterator import ArrowFileIterator
from v_sequence_iterator import SequenceIterator
from v_packing_iterator import PackingIterator, PackingArgs, PackingMode
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

# begin = time.time()

# for _ in range(759):
#     print(type(next(iterator)))

# end = time.time()
# print(f"Time taken: {end - begin}")

args = DataloaderArgs()


# testing purposes; debugging iterators implementation
def get_batch():
    train_loader = iter(args.build_from_rank(0, 1))
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
        s3_profile=s3_profile,
    )

    iterator = iter(
        ArrowFileIterator(
            file_path=None,
            file_format=file_format,
            worker_id=rank,
            num_workers=world_size,
            preprocess_dir=preprocess_dir,
            dataset_files=dataset_chunks,
            entropy_model_name=entropy_model_name,
            arrow_batch_size=arrow_batch_size,
        )
    )

    def tokenize(text):
        tokens = bytes(text, encoding="utf-8", errors="ignore")
        tokens = [int(unit) + 4 for unit in tokens]  # 4 is offset
        tokens.append(2)  # 2 is EOS token
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
        s3_profile=s3_profile,
    )

    arrow_iterator = iter(
        ArrowFileIterator(
            file_path=None,
            file_format=file_format,
            worker_id=rank,
            num_workers=world_size,
            preprocess_dir=preprocess_dir,
            dataset_files=dataset_chunks,
            entropy_model_name=entropy_model_name,
            arrow_batch_size=args.arrow_batch_size,
        )
    )

    preprocess_iterator = iter(
        PreprocessIterator(
            arrow_batch_iterator=arrow_iterator,
            add_patches=True,
            patcher_args=args.patcher_args,
        )
    )

    sequence_iterator = iter(
        SequenceIterator(
            preprocess_iterator=preprocess_iterator,
            max_seq_patches=args.seq_len
            * args.buffer_size,  # add one for separate EOS token patch
        )
    )

    packing_args = PackingArgs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pad_id=0,
        max_length=args.max_encoder_seq_length,
        pad_to_max_length=args.pad_to_max_length,
        enable_byte_ngrams=args.enable_byte_ngrams,
        packing_mode=PackingMode.PATCHING,
    )

    packing_iterator = iter(
        PackingIterator(
            sequence_iterator=sequence_iterator,
            packing_args=packing_args,
        )
    )

    batch = next(packing_iterator)
    import IPython
    ns = locals().copy()
    ns.update(globals())
    IPython.embed(user_ns=ns)
    exit()

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
