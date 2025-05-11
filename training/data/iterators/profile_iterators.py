import time
from args import DataloaderArgs, find_and_sanitize_chunks

from preprocess_iterator import PreprocessIterator as PreprocessIteratorOld
from packing_iterator import PackingIterator as PackingIteratorOld, PackingArgs as PackingArgsOld
from arrow_iterator import ArrowFileIterator as ArrowFileIteratorOld
from sequence_iterator import SequenceIterator as SequenceIteratorOld, SequencePackingArgs

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
    return batch

def serialize_iterator():
    file_format = args.file_format
    dataset_path = args.root_dir
    entropy_files = "entropies_validation0_repeated.arrow"
    preprocess_dir = args.preprocess_dir
    entropy_model_name = args.entropy_model_name
    arrow_batch_size = args.arrow_batch_size
    rank = 0
    world_size = 1
    s3_profile = args.s3_profile

    sequence_packing_args = SequencePackingArgs(
        output_seq_len=args.seq_len,
        buffer_size=args.buffer_size,
    )

    dataset_chunks = find_and_sanitize_chunks(
        dataset_path=dataset_path,
        world_size=world_size,
        file_pattern=entropy_files,
        s3_profile=s3_profile,
    )

    arrow_iterator = ArrowFileIteratorOld(
            file_path=None,
            file_format=file_format,
            worker_id=rank,
            num_workers=world_size,
            preprocess_dir=preprocess_dir,
            dataset_files=dataset_chunks,
            entropy_model_name=entropy_model_name,
            arrow_batch_size=args.arrow_batch_size,
        )
    
    preprocess_iterator = PreprocessIteratorOld(
        arrow_iterator,
        patcher_args=args.patcher_args,
        tokenizer_args=args.tokenizer_args,
        add_patches=args.add_patches,
    )

    sequence_iterator = SequenceIteratorOld(
        preprocess_iterator,
        sequence_packing_config=sequence_packing_args,
        # rng_state=shuffle_rng_state,
    )
    packing_args = PackingArgsOld(
        batch_size=16,
        seq_len=4096,
        pad_id=0,
        max_length=8192,
        pad_to_max_length=True,
        enable_byte_ngrams=False,
        packing_mode=PackingMode.PATCHING

    )
    packing_iterator = iter(PackingIteratorOld(sequence_iterator, packing_args=packing_args))

    return packing_iterator
    

def vectorized_iterator():
    file_format = args.file_format
    dataset_path = args.root_dir
    entropy_files = "entropies_validation0_repeated.arrow"
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

    arrow_iterator = ArrowFileIterator(
            file_path=None,
            file_format=file_format,
            worker_id=rank,
            num_workers=world_size,
            preprocess_dir=preprocess_dir,
            dataset_files=dataset_chunks,
            entropy_model_name=entropy_model_name,
            arrow_batch_size=args.arrow_batch_size,
        )
    

    preprocess_iterator = PreprocessIterator(
            arrow_batch_iterator=arrow_iterator,
            add_patches=True,
            patcher_args=args.patcher_args,
        )

    sequence_iterator = SequenceIterator(
            preprocess_iterator=preprocess_iterator,
            max_seq_patches=args.seq_len
            * args.buffer_size,  # add one for separate EOS token patch
    )

    packing_args = PackingArgs(
        batch_size=16,
        seq_len=4096,
        pad_id=0,
        max_length=8192,
        pad_to_max_length=True,
        enable_byte_ngrams=False,
        packing_mode=PackingMode.PATCHING
    )

    packing_iterator = iter(PackingIterator(
            sequence_iterator=sequence_iterator,
            packing_args=packing_args,
        ))
    

    return packing_iterator

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
    # s_iter = serialize_iterator()
    v_iter = vectorized_iterator()

    # batch1 = next(s_iter)
    batch2 = next(v_iter)
    sum_ = 0

    while True:
        try:
            batch2 = next(v_iter)
        except StopIteration:
            break
    

    # i = 1
    # while (batch1 is not None) and (batch2 is not None):
    #     assert torch.from_numpy(batch1.x).eq(batch2.x).all(), f"Failed assert on iteration {i}. Batch1: {batch1.x} Batch2: {batch2.x}"
    #     assert torch.from_numpy(batch1.y).eq(batch2.y).all(), f"Failed assert on iteration {i}. Batch1: {batch1.y} Batch2: {batch2.y}"
    #     assert torch.from_numpy(batch1.mask).eq(batch2.mask).all()
    #     assert torch.from_numpy(batch1.patch_lengths).eq(batch2.patch_lengths).all()
    #     batch1 = next(s_iter)
    #     batch2 = next(v_iter)
    #     i += 1

    
    # assert (batch1 is None) and (batch2 is None)
