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
    entropy_files = "entropies_validation0_rndmized.arrow"
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

    # sequence_iterator = SequenceIteratorOld(
    #     preprocess_iterator,
    #     sequence_packing_config=sequence_packing_args,
    #     # rng_state=shuffle_rng_state,
    # )
    # packing_args = PackingArgsOld(
    #     batch_size=16,
    #     seq_len=4096,
    #     pad_id=0,
    #     max_length=8192,
    #     pad_to_max_length=True,
    #     enable_byte_ngrams=False,
    #     packing_mode=PackingMode.PATCHING

    # )
    # packing_iterator = iter(PackingIteratorOld(sequence_iterator, packing_args=packing_args))

    return iter(preprocess_iterator)
    

def vectorized_iterator():
    file_format = args.file_format
    dataset_path = args.root_dir
    entropy_files = "entropies_validation0_rndmized.arrow"
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

    # sequence_iterator = SequenceIterator(
    #         preprocess_iterator=preprocess_iterator,
    #         max_seq_patches=args.seq_len
    #         * args.buffer_size,  # add one for separate EOS token patch
    # )

    # packing_args = PackingArgs(
    #     batch_size=16,
    #     seq_len=4096,
    #     pad_id=0,
    #     max_length=8192,
    #     pad_to_max_length=True,
    #     enable_byte_ngrams=False,
    #     packing_mode=PackingMode.PATCHING
    # )

    # packing_iterator = iter(PackingIterator(
    #         sequence_iterator=sequence_iterator,
    #         packing_args=packing_args,
    #     ))
    

    return iter(preprocess_iterator)

if __name__ == "__main__":
    s_iter = serialize_iterator()
    v_iter = vectorized_iterator()
    # PREPROCESS ITERATOR TEST 
    i = 0
    while True:
        batch2 = next(v_iter)
        batch1_x = []
        batch1_pl = []
        for _ in range(batch2.tokens.shape[0]):
            sample = next(s_iter)
            batch1_x.append(torch.tensor(sample.tokens))
            batch1_pl.append(torch.tensor(sample.patch_lengths))
        
        tok_lengths = batch2.patch_lengths.sum(dim=1)
        assert all(batch1_x[i].eq(batch2.tokens[i, :tok_lengths[i]]).all().item() for i in range(batch2.tokens.shape[0]))
        import IPython
        ns = locals().copy()
        ns.update(globals())
        IPython.embed(user_ns=ns)
        exit()
        # assert all(batch1_pl[pl1].eq(batch2.patch_lengths[pl1, :batch1_pl[pl1].shape[0]]).all().item() for pl1 in range(len(batch1_pl)))

        i += 1
        print(i)

    # SEQUENCE ITERATOR TEST
    # batch2 = next(v_iter)
    # batch1_x = []
    # batch1_pl = []
    # for _ in range(64):
    #     sample = next(s_iter)
    #     batch1_x.extend(torch.tensor(sample.tokens))
    #     batch1_pl.extend(torch.tensor(sample.patch_lengths))

    # batch1_x = torch.stack(batch1_x)
    # batch1_pl = torch.stack(batch1_pl)


    # PACKING ITERATOR TEST
    # batch1 = next(s_iter)
    # batch2 = next(v_iter)
    

    # i = 1
    # while (batch1 is not None) and (batch2 is not None):
    #     if not torch.from_numpy(batch1.x).eq(batch2.x).all():
    #         import IPython
    #         ns = locals().copy()
    #         ns.update(globals())
    #         IPython.embed(user_ns=ns)
    #         exit()
    #     batch1 = next(s_iter)
    #     batch2 = next(v_iter)
    #     i += 1

    
    # assert (batch1 is None) and (batch2 is None)