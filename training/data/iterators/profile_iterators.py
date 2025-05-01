import time
# Import base iterators
from args import DataloaderArgs, find_and_sanitize_chunks
from arrow_iterator import ArrowFileIterator
from preprocess_iterator import PreprocessIterator
from sequence_iterator import SequenceIterator, SequencePackingArgs
from sampling_iterator import SamplingIterator
from packing_iterator import PackingIterator, PackingArgs, PackingMode

# Import profiling wrappers
from profiling_wrappers import (
    ArrowFileIteratorWrapper,
    PreprocessIteratorWrapper,
    SequenceIteratorWrapper,
    SamplingIteratorWrapper,
    PackingIteratorWrapper
)

import numpy as np

# You may need to adjust these imports according to your project structure

def main():
    # Load your DataloaderArgs as you would in training
    # TODO: Ensure DataloaderArgs loads correctly or replace with manual setup
    try:
        args = DataloaderArgs()
    except Exception as e:
        print(f"Error loading DataloaderArgs: {e}")
        print("Please ensure DataloaderArgs can be instantiated or configure manually.")
        # Example manual config (replace with your actual paths/settings)
        class MockArgs:
            file_format = "arrow"
            root_dir = "/cluster/projects/bwanggroup/open-genome" # From your args change
            sources = {"train": {"16b1.arrow": 1}} # From your args change
            preprocess_dir = None # Set if needed
            entropy_model_name = None # Set if needed
            arrow_batch_size = 100
            s3_profile = None
            patcher_args = None # Define PatcherArgs if needed
            tokenizer_args = None # Define TokenizerArgs if needed
            add_patches = False # Set based on your config
            add_tokens = True   # Set based on your config
            seed = 42
            output_seq_len = 4096
            buffer_size = 512
            batch_size = 16
            seq_len = 4096
            max_encoder_seq_length = 4096
            pad_to_max_length = True
            enable_byte_ngrams = False
            # Need to figure out how patcher_args is built if needed
            # class MockPatcherArgs:
            #     patching_mode = "byte" # Example
            # patcher_args = MockPatcherArgs()
            class MockTokenizerArgs:
                name = "bytes" # Example
                def build(self): return type('MockTokenizer', (), {'boe_id': 0})() # Dummy tokenizer
            tokenizer_args = MockTokenizerArgs()
            patcher_args = type('MockPatcherArgs', (), {'patching_mode': 'byte'})() # Simpler mock


        args = MockArgs() # Use mock if loading fails

    rank = 0
    world_size = 1

    print("Profiling nested iterator speeds (first 5 batches)...\n")

    # Construct the iterator chain using wrappers

    # 1. ArrowFileIterator
    # Use the first source file found in args.sources['train']
    train_sources = list(args.sources['train'].keys())
    if not train_sources:
        raise ValueError("No training source files defined in DataloaderArgs.sources['train']")
    first_train_source = train_sources[0]

    # We need the actual list of files for ArrowFileIterator if not using file_path
    # For simplicity, assuming dataset_path points to a single file or a dir Arrow can handle
    # If dataset_files is required, populate it based on root_dir and sources
    dataset_chunks = find_and_sanitize_chunks(
        args.root_dir, world_size, first_train_source 
    )
    arrow_iterator = ArrowFileIterator(
        file_path=None, # Assume direct path works, adjust if needed
        file_format=args.file_format,
        worker_id=rank,
        num_workers=world_size,
        preprocess_dir=args.preprocess_dir,
        dataset_files=dataset_chunks, # Set this if file_path is None
        entropy_model_name=args.entropy_model_name,
        arrow_batch_size=args.arrow_batch_size
    )
    arrow_wrapper = ArrowFileIteratorWrapper(arrow_iterator)

    # 2. PreprocessIterator
    preprocess_iterator = PreprocessIterator(
        arrow_wrapper, # Pass the wrapper
        patcher_args=args.patcher_args,
        tokenizer_args=args.tokenizer_args,
        add_patches=args.add_patches,
    )
    preprocess_wrapper = PreprocessIteratorWrapper(preprocess_iterator)

    # 3. SequenceIterator
    # sequence_packing_args = args.sequence_packing_args # This arg doesn't seem to exist directly
    # Build SequencePackingArgs manually based on DataloaderArgs
    sequence_packing_args = SequencePackingArgs(
        output_seq_len=args.seq_len, # Use seq_len from DataloaderArgs
        buffer_size=args.buffer_size
    )
    shuffle_rng_state = np.random.default_rng((args.seed, rank, world_size)).bit_generator.state
    sequence_iterator = SequenceIterator(
        preprocess_wrapper, # Pass the wrapper
        sequence_packing_config=sequence_packing_args,
        rng_state=shuffle_rng_state, # Use None for profiling simplicity
    )
    sequence_wrapper = SequenceIteratorWrapper(sequence_iterator)

    # 4. SamplingIterator
    source_to_weight = args.sources['train'] # Use the 'train' sources directly
    # The iterator passed here should be the one providing the final sequences for this source
    source_to_iterator = {first_train_source: sequence_wrapper} # Use the filename as key, pass wrapper
    sampling_iterator = SamplingIterator(
        rng_state=shuffle_rng_state, # Use None for profiling simplicity
        source_to_weight=source_to_weight,
        source_to_iterator=source_to_iterator,
    )
    sampling_wrapper = SamplingIteratorWrapper(sampling_iterator)

    # 5. PackingIterator
    tokenizer = args.tokenizer_args.build()
    if args.tokenizer_args.name == "bytes":
        pad_id = 0
    else:
        pad_id = tokenizer.boe_id

    packing_args = PackingArgs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pad_id=pad_id,
        max_length=getattr(args, 'max_encoder_seq_length', args.seq_len), # Fallback to seq_len
        pad_to_max_length=args.pad_to_max_length,
        enable_byte_ngrams=getattr(args, 'enable_byte_ngrams', False),
        # Determine PackingMode based on patcher_args if available
        packing_mode=(
            PackingMode.BYTES
            if args.patcher_args and args.patcher_args.patching_mode == 'byte'
            else PackingMode.PATCHING
        )
    )
    packing_iterator = PackingIterator(sampling_wrapper, packing_args=packing_args) # Pass the wrapper
    packing_wrapper = PackingIteratorWrapper(packing_iterator)

    # Iterate through the top-level wrapper
    print("Starting iteration through PackingIteratorWrapper...")
    total_batches_processed = 0
    total_start_time = time.time()

    try:
        for i, batch in enumerate(packing_wrapper):
            print(f"--> Received Batch {i} from top-level PackingIteratorWrapper")
            total_batches_processed += 1
            # Process batch minimally if needed for debugging, e.g.:
            # print(f"    Batch keys: {batch.keys()}")
            # print(f"    Token shape: {batch['tokens'].shape}")
            if i >= 4:  # Limit to 5 batches for profiling
                print("\nReached target number of batches (5). Stopping profiling.")
                break
            print("-" * 30) # Separator between batches
    except Exception as e:
        print(f"\nError during iteration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_end_time = time.time()
        print("="*50)
        print(f"Profiling finished.")
        print(f"Total batches processed: {total_batches_processed}")
        print(f"Total time: {total_end_time - total_start_time:.4f} seconds")
        if total_batches_processed > 0:
            avg_time = (total_end_time - total_start_time) / total_batches_processed
            print(f"Average time per batch: {avg_time:.4f} seconds")
        print("="*50)


if __name__ == "__main__":
    main()
