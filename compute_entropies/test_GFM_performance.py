import os
import time
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from datasets import config, load_dataset
from evo2 import Evo2
import pyarrow as pa
MAX_LENGTH = 8192
PAD_TOKEN = 1



# -------------------------------------------------------------------------
# Custom distributed sampler
# -------------------------------------------------------------------------
class LengthAwareDistributedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=False,
        lengths=None,
    ):
        # Initialize num_replicas, rank, and compute lengths as before...
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

        if lengths is None:
            self.lengths = [sample["length"] for sample in dataset]
        else:
            self.lengths = lengths
        self.dataset_size = len(self.dataset)

    def __iter__(self):
        # 1. Sort indices by sequence length.
        indices = list(range(self.dataset_size))
        indices.sort(key=lambda idx: self.lengths[idx])

        # 2. Group indices into batches.
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.shuffle:
            np.random.shuffle(batches)

        total_batches = len(batches)
        base_batches = total_batches // self.num_replicas
        even_count = base_batches * self.num_replicas

        # 3. Assign batches evenly (via round-robin) for the evenly-divisible part.
        batches_for_rank = [
            batches[i]
            for i in range(even_count)
            if (i % self.num_replicas == self.rank)
        ]
        # 4. Allocate remaining extra batches solely to rank 0.
        if self.rank == 0:
            batches_for_rank.extend(batches[even_count:])

        yield from batches_for_rank

    def __len__(self):
        total_batches = (self.dataset_size + self.batch_size - 1) // self.batch_size
        base_batches = total_batches // self.num_replicas
        remainder = total_batches % self.num_replicas
        if self.rank == 0:
            return base_batches + remainder
        else:
            return base_batches


# -------------------------------------------------------------------------
# Collate function (unchanged)
# -------------------------------------------------------------------------
def collate(sequences: list):
    # Pad to max length in the batch
    text = [s["text"] for s in sequences]
    record = [s["record"] for s in sequences]
    lengths = [s["length"] for s in sequences]


    return (
        pad_sequence(
            [
                torch.from_numpy(
                    np.frombuffer(bytearray(s.encode("utf-8")), dtype=np.uint8)
                )
                for s in text
            ],
            batch_first=True,
            padding_value=PAD_TOKEN,  # StripedHyena CharTokenizer pad token. eod and eos are 0.
        ),
        record,
        text,
        lengths,
    )


# -------------------------------------------------------------------------
# Entropy Helpers (unchanged)
# -------------------------------------------------------------------------
def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]
    """
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    return -p_log_p.sum(dim=-1)


def calculate_entropies(tokens: torch.tensor, model: torch.nn.Module, device):
    """
    tokens: [batch_size, seq_len]
    Return shape: [batch_size, seq_len]
    """
    with torch.inference_mode():
        entropies = []
        # MAX_LENGTH = getattr(model, "MAX_LENGTH", 8192)
        batch_numel = MAX_LENGTH * tokens.size(0)

        # Flatten and split into blocks of (batch_size * MAX_LENGTH)
        splits = torch.split(tokens.flatten(), batch_numel)

        for split in splits:
            pad_size = (MAX_LENGTH - (split.numel() % MAX_LENGTH)) % MAX_LENGTH
            pad = torch.ones(
                pad_size, dtype=split.dtype, device=split.device
            )  # Evo tokenizer pad is 1
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, MAX_LENGTH)

            split = split.to(device)

            # NOTE: StripedHyena2 seems to output some "inference_params_dict_out" object that we don't need.
            outputs, _ = model(split)  # => [batch, MAX_LENGTH, vocab]
            pred = outputs[0]

            # Remove padding
            pred = pred.reshape(-1, pred.shape[-1])[: split.numel() - pad_size, :]

            pred_entropies = entropy(pred)  # => [batch * seq_len]
            entropies.append(pred_entropies)

        concat_entropies = torch.cat(entropies, dim=0)
        concat_entropies = concat_entropies.reshape(tokens.shape)
    return concat_entropies


def add_length(example):
    example["length"] = len(example["text"])
    return example

def init_distributed_training(
    rank,
    world_size,
    master_addr,
    master_port,
    backend,
    gpu_per_node,
    data_path,
    split,
    batch_size,
    arrow_batch=10,
):
    # Set environment variables for master address and port
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Set GPU device
    torch.cuda.set_device(rank % gpu_per_node)

    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Synchronize all processes
    dist.barrier()

    # Message indicating the process has passed the barrier
    print(f"Process {rank} passed barrier")
    data = load_dataset(f"{data_path}/stage1", split=split).with_format("torch")

    if rank > 0:
        dist.barrier()

    data = data.map(add_length)

    if rank == 0:
        dist.barrier()

    # Standard PyTorch DistributedSampler (removes your custom length-sorting).
    # If you need length-based sorting globally, you need a custom distributed sampler.
    dist_sampler = LengthAwareDistributedBatchSampler(
        data,
        batch_size,
        num_replicas=world_size,
        rank=rank,
    )

    dataloader = DataLoader(
        data,
        batch_sampler=dist_sampler,  # ensures each rank sees a unique subset
        collate_fn=collate,
    )

    # ---------------------------------------------------------------------
    # 3. Load model & wrap with DistributedDataParallel
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Resume functionality - check if we need to resume from an existing file
    # ---------------------------------------------------------------------
    resume_mode = False
    last_sample_ids = []
    rank_output_file = f"entropies_rank{rank}.arrow"
    
    # Check if the output file exists to potentially resume
    if os.path.exists(rank_output_file):
        try:
            # Read the existing Arrow file to get the last batch
            with pa.memory_map(rank_output_file, "r") as source:
                reader = pa.ipc.open_file(source)
                # Get the last batch if any batches exist
                num_processed_batches = reader.num_record_batches
                if num_processed_batches > 0:
                    last_batch = reader.get_batch(num_processed_batches - 1)
                    last_batch_dict = last_batch.to_pydict()
                    last_sample_ids = last_batch_dict["sample_id"]
                    resume_mode = True
                    print(f"Rank {rank}: Found existing file with {num_processed_batches} batches.")
                    print(f"Rank {rank}: Last batch has {len(last_sample_ids)} samples.")
        except Exception as e:
            print(f"Rank {rank}: Error reading existing file: {e}")
            print(f"Rank {rank}: Starting from beginning")
            resume_mode = False

    entropy_model = Evo2("evo2_1b_base", device=rank)
    entropies_buffer = []
    text_buffer = []
    sample_ids_buffer = []
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])
    processed_batches = 0

    try:
        with pa.OSFile(rank_output_file, 'wb') as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                for tokens, sample_ids, texts, lengths in dataloader:
                    if resume_mode:
                        # Check if this batch contains any of the last sample IDs
                        if any(sid in last_sample_ids for sid in sample_ids):
                            print(
                                f"Rank {rank}: Found batch containing last processed samples"
                            )
                            # Check if this batch exactly matches or contains all the last saved sample IDs
                            if set(sample_ids) == set(last_sample_ids) or all(
                                sid in last_sample_ids for sid in sample_ids
                            ):
                                print(
                                    f"Rank {rank}: Found last processed batch, will resume from next batch"
                                )
                                processed_batches += 1
                                # Mark that we've found our resume point
                                # so the next batch will be processed
                                resume_mode = False
                                continue
                            else:
                                # This is a partially processed batch - only keep unprocessed samples
                                indices_to_process = [
                                    i
                                    for i, sid in enumerate(sample_ids)
                                    if sid not in last_sample_ids
                                ]
                                tokens = tokens[indices_to_process]
                                texts = [texts[i] for i in indices_to_process]
                                sample_ids = [
                                    sample_ids[i] for i in indices_to_process
                                ]
                                resume_mode = False
                                print(
                                    f"Rank {rank}: Resuming with {len(indices_to_process)} new samples in partially processed batch"
                                )
                    else:
                        # Skip this batch as it was already processed
                        processed_batches += 1
                        continue
                    tokens = tokens.to(
                        dtype=torch.int, device=rank
                    )  # push tokens to GPU

                    # Calculate entropies
                    start_time = time.time()
                    scores = calculate_entropies(tokens, entropy_model, device=rank)
                    end_time = time.time()
                    print(f"Took {end_time - start_time} seconds to process {tokens.shape[0]}.")
                    for row, length in zip(scores, lengths):
                        entropies_buffer.append(row[:length].to("cpu", dtype=torch.float16).numpy())
                    text_buffer.extend(texts)
                    sample_ids_buffer.extend(sample_ids)
                    
                    if len(entropies_buffer) >= arrow_batch:
                        # Create pyarrow table
                        batch = pa.record_batch(
                            {"entropies": entropies_buffer, "text": text_buffer, "sample_id": sample_ids},
                            schema
                        )
                        
                        # Use 'ab' (append binary) mode for file operations
                        # For the first write, this is same as 'wb', for subsequent writes it appends
                        
                        writer.write_batch(batch)
                        
                        entropies_buffer = []
                        text_buffer = []
                        sample_ids_buffer = []

                        """
                        with pa.memory_map(output_file, "r") as source:
                            reader = pa.ipc.open_file(source)

                            # Process and write one batch at a time
                            df = reader.read_pandas()

                        """
                    
                if len(entropies_buffer) > 0:
                    batch = pa.record_batch(
                        {"entropies": entropies_buffer, "text": text_buffer, "sample_id": sample_ids_buffer},
                        schema
                    )
                    writer.write_batch(batch)

                    entropies_buffer = []
                    text_buffer = []
                    sample_ids_buffer = []

    except Exception as e:
        # Just log the error - the partial Arrow file is already saved and can be used for resuming
        print(f"Rank {rank}: Error during processing: {e}")
        # Re-raise the original error
        raise e

    finally:
        # -----------------------------------------------------------------
        # 6. Destroy process group to clean up
        # -----------------------------------------------------------------
        dist.destroy_process_group()


def main_worker(local_rank, args):
    print(os.environ["SLURM_PROCID"])
    if "SLURM_PROCID" in os.environ:
        node_rank = int(os.environ["SLURM_PROCID"])
    else:
        node_rank = 0
    print("node rank:", node_rank)
    global_rank = node_rank * args.gpu_per_node + local_rank
    world_size = args.world_size
    print("global rank:", global_rank)
    print("world size:", world_size)
    init_distributed_training(
        rank=global_rank,
        world_size=world_size,
        master_addr=args.master_addr,
        master_port=args.master_port,
        backend=args.backend,
        gpu_per_node=args.gpu_per_node,
        data_path=args.data_path,
        split=args.split,
        batch_size=args.batch_size,
        arrow_batch=args.arrow_batch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Training Test with mp.spawn"
    )
    parser.add_argument(
        "--master_addr", type=str, required=True, help="Address of the master node"
    )
    parser.add_argument(
        "--master_port", type=int, required=True, help="Port of the master node"
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["gloo", "nccl"],
        help="Distributed backend",
    )
    parser.add_argument(
        "--world_size", type=int, required=True, help="Number of nodes in total"
    )
    parser.add_argument(
        "--gpu_per_node", type=int, required=True, help="Number of GPUs per node"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="The path to the Open Genome dataset",
    )
    parser.add_argument(
        "--split", type=str, required=True, help="The train/test/val split"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="The batch size to process entropies",
    )
    parser.add_argument(
        "--arrow_batch",
        type=int,
        required=False,
        help="The batch size to write arrow files",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        required=True,
        help="The directory to cache the Open Genome dataset",
    )

    args = parser.parse_args()

    # Use mp.spawn to launch multiple processes, each corresponding to a GPU

    os.environ["HF_HOME"] = args.data_path
    os.environ["HF_DATASETS_CACHE"] = args.data_cache_dir
    config.HF_DATASETS_CACHE = args.data_cache_dir

    mp.spawn(main_worker, nprocs=args.gpu_per_node, args=(args,))
