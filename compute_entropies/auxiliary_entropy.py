import os
import time
import argparse
import random
from math import ceil
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import config, load_dataset
from evo2 import Evo2
import pyarrow as pa

MAX_LENGTH = 8192
PAD_TOKEN = 1


# -------------------------------------------------------------------------
# Custom dataloader
# -------------------------------------------------------------------------
class EntropyDataset(Dataset):
    def __init__(self, hf_dataset):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # idx is of format (doc_id, start_idx, length)
        doc_id, start_idx, length = idx
        sample = self.dataset[doc_id]
        text_sample = sample["text"][start_idx : start_idx + length]
        record = sample["record"] + "_" + str(start_idx) + "_" + str(length)
        tokenized_text = torch.from_numpy(
            np.frombuffer(bytearray(text_sample.encode("utf-8")), dtype=np.uint8)
        )
        return {
            "tokens": tokenized_text,
            "record": record,
            "text": text_sample,
        }


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
        seed=42,
        segment_size=None,
        skip_segments=0,
    ):
        # NOTE: This should just return the indices of the segments. We'll need to define them all and then split them into batches and allocate to ranks.

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
        self.seed = seed
        self.segment_size = segment_size
        self.skip_segments = skip_segments

        self.segments_per_doc = []

        # TOTAL_SEGMENT_FORMAT: (doc_id, start_idx, length)
        random.seed(self.seed)

        for i, sample in enumerate(self.dataset):
            quotient, remainder = divmod(len(sample["text"]), self.segment_size)
            # [ACT][GAC][TGA][CT] when start = 0. [A][CTG][ACT][GAC][T] when start = 1. [AC][TGA][CTG][ACT] when start = 2.
            start = random.randint(0, remainder) if quotient else 0
            end = remainder - start
            if start:
                self.segments_per_doc.append((i, 0, start))

            for x in range(quotient):
                self.segments_per_doc.append((i, start, self.segment_size))
                start += self.segment_size

            if end:
                self.segments_per_doc.append((i, start, end))

        self.total_segments = len(self.segments_per_doc)
        self.effective_total_segments = max(0, self.total_segments - self.skip_segments)

    def __iter__(self):
        # 1. Sort indices by sequence length.
        self.segments_per_doc.sort(key=lambda tup: tup[2], reverse=True)

        # Remove segments that were processed in a previous run
        segments = self.segments_per_doc[self.skip_segments:]
        total_segments = len(segments)

        batches = [
            segments[i : min(i + self.batch_size, total_segments)]
            for i in range(0, total_segments, self.batch_size)
        ]
        total_batches = len(batches)
        base_batches = total_batches // self.num_replicas
        even_count = base_batches * self.num_replicas

        self.batches_for_rank = [
            batches[i]
            for i in range(even_count)
            if (i % self.num_replicas == self.rank)
        ]
        # 4. Allocate remaining extra batches solely to rank 0. Trivial max of 3 additional samples lol (num GPUs).
        if self.rank == 0:
            self.batches_for_rank.extend(batches[even_count:])
        yield from self.batches_for_rank

    def __len__(self):
        num_batches = ceil(self.effective_total_segments / self.batch_size)
        base_batches = num_batches // self.num_replicas
        return base_batches + (num_batches - base_batches if self.rank == 0 else 0)


# -------------------------------------------------------------------------
# Collate function (unchanged)
# -------------------------------------------------------------------------
def collate(sequences: list):
    return (
        # actual tokens
        pad_sequence(
            [s["tokens"] for s in sequences],
            batch_first=True,
            padding_value=PAD_TOKEN,  # StripedHyena CharTokenizer pad token. eod and eos are 0.
        ),
        [s["record"] for s in sequences],  # database id
        [s["text"] for s in sequences],  # nucleotide
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
        # MAX_LENGTH = getattr(model, "MAX_LENGTH", 8192)
        mask = tokens != PAD_TOKEN
        seq_lengths = mask.sum(dim=1).tolist()

        # NOTE: StripedHyena2 seems to output some "inference_params_dict_out" object that we don't need.
        outputs, _ = model(tokens)  # => [batch, MAX_LENGTH, vocab]
        pred = outputs[0]
        pred_entropies = (
            entropy(pred).to(dtype=torch.float16, device="cpu").numpy()
        )  # => [batch, seq_len]
        mask_np = mask.cpu().numpy()

        valid_pred_entropies = pred_entropies[mask_np]
        split_indices = np.cumsum(seq_lengths)[:-1]
        nested_entropies = np.split(valid_pred_entropies, split_indices)

    return nested_entropies, sum(seq_lengths)


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
    num_tokens=4e10,
):
    # Set environment variables for master address and port
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % gpu_per_node)
        device = torch.device("cuda", rank % gpu_per_node)
    else:
        device = torch.device("cpu")

    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    # Synchronize all processes
    dist.barrier()

    # Message indicating the process has passed the barrier
    print(f"Process {rank} passed barrier")
    hf_data = load_dataset(f"LongSafari/open-genome", "stage1", split=split).with_format("torch")
    # ---------------------------------------------------------
    # Count how many segments have already been processed
    # across all previously saved rank files so we can
    # resume even if the number of ranks has changed.
    # ---------------------------------------------------------
    processed_segments = 0
    if rank == 0:
        import glob

        for afile in glob.glob("/large_storage/goodarzilab/ashah/16b*.arrow"):
            try:
                with pa.memory_map(afile, "r") as src:
                    reader = pa.ipc.open_file(src)
                    for i in range(reader.num_record_batches):
                        processed_segments += reader.get_record_batch(i).num_rows
            except Exception as e:
                print(f"Rank 0: Could not read {afile}: {e}")

    processed_segments_tensor = torch.tensor([processed_segments], dtype=torch.long, device=device)
    dist.broadcast(processed_segments_tensor, src=0)
    processed_segments = processed_segments_tensor.item()

    dist_sampler = LengthAwareDistributedBatchSampler(
        hf_data,
        batch_size,
        num_replicas=world_size,
        rank=rank,
        segment_size=MAX_LENGTH,
        skip_segments=processed_segments,
    )
    data = EntropyDataset(hf_data)

    # Standard PyTorch DistributedSampler (removes your custom length-sorting).
    # If you need length-based sorting globally, you need a custom distributed sampler.

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
    rank_output_file = f"/large_storage/goodarzilab/ashah/16b{rank+1}_1.arrow"


    entropy_model = Evo2("evo2_1b_base", device=rank)
    entropies_buffer = []
    text_buffer = []
    sample_ids_buffer = []
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])
    processed_tokens = 0

    try:
        with pa.OSFile(rank_output_file, "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                data_iter = iter(dataloader)  # Sampler already skips processed segments
                for tokens, sample_ids, texts in data_iter:
                    tokens = tokens.to(
                        dtype=torch.int, device=device  # push tokens to chosen device
                    )

                    # Calculate entropies
                    start_time = time.time()
                    scores, batch_tokens = calculate_entropies(
                        tokens, entropy_model, device=device
                    )
                    end_time = time.time()
                    print(
                        f"Took {end_time - start_time} seconds to process {tokens.shape[0]}."
                    )
                    entropies_buffer.extend(scores)
                    text_buffer.extend(texts)
                    sample_ids_buffer.extend(sample_ids)

                    processed_tokens += batch_tokens

                    if len(entropies_buffer) >= arrow_batch:
                        # Create pyarrow table
                        batch = pa.record_batch(
                            {
                                "entropies": entropies_buffer,
                                "text": text_buffer,
                                "sample_id": sample_ids_buffer,
                            },
                            schema,
                        )

                        # Use 'ab' (append binary) mode for file operations
                        # For the first write, this is same as 'wb', for subsequent writes it appends

                        writer.write_batch(batch)

                        entropies_buffer = []
                        text_buffer = []
                        sample_ids_buffer = []

                        if processed_tokens >= num_tokens:
                            break

                if len(entropies_buffer) > 0:
                    batch = pa.record_batch(
                        {
                            "entropies": entropies_buffer,
                            "text": text_buffer,
                            "sample_id": sample_ids_buffer,
                        },
                        schema,
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
        dist.barrier()
        dist.destroy_process_group()


def main_worker(local_rank, args):
    # Determine node_rank safely—works whether or not we're inside a SLURM job.
    node_rank_env = os.environ.get("SLURM_PROCID")
    if node_rank_env is not None:
        node_rank = int(node_rank_env)
        print(f"node rank (from SLURM_PROCID): {node_rank}")
    else:
        node_rank = 0
        print("SLURM_PROCID not found; defaulting node rank to 0")
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
        num_tokens=args.num_tokens,
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
    parser.add_argument(
        "--num_tokens",
        type=int,
        required=True,
        help="The number of tokens to process entropies",
    )

    args = parser.parse_args()

    # Use mp.spawn to launch multiple processes, each corresponding to a GPU

    os.environ["HF_HOME"] = args.data_path
    os.environ["HF_DATASETS_CACHE"] = args.data_cache_dir
    config.HF_DATASETS_CACHE = args.data_cache_dir

    mp.spawn(main_worker, nprocs=args.gpu_per_node, args=(args,))
