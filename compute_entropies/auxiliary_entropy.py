import os
import time
import argparse
import fsspec
import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence

from rich.progress import Progress, TextColumn
from datasets import config, load_dataset
from evo2 import Evo2

MAX_LENGTH = 8192


# -------------------------------------------------------------------------
# Collate function (unchanged)
# -------------------------------------------------------------------------
def collate(sequences: list):
    # Pad to max length in the batch
    text = [s["text"] for s in sequences]
    record = [s["record"] for s in sequences]
    bos_token = torch.tensor(
        [0], dtype=torch.uint8
    )  # Beginning of sequence token in EVO

    return (
        pad_sequence(
            [
                torch.cat(
                    (
                        bos_token,
                        torch.from_numpy(
                            np.frombuffer(bytearray(s.encode("utf-8")), dtype=np.uint8)
                        ),
                    )
                )
                for s in text
            ],
            batch_first=True,
            padding_value=1,  # StripedHyena CharTokenizer pad token. eod and eos are 0.
        ),
        record,
        text,
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
    with torch.no_grad():
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


# test_files = {
#     'test': [
#         'hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/gtdb/gtdb_test.parquet',
#         'hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/imgpr/imgpr_test.parquet'
#     ]
# }
# data = datasets.load_dataset("parquet", data_files=test_files, split="test")


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
    # data = datasets.load_dataset(f"{data_path}/stage1", split=split).with_format('torch')
    # test_files = {
    #     "test": [
    #         "/cluster/home/t136085uhn/.cache/huggingface/hub/datasets--LongSafari--open-genome/snapshots/84369c058d192dcb607086d71679b877421e3250/stage1/gtdb/gtdb_test.parquet",
    #         "/cluster/home/t136085uhn/.cache/huggingface/hub/datasets--LongSafari--open-genome/snapshots/84369c058d192dcb607086d71679b877421e3250/stage1/imgpr/imgpr_test.parquet",
    #     ]
    # }
    # data = datasets.load_dataset(
    #     "parquet", data_files=test_files, split="test"
    # ).with_format("torch")
    data = load_dataset(f"{data_path}/stage1", split=split).with_format("torch")

    # Standard PyTorch DistributedSampler (removes your custom length-sorting).
    # If you need length-based sorting globally, you need a custom distributed sampler.
    dist_sampler = DistributedSampler(
        data,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # or False, up to you
    )

    dataloader = DataLoader(
        data,
        sampler=dist_sampler,  # ensures each rank sees a unique subset
        batch_size=batch_size,
        collate_fn=collate,
        drop_last=False,
    )

    # ---------------------------------------------------------------------
    # 3. Load model & wrap with DistributedDataParallel
    # ---------------------------------------------------------------------

    entropy_model = Evo2("evo2_1b_base", device=rank)

    # ---------------------------------------------------------------------
    # 4. Prepare Arrow writing
    # ---------------------------------------------------------------------
    # We'll have each rank write to its own file: out.arrow.{rank}
    # Or gather all results to rank 0 (requires extra logic).
    rank_output_file = f"out.arrow.{rank}"
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])

    arrow_batch_size = arrow_batch
    output_fs = fsspec.filesystem("file")
    # ---------------------------------------------------------------------
    # 5. Compute entropies and write out
    # ---------------------------------------------------------------------
    try:
        with output_fs.open(rank_output_file, "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                id_buffer = []
                entropies_buffer = []
                text_buffer = []

                with Progress(
                    *Progress.get_default_columns(),
                    TextColumn("Completed: {task.completed}"),
                ) as progress:
                    task = progress.add_task(
                        f"[green]Rank {rank} calculating entropies...", total=None
                    )

                    # Each rank processes only its portion of data
                    for tokens, sample_ids, texts in dataloader:
                        tokens = tokens.to(
                            dtype=torch.int, device=rank
                        )  # push tokens to GPU

                        # Calculate entropies
                        start_time = time.time()
                        scores = calculate_entropies(tokens, entropy_model, device=rank)
                        end_time = time.time()
                        scores = scores.cpu().contiguous().view(torch.float16).numpy()
                        bsz_ = len(scores)

                        print(
                            f"Processed {bsz_} batches in {end_time - start_time} seconds"
                        )

                        for i in range(bsz_):  # should all have shape bsz
                            entropies_buffer.append(scores[i])  # shape [seq_len]
                            id_buffer.append(sample_ids[i])  # a single sample_id
                            text_buffer.append(texts[i])

                        # Write to arrow in chunks
                        if len(entropies_buffer) == arrow_batch_size:
                            batch = pa.record_batch(
                                {
                                    "entropies": entropies_buffer,
                                    "sample_id": id_buffer,
                                    "text": text_buffer,
                                },
                                schema,
                            )
                            writer.write(batch)

                            entropies_buffer.clear()
                            id_buffer.clear()
                            text_buffer.clear()

                        progress.update(task, advance=1)

                    # Write any leftover items in buffers
                    if len(entropies_buffer) > 0:
                        batch = pa.record_batch(
                            {
                                "entropies": entropies_buffer,
                                "sample_id": id_buffer,
                                "text": text_buffer,
                            },
                            schema,
                        )
                        writer.write(batch)

        # Touch a .complete for each rank's file
        output_fs.touch(f"{rank_output_file}.complete")

    except Exception as e:
        # Clean up partial file if something goes wrong
        if output_fs.exists(rank_output_file):
            output_fs.rm(rank_output_file)
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
