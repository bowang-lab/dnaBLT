import os
import random
import fsspec
import numpy as np
import pyarrow as pa
import torch
import torch.distributed as dist
from evo import Evo

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from rich.progress import Progress, TextColumn

import datasets

# -------------------------------------------------------------------------
# Collate function (unchanged)
# -------------------------------------------------------------------------
def collate(sequences: list):
    items = []
    records = []
    texts = []
    for seq in sequences:
        items.append(torch.tensor(seq['tokens'], dtype=torch.int))
        records.append(seq['record'])
        texts.append(seq['text'])
    # Pad to max length in the batch
    return pad_sequence(items, batch_first=True, padding_value=0), records, texts

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

def calculate_entropies(tokens: torch.tensor, model: torch.nn.Module, device: str = "cuda"):
    """
    tokens: [batch_size, seq_len]
    Return shape: [batch_size, seq_len]
    """
    with torch.no_grad():
        entropies = []
        # max_length = getattr(model, "max_length", 8192)
        max_length = 8192
        batch_numel = max_length * tokens.size(0)

        # Flatten and split into blocks of (batch_size * max_length)
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(pad_size, dtype=split.dtype, device=split.device)
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)

            split = split.to(device)

            # Forward pass
            pred, _ = model(split)  # => [batch, max_length, vocab]
            # Remove padding
            pred = pred.reshape(-1, pred.shape[-1])[: split.numel() - pad_size, :]

            pred_entropies = entropy(pred)  # => [batch * seq_len]
            entropies.append(pred_entropies)

        concat_entropies = torch.cat(entropies, dim=0)
        concat_entropies = concat_entropies.reshape(tokens.shape)
    return concat_entropies

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(
    output_file: str = "out.arrow",
    patching_device: str = "cuda",
    entropy_model_checkpoint_dir: str = "public_data/entropy_checkpoint",
    entropy_model_state_dict_path: str = "public_data/entropy_model.pth",
    local_rank: int = 0,  # DDP: each process gets a different local rank
):
    """
    Demonstration of torch.nn.DistributedDataParallel.
    Run with something like:
      torchrun --nproc_per_node=NUM_GPUS script.py --output-file out.arrow
    """
    # ---------------------------------------------------------------------
    # 1. Initialize process group
    # ---------------------------------------------------------------------

    real_local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(real_local_rank)

    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()  # global rank
    device = torch.device(f"cuda:{real_local_rank}")
    print(real_local_rank, rank)

    # ---------------------------------------------------------------------
    # 2. Load dataset on each rank
    #    (Each process will see the same data, but a DistributedSampler
    #     ensures non-overlapping subsets.)
    # ---------------------------------------------------------------------
    test_files = {
        'test': [
            'hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/gtdb/gtdb_test.parquet', 
            'hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/imgpr/imgpr_test.parquet'
        ]
    }
    data = datasets.load_dataset("parquet", data_files=test_files, split="test")
    # data = datasets.load_dataset("/projects/llm/open-genome", "stage1")

    evo_model = Evo('evo-1-8k-base')
    entropy_model, tokenizer = evo_model.model, evo_model.tokenizer
    entropy_model.to(device)
    entropy_model.eval()

    print("Begin mapping data")
    encoded_dataset = data.map(
        lambda examples: {"tokens": tokenizer.tokenize_batch(examples['text'])},
        batched=True,
        batch_size=100,
    )

    print("Ended mapping data")

    # Standard PyTorch DistributedSampler (removes your custom length-sorting).
    # If you need length-based sorting globally, you need a custom distributed sampler.
    dist_sampler = DistributedSampler(
        encoded_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True  # or False, up to you
    )

    batch_size = 32
    dataloader = DataLoader(
        encoded_dataset,
        sampler=dist_sampler,       # ensures each rank sees a unique subset
        batch_size=batch_size,
        collate_fn=collate,
        drop_last=False
    )

    # ---------------------------------------------------------------------
    # 3. Load model & wrap with DistributedDataParallel
    # ---------------------------------------------------------------------
    # entropy_model = load_entropy_model(
    #     entropy_model_checkpoint_dir,
    #     entropy_model_state_dict_path,
    #     device=device
    # )
    # Move to device and wrap with DDP
    print("Begin DDP model")
    entropy_model = DDP(entropy_model, device_ids=[real_local_rank])
    print("End DDP model")

    # ---------------------------------------------------------------------
    # 4. Prepare Arrow writing
    # ---------------------------------------------------------------------
    # We'll have each rank write to its own file: out.arrow.{rank}
    # Or gather all results to rank 0 (requires extra logic).
    rank_output_file = f"{output_file}.{rank}"
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])

    arrow_batch_size = 10
    print("Begin fsspec shit")
    output_fs = fsspec.filesystem("file")
    print("End fsspec shit")
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
                    TextColumn("Completed: {task.completed}")
                ) as progress:
                    task = progress.add_task(
                        f"[green]Rank {rank} calculating entropies...", total=None
                    )

                    # Each rank processes only its portion of data
                    iter = 0
                    for tokens, sample_ids, texts in dataloader:
                        print(f"BEGIN ITERATION {iter}")
                        tokens = tokens.to(device)  # push tokens to GPU
                        
                        # Calculate entropies
                        scores = calculate_entropies(tokens, entropy_model, device=device)
                        scores = scores.cpu().numpy().astype(np.float16)

                        entropies_buffer.append(scores)
                        id_buffer.append(sample_ids)
                        text_buffer.append(texts)

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
                        iter += 1

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

# -------------------------------------------------------------------------
# Typer Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Example usage:

      torchrun --nproc_per_node=2 your_script.py \
         --output-file my_entropy_output.arrow \
         --patching-device cuda

    Each process (rank) will produce a separate file:
      my_entropy_output.arrow.0
      my_entropy_output.arrow.1
    """
    main()
