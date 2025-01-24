# export HF_HOME=/workspace/DNABLT/.hf_cache
import random
from torch.utils.data import Sampler
import datasets
from arnav_tokenizer import CharLevelTokenizer
from torch.nn.utils.rnn import pad_sequence
import torch

import fsspec
import numpy as np
import pyarrow as pa
import torch
import typer
from rich.progress import Progress, TextColumn

# from bytelatent.data.file_util import get_fs
# from bytelatent.data.patcher import calculate_entropies
# from bytelatent.entropy_model import load_entropy_model
# from bytelatent.blt_tokenizers.build_tokenizer import TokenizerArgs

# from transformers import AutoModel


def collate(sequences: list):
    items = []
    records = []
    text = []
    for seq in sequences:
        items.append(torch.tensor(seq['tokens'], dtype=torch.int))
        records.append(seq['record'])
        text.append(seq['text'])
    return pad_sequence(items, batch_first=True, padding_value=0), records, text

class CustomBatchSampler(Sampler):
    def __init__(self, data, batch_size, shuffle=True):
        """
        Custom batch sampler to group sequences of similar lengths into batches.
        
        Args:
            data (list of dict): The dataset, where each item is a dictionary containing a 'text' key.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data indices before batching.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        # Get indices of data
        indices = list(range(len(self.data)))
        
        # Shuffle if required
        if self.shuffle:
            random.shuffle(indices)
        
        # Sort indices by the length of the 'text' field
        sorted_indices = sorted(indices, key=lambda i: len(self.data[i]['text']))

        # Yield batches of indices
        for i in range(0, len(sorted_indices), self.batch_size):
            yield sorted_indices[i:i + self.batch_size]

    def __len__(self):
        # Number of batches
        return (len(self.data) + self.batch_size - 1) // self.batch_size


def entropy(scores):
    """
    scores: [bs, seq_len, vocab]
    returns [bs, seq_len]

    Computes the entropy for each token in the batch.
    Note: uses natural log.
    """
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.sum(dim=-1)
    return entropy

def calculate_entropies(
    tokens: torch.tensor, entropy_model, device: str | None = None
):
    """
    tokens: 2D tensor of shape [batch_size, seq_len]
    Return 2D tensor of shape [batch_size, seq_len] with entropies for each token.

    Splits the tokens into chunks of size max_length and calculates entropies for each chunk.
    Entropy model can be executed on cpu or gpu, specify either 'cuda' or 'cpu' in the device argument.
    """
    with torch.no_grad():
        entropies = []
        max_length = getattr(entropy_model, "max_length", 8192)
        batch_numel = max_length * tokens.size(0)
        splits = torch.split(tokens.flatten(), batch_numel)
        for split in splits:
            pad_size = (max_length - (split.numel() % max_length)) % max_length
            pad = torch.zeros(
                pad_size, dtype=split.dtype, device=split.device, requires_grad=False
            )
            split = torch.cat((split, pad), dim=0)
            split = split.reshape(-1, max_length)
            if device is not None:
                split = split.to(device)
            assert torch.all(split >= 0) and torch.all(split < 260)
            pred = entropy_model(split)
            pred = pred.reshape(-1, pred.shape[-1])[
                : split.numel() - pad_size, :
            ]  # [batch_size * seq_len, vocab]
            pred_entropies = entropy(pred)
            entropies.append(pred_entropies)

        concat_entropies = torch.cat(entropies, dim=0)
        concat_entropies = concat_entropies.reshape(tokens.shape)
    return concat_entropies


def main(
    output_file: str,
    patching_device: str = "cuda",
    entropy_model_checkpoint_dir: str = "public_data/entropy_checkpoint",
    entropy_model_state_dict_path: str = "public_data/entropy_model.pth",
):
    # CHANGE ONCE ON GPUS (use evo model and tokenizer, load train stage1 dataset)
    validation_files = {'validation': ['hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/gtdb/gtdb_valid_small.parquet', 'hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/imgpr/imgpr_valid_small.parquet']}
    sample_data = datasets.load_dataset("parquet", data_files=validation_files, split="validation")
    encoded_dataset = sample_data.map(lambda examples: {"tokens": tokenizer.tokenize_batch(examples['text'])}, batched=True, batch_size=100)
    valid_loader = torch.utils.data.DataLoader(encoded_dataset, collate_fn=collate, batch_sampler=CustomBatchSampler(encoded_dataset, batch_size=32))

    entropy_model = load_entropy_model(
        entropy_model_checkpoint_dir,
        entropy_model_state_dict_path,
        device=patching_device,
    )

    tokenizer = CharLevelTokenizer(512)

    patching_batch_size = 32
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])
    arrow_batch_size = 2

    output_fs = fsspec.filesystem("file")

    try:
        with output_fs.open(output_file, "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                id_buffer = []
                entropies_buffer = []
                text_buffer = []
                with Progress(
                    *Progress.get_default_columns(),
                    TextColumn("Completed: {task.completed}"),
                ) as progress:
                    task = progress.add_task(
                        "[green]Calculating entropies...", total=None
                    )
                    for tokens, labels, text in valid_loader:
                        scores = calculate_entropies(
                            tokens,
                            entropy_model,
                            patching_batch_size,
                            patching_device,
                        )
                        entropies_buffer.append(
                            np.array(scores.tolist(), dtype=np.float16)
                        )
                        id_buffer.append(labels)
                        text_buffer.append(text)
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
                            entropies_buffer = []
                            id_buffer = []
                            text_buffer = []
                        progress.update(task, advance=1)
                    if len(entropies_buffer) > 0:
                        # Write last things
                        batch = pa.record_batch(
                            {
                                "entropies": entropies_buffer,
                                "sample_id": id_buffer,
                                "text": text_buffer,
                            },
                            schema,
                        )
                        writer.write(batch)
                        entropies_buffer = []
                        id_buffer = []
                        text_buffer = []
        output_fs.touch(f"{output_file}.complete")
    except:
        if output_fs.exists(output_file):
            output_fs.rm(output_file)
        raise


if __name__ == "__main__":
    typer.run(main)
