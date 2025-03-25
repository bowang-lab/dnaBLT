import random
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import pyarrow as pa
from tqdm import tqdm

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
    ):
        # NOTE: This should just return the indices of the segments. We'll need to define them all and then split them into batches and allocate to ranks.

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.segment_size = segment_size

        self.segments_per_doc = []

        # TOTAL_SEGMENT_FORMAT: (doc_id, start_idx, length)
        random.seed(self.seed)

        for i, sample in enumerate(tqdm(self.dataset)):
            quotient, remainder = divmod(len(sample["text"]), self.segment_size)
            # [ACT][GAC][TGA][CT] when start = 0. [A][CTG][ACT][GAC][T] when start = 1. [AC][TGA][CTG][ACT] when start = 2.
            start = random.randint(0, remainder)
            end = remainder - start
            if start:
                self.segments_per_doc.append((i, 0, start))

            for x in range(quotient):
                self.segments_per_doc.append((i, start, self.segment_size))
                start += self.segment_size

            if end:
                self.segments_per_doc.append((i, start, end))

        self.total_segments = len(self.segments_per_doc)

    def __iter__(self):
        # 1. Sort indices by sequence length.
        self.segments_per_doc.sort(key=lambda tuple: tuple[2], reverse=True)

        batches = [
            self.segments_per_doc[i : min(i + self.batch_size, self.total_segments)]
            for i in range(0, self.total_segments, self.batch_size)
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
        num_batches = ceil(self.total_segments / self.batch_size)
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
        # outputs, _ = model(tokens)  # => [batch, MAX_LENGTH, vocab]
        # pred = outputs[0]
        # pred_entropies = (
        #     entropy(pred).to(dtype=torch.float16, device="cpu").numpy()
        # )  # => [batch, seq_len]

        pred_entropies = torch.rand(
            (tokens.shape[0], tokens.shape[1]), dtype=torch.float16, device=device
        ).numpy()
        mask_np = mask.cpu().numpy()
        valid_pred_entropies = pred_entropies[mask_np]
        split_indices = np.cumsum(seq_lengths)[:-1]
        nested_entropies = np.split(valid_pred_entropies, split_indices)

    return nested_entropies, sum(seq_lengths)


def init_distributed_training(
    rank,
    batch_size,
):
    # Set environment variables for master address and port
    test_files = {
        "test": [
            "hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/gtdb/gtdb_test.parquet",
            "hf://datasets/LongSafari/open-genome@84369c058d192dcb607086d71679b877421e3250/stage1/imgpr/imgpr_test.parquet",
        ]
    }
    hf_data = load_dataset("parquet", data_files=test_files, split="test").with_format(
        "torch"
    )
    dist_sampler = LengthAwareDistributedBatchSampler(
        hf_data, batch_size, num_replicas=2, rank=rank, segment_size=MAX_LENGTH
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
    entropies_buffer = []
    text_buffer = []
    sample_ids_buffer = []
    entropy_field = pa.field("entropies", pa.list_(pa.float16()), nullable=False)
    text_field = pa.field("text", pa.string(), nullable=False)
    sample_id_field = pa.field("sample_id", pa.string(), nullable=False)
    schema = pa.schema([sample_id_field, text_field, entropy_field])
    processed_tokens = 0
    try:
        with pa.OSFile("test_inference.arrow", "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                data_iter = iter(dataloader)
                for tokens, sample_ids, texts in data_iter:
                    tokens = tokens.to(dtype=torch.int, device="cpu")
                    scores, batch_tokens = calculate_entropies(
                        tokens, None, device="cpu"
                    )
                    entropies_buffer.extend(scores)
                    text_buffer.extend(texts)
                    sample_ids_buffer.extend(sample_ids)

                    processed_tokens += batch_tokens

                    if len(entropies_buffer) >= 16:
                        # Create pyarrow table
                        batch = pa.record_batch(
                            {
                                "entropies": entropies_buffer,
                                "text": text_buffer,
                                "sample_id": sample_ids,
                            },
                            schema,
                        )

                        # Use 'ab' (append binary) mode for file operations
                        # For the first write, this is same as 'wb', for subsequent writes it appends

                        writer.write_batch(batch)

                        entropies_buffer = []
                        text_buffer = []
                        sample_ids_buffer = []

                        if processed_tokens >= 4e10:
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
        raise e


if __name__ == "__main__":
    init_distributed_training(rank=0, batch_size=16)
