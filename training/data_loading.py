import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from math import ceil
from collections import defaultdict
from tqdm import tqdm

MAX_LENGTH = 8192
PAD_TOKEN = 1

# -------------------------------------------------------------------------
# Custom dataset
# -------------------------------------------------------------------------
class DNADataset(Dataset):
    def __init__(self, hf_dataset):
        """
        hf_dataset: A HuggingFace dataset or similar structure where
                    hf_dataset[i] = {"text": <string>, "record": <some_id>} 
        """
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        idx is of format (doc_id, start_idx, length),
        which we generate externally using the sampler logic.
        """
        doc_id, start_idx, length = idx
        sample = self.dataset[doc_id]

        text_sample = sample["text"][start_idx : start_idx + length]
        record = f"{sample['record']}_{start_idx}_{length}"

        # Convert string to a torch tensor of uint8
        tokenized_text = torch.from_numpy(
            np.frombuffer(bytearray(text_sample.encode("utf-8")), dtype=np.uint8)
        )

        return {
            "tokens": tokenized_text,
            "record": record,
            "text": text_sample,
        }

# -------------------------------------------------------------------------
# Custom distributed sampler with length-based bucketing
# -------------------------------------------------------------------------
class LengthAwareDistributedBatchSampler(Sampler):
    """
    A sampler that:
      1. Splits the dataset into segments (doc_id, start_idx, length).
      2. Groups segments by length (or approximate length).
      3. Shuffles within each length bucket.
      4. Chunks buckets into batches (ignoring remainder).
      5. Shuffles the overall list of batches.
      6. Distributes batches among ranks for multi-GPU training.

    The data_source here is the entire hf_dataset, but we rely on the logic
    in __init__ to build 'segments_per_doc'. Each segment is a possible
    index for EntropyDataset.
    """
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=42,
        segment_size=1024,
    ):
        """
        dataset:        The hf_dataset 
        batch_size:     Number of segments in each batch.
        num_replicas:   Number of distributed processes (world_size).
        rank:           Current process rank.
        shuffle:        Whether to shuffle the final batches.
        seed:           Random seed for reproducibility.
        segment_size:   Max segment length to chunk from each doc.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas or 1
        self.rank = rank or 0
        self.shuffle = shuffle
        self.seed = seed
        self.segment_size = segment_size

        # Build a list of all possible segments: (doc_id, start_idx, length)
        self.segments_per_doc = []
        random.seed(self.seed)

        # Step 1: Create segments of size `segment_size` (plus remainder).
        #         Start offset is randomized so we shift the leftover.
        for i, sample in enumerate(tqdm(self.dataset, desc="Building segments")):
            text_len = len(sample["text"])
            quotient, remainder = divmod(text_len, self.segment_size)

            # we get a random start offset for the first leftover portion
            start = random.randint(0, remainder) if remainder > 0 else 0
            end = remainder - start

            # initial leftover chunk at the front:
            if start > 0:
                self.segments_per_doc.append((i, 0, start))

            #  slice out uniform segments
            offset = start
            for _ in range(quotient):
                self.segments_per_doc.append((i, offset, self.segment_size))
                offset += self.segment_size

            # leftover at the end
            if end > 0:
                self.segments_per_doc.append((i, offset, end))

        self.total_segments = len(self.segments_per_doc)

        # over here, we are grouping segments by length, but we amy want to change this to a range?
        # suppose we have segments of size 500, 501, 502 and 1 of each, then we'll only have 1 segment in each bucket
        self.length_buckets = defaultdict(list)
        for seg in self.segments_per_doc:
            # seg is (doc_id, start_idx, length)
            length = seg[2]
            self.length_buckets[length].append(seg)

        #  shuffle segments, chunk into full batches ignoring remainders
        self.all_batches = self._create_batches()

        # shuffle the entire batch list
        if self.shuffle:
            random.shuffle(self.all_batches)

        self.total_batches = len(self.all_batches)

    def _create_batches(self):
        """ Turn each length bucket into many full-size batches. """
        all_batches = []
        random.seed(self.seed)

        for length, seg_list in self.length_buckets.items():
            # Shuffle within this bucket
            random.shuffle(seg_list)

            # Chunk into full batches 
            # this is also where we only care about full batches and get rid of the remainder
            # if we want to include reaminders, we simply get rid of the if condition

            for i in range(0, len(seg_list), self.batch_size):
                if i + self.batch_size <= len(seg_list):
                    batch = seg_list[i : i + self.batch_size]
                    all_batches.append(batch)

        return all_batches

    def __iter__(self):
        """
        For distributed training:
          - We need to split self.all_batches among the available ranks.
          - We'll do a standard approach: each rank takes every N-th batch.
          - If there's leftover batches that don't evenly divide, rank 0 might
            get them 
        """
        if self.shuffle:
            random.seed(self.seed + self.rank)  # differ by rank
            random.shuffle(self.all_batches)

        total_batches = len(self.all_batches)
        base_batches = total_batches // self.num_replicas
        even_count = base_batches * self.num_replicas

        # The "main" portion, evenly split across ranks
        batches_for_rank = [
            self.all_batches[i]
            for i in range(even_count)
            if (i % self.num_replicas == self.rank)
        ]

        # Remainder goes to rank 0
        if self.rank == 0:
            batches_for_rank.extend(self.all_batches[even_count:])

        # Yield each batch (list of segments)
        for batch in batches_for_rank:
            yield batch

    def __len__(self):
        """
        Number of batches that *this rank* will process.
        We compute total number of batches, then figure out how many
        are assigned to this rank.
        """
        num_batches = self.total_batches
        base_batches = num_batches // self.num_replicas
        remainder = num_batches - (base_batches * self.num_replicas)
        # Rank 0 gets the remainder
        if self.rank == 0:
            return base_batches + remainder
        else:
            return base_batches

# -------------------------------------------------------------------------
# USAGE EXAMPLE
# -------------------------------------------------------------------------
if __name__ == "__main__":

    mock_dataset = [
        {"text": "ACGTACGTAC", "record": "genome_1"},
        {"text": "ACGT", "record": "genome_2"},
        {"text": "ACGACGACGACG", "record": "genome_3"},
    ]

    """
    Ok so let's walk through this example above:

    for the first text / record, we have length 10. then length 4, and lastly length 12. Note that we are going to set segment_size = 5 for this example.

    Document 0:

    length = 10, segment_size = 5, quotient = 2, remainder = 0 (this is computed by 10 / 5 = 2)
    for the random offset, we have remainder = start = 0
    thus, there is no initial leftover chunk

    the output segments:
    (0,0,5) -> from index 0 to 5 -> ACGTA
    (0,5,5) -> from index 5 to 10 -> CGTAC

    there will be no leftover at the end since remainder = 0
    
    another thing to note is that since we discard remainder sequences, we won't print the 2nd one as it has length 4
    
    
    """

    #  create the distributed sampler with the raw data.
    batch_size = 2
    sampler = LengthAwareDistributedBatchSampler(
        dataset=mock_dataset,
        batch_size=batch_size,
        num_replicas=1,
        rank=0,
        shuffle=False,
        seed=42,
        segment_size=5,  # break text into segments of length 5 for example
    )


    entropy_dataset = DNADataset(mock_dataset)


    loader = DataLoader(
        entropy_dataset,
        batch_sampler=sampler,
        collate_fn=lambda x: x,  
    )

    print("=== Starting iteration ===")
    for step, batch_segments in enumerate(loader):
        print(f"\nStep {step}:")
        for seg in batch_segments:
            print("Tokens:", seg["tokens"], "Record:", seg["record"], "Text:", seg["text"])
