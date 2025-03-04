import pyarrow as pa
from tqdm import tqdm


def count_mask_occurrences(mask):
    from collections import defaultdict

    counts = defaultdict(int)
    false_count = 0

    for value in mask:
        if value:
            # If there was a run of False values, record it
            if false_count > 0:
                counts[false_count] += 1
                false_count = 0
            # Count the True value as 1
            counts[1] += 1
        else:
            # Increment the False run counter
            false_count += 1

    # After the loop, check if the last elements were False
    if false_count > 0:
        counts[false_count] += 1

    return dict(counts)


rs = 0
with pa.memory_map("outputs/out.arrow.0", "r") as source:
    loaded_arrays_1 = pa.ipc.open_file(source).read_all()

with pa.memory_map("outputs/out.arrow.1", "r") as source:
    loaded_arrays_2 = pa.ipc.open_file(source).read_all()

df1 = loaded_arrays_1.to_pandas()["entropies"]
df2 = loaded_arrays_2.to_pandas()["entropies"]

for entropies in tqdm(df1):
    mask = (entropies > 1.335442066192627).tolist()
    # total = sum(count * occurrences for count, occurrences in count_mask_occurrences(mask).items())
    total_occurrences = sum(count_mask_occurrences(mask).values())
    mean = len(mask) / total_occurrences
    rs += mean

for entropies in tqdm(df2):
    mask = (entropies > 1.335442066192627).tolist()
    # total = sum(count * occurrences for count, occurrences in count_mask_occurrences(mask).items())
    total_occurrences = sum(count_mask_occurrences(mask).values())
    mean = len(mask) / total_occurrences
    rs += mean

rs /= len(df1) + len(df2)  # should be 50
print(rs)
