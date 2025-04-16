import pyarrow as pa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def block_lengths(mask):
    """
    Given a boolean mask where True means the value is under the threshold and
    False means above the threshold, return a list of block lengths.

    Consecutive True values are counted as one block with length equal to the 
    count of True values. Each False value is counted as a block of length 1.
    """
    blocks = []
    current_run = 0
    for value in mask:
        if value:
            current_run += 1
        else:
            # If there was a run of True values, record it first
            if current_run > 0:
                blocks.append(current_run)
                current_run = 0
            # Record the False value as a block of length 1
            blocks.append(1)
    
    # If the mask ends with a True run, record it
    if current_run > 0:
        blocks.append(current_run)
    
    return blocks

# Open the Arrow file and load the DataFrame
with pa.memory_map("entropies_rank0.arrow", "r") as source:
    reader = pa.ipc.open_file(source)
    df1 = reader.read_pandas()

# Lists to store per-sequence average block lengths and all block lengths across sequences.
average_block_lengths = []
all_block_lengths = []

for i in tqdm(range(len(df1))):
    # Create a boolean mask for tokens under the threshold.
    # Use the length of the text to truncate the entropies list.
    mask = (df1.iloc[i]['entropies'][:len(df1.iloc[i]['text'])] < 1.335442066192627).tolist()
    blocks = block_lengths(mask)
    # Save the block lengths
    all_block_lengths.extend(blocks)
    # Compute the average block length for this sequence
    avg_length = sum(blocks) / len(blocks) if blocks else 0
    average_block_lengths.append(avg_length)

# Compute aggregate statistics across all block lengths:
all_blocks_array = np.array(all_block_lengths)
mean_length = np.mean(all_blocks_array)
variance_length = np.var(all_blocks_array)
median_length = np.median(all_blocks_array)
min_length = np.min(all_blocks_array)
max_length = np.max(all_blocks_array)

print("Aggregated Block Length Statistics:")
print(f"Mean: {mean_length:.3f}")
print(f"Variance: {variance_length:.3f}")
print(f"Median: {median_length}")
print(f"Min: {min_length}")
print(f"Max: {max_length}")

# Optionally: Plot a histogram of the aggregated block lengths
plt.figure(figsize=(8, 6))
plt.hist(all_block_lengths, bins=range(1, max_length + 2), edgecolor='black', align='left')
plt.title("Distribution of Block Lengths Across All Sequences")
plt.xlabel("Block Length")
plt.ylabel("Frequency")
plt.xticks(range(1, max_length + 1))
plt.show()


# based on these stats we'll want a max of 16 bytes 