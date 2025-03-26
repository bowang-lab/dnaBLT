# Training Overview for BLT


## Overview

The primary entry point is **`train.py`**. This file contains the main training loop as well as the distributed GPU training code. However, not all the work happens in `train.py` itself. In particular, the magic happens in how data is read and preprocessed before training begins. Two important components of this pipeline are:

1. **Arrow Iteration:** Reading raw data (e.g., text and precomputed entropies) from Arrow (or JSON) files.
2. **Patching:** Segmenting tokenized text into “patches” (or chunks) based on configurable rules (like entropy, BPE boundaries, spaces, or static sizes).

The key files handling these tasks are:

- **`arrow_iterator.py`**: Contains code to load and iterate over Arrow files.
- **`patcher.py`**: Defines various patching methods for splitting token sequences.
- **`preprocess_iterator.py`**: Integrates the Arrow iterator with the patching (and tokenization) steps to produce fully preprocessed examples.

## File Breakdown

### 1. `train.py`

- **Purpose:**  
  Orchestrates the training process, including distributed GPU training.
  
- **Key Detail:**  
  The training script imports `TrainArgs`, which—under the hood—pulls in the iterators and patching logic from the other files. This is how `train.py` ends up using the Arrow Iterator and patching methods even though they’re defined in separate files.

### 2. `arrow_iterator.py`

- **Purpose:**  
  Handles reading raw data from Arrow (or JSON) files.

- **How It Works:**
  - **Initialization:**  
    The iterator sets up file paths and determines whether to read from local disk or S3 (using libraries like `fsspec` and `pyarrow`).
  - **Dataset Creation:**  
    It uses `pyarrow.dataset.dataset` to load one or more Arrow shards.  
  - **Batching:**  
    Data is loaded in batches (using the `to_batches()` method) and then converted to Python dictionaries.
  - **Row Yielding:**  
    Each row is wrapped in a `BltExample` object with fields such as `sample_id`, `text`, and (optionally) `entropies`. Multi-worker sharding is performed by yielding only the rows that belong to the current worker.

### 3. `patcher.py`

- **Purpose:**  
  Contains various methods to “patch” token sequences, i.e., to split tokens into meaningful chunks.
  
- **Patching Modes:**  
  The file defines an enum (`PatchingModeEnum`) with several modes:
  - **`entropy`**: Uses model-based entropy to determine where a new patch should begin.
  - **`bpe`**: Splits based on BPE delimiter tokens.
  - **`bpe_patcher`**: Uses a learned model to predict patch boundaries.
  - **`space`**: Uses space-like tokens (or other heuristics) as delimiters.
  - **`static`**: Divides tokens into fixed-size patches.
  - **`byte`**: Treats each token as its own patch.

- **Key Functions:**  
  Functions such as `calculate_entropies`, `find_entropy_patch_start_ids`, and `patch_lengths_from_start_ids` work together to process token sequences and return patch lengths.

### 4. `preprocess_iterator.py`

- **Purpose:**  
  Acts as an integration layer that takes raw examples from the Arrow iterator and applies further preprocessing, such as tokenization and patching.

- **Workflow:**
  1. **Reading Data:**  
     It starts by using an underlying iterator (for example, an instance of `ArrowFileIterator`) to load raw examples. These examples come as `BltExample` objects containing basic fields like `sample_id`, `text`, and (optionally) `entropies`.
  2. **Tokenization:**  
     If enabled (via `add_tokens`), the iterator uses a tokenizer (built from `TokenizerArgs`) to convert the raw text into token IDs.
  3. **Patching:**  
     If patching is enabled (via `add_patches`), it uses the patcher (constructed from `PatcherArgs`) to compute patch lengths for the tokenized input. For example, when using the **entropy** patching mode, the patcher requires entropy information from the example and computes boundaries accordingly.
  4. **Yielding Preprocessed Examples:**  
     Finally, a new `BltExample` is yielded with all the additional fields populated (tokens, mask, patch lengths, etc.).

- **Key Method:**  
  The `create_iter()` method in `PreprocessIterator` orchestrates the entire process:
  - It instantiates the tokenizer and patcher if needed.
  - Iterates over raw examples, tokenizes the text, and applies patching.
  - Outputs enhanced `BltExample` objects that are ready for the training loop.