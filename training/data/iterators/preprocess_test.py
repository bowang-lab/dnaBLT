import os
import tempfile
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from dataclasses import dataclass
from typing import List, Optional
from preprocess_iterator import PreprocessIterator
from arrow_iterator import ArrowFileIterator

# Import the PreprocessIterator from your module.
# from your_module import PreprocessIterator

# For this test we define a dummy version of BltExample.
@dataclass
class DummyBltExample:
    sample_id: str
    text: str
    tokens: Optional[List[int]] = None
    mask: Optional[List[bool]] = None
    patch_lengths: Optional[List[int]] = None
    entropies: Optional[List[float]] = None

# --- Dummy Implementations for Testing ---

# We'll monkey-patch PreprocessIterator to yield DummyBltExample
# instead of a full-blown BltExample.
# In your real code, BltExample is defined elsewhere.
class DummyPreprocessIterator(PreprocessIterator):
    def create_iter(self) -> List[DummyBltExample]:
        self._init_preprocessing_tools()
        results = []
        for example in self.base_iterator.create_iter():
            tokens = self._process_tokens(example)
            patch_lengths, entropies = self._process_patches(example, tokens)
            # Create a DummyBltExample as output.
            results.append(DummyBltExample(
                sample_id=example.sample_id,
                text=example.text,
                tokens=tokens,
                mask=[True] * len(tokens) if tokens else None,
                patch_lengths=patch_lengths,
                entropies=entropies,
            ))
        return results

# Dummy Tokenizer: converts text to list of ASCII codes.
class SimpleTokenizer:
    def __init__(self):
        self.n_words = 4
        self.unk_token = 78
    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))
    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode("utf-8")

# Dummy TokenizerArgs that builds our SimpleTokenizer.
class DummyTokenizerArgs:
    name = "simple"
    def build(self):
        return SimpleTokenizer()

# Dummy Patcher that simulates entropy-based patching.
class DummyPatcher:
    patching_mode = "entropy"  # Simulate PatchingModeEnum.entropy
    def patch(self, tokens_tensor, include_next_token, entropies):
        # For simplicity, let the patch length be the length of the token sequence.
        token_length = len(tokens_tensor[0])
        # Return nested list structure so that:
        # patch_lengths = self.patcher.patch(...)[0][0].tolist()
        return [[torch.tensor([token_length])]]
    
# Dummy PatcherArgs that builds the DummyPatcher.
class DummyPatcherArgs:
    patching_mode = "entropy"
    def build(self):
        return DummyPatcher()

# We'll use ArrowFileIterator as our base iterator.
# For testing, we'll write a temporary Arrow file with two rows.
from data.iterators.arrow_iterator import ArrowFileIterator

def create_dummy_arrow_file(temp_dir: str) -> str:
    temp_file = os.path.join(temp_dir, "dummy.arrow")
    # Create a table with three columns: sample_id, text, and entropies.
    table = pa.Table.from_pydict({
        "sample_id": ["1", "2"],
        "text": ["Hello", "World"],
        # Each row's entropies is a list of 5 float values.
        "entropies": [[0.1, 0.2, 0.3, 0.4, 0.5],
                      [0.5, 0.4, 0.3, 0.2, 0.1]],
    })
    with pa.OSFile(temp_file, 'wb') as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return temp_file

# We also need a dummy base iterator that wraps ArrowFileIterator so that
# its create_iter() yields DummyBltExample objects.
class DummyArrowFileIteratorWrapper:
    def __init__(self, arrow_file: str):
        # Provide dataset_files so that the ArrowFileIterator uses them directly.
        self.arrow_iterator = ArrowFileIterator(
            file_path=arrow_file,
            dataset_files=[arrow_file],
            worker_id=0,
            num_workers=1,
            preprocess_dir=os.path.dirname(arrow_file),
            entropy_model_name=None,
            arrow_batch_size=2,
            s3_profile=None,
            file_format="arrow",
        )
    def create_iter(self):
        # The ArrowFileIterator returns BltExample objects.
        # For our dummy, we transform them into DummyBltExample.
        for row in self.arrow_iterator.create_iter():
            # Assume row has attributes sample_id, text, and entropies.
            yield DummyBltExample(
                sample_id=row.sample_id,
                text=row.text,
                tokens=row.tokens,         # likely None
                mask=row.mask,             # likely None
                patch_lengths=row.patch_lengths,  # likely None
                entropies=row.entropies
            )
    def get_state(self):
        return {}  # Simplified for testing.

def test_preprocess_iterator_with_arrow():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a dummy Arrow file.
        arrow_file = create_dummy_arrow_file(temp_dir)
        
        # Instantiate the dummy ArrowFileIterator wrapper.
        dummy_arrow_iterator = DummyArrowFileIteratorWrapper(arrow_file)
        
        # Create the PreprocessIterator using the dummy ArrowFileIterator.
        preprocess_iterator = DummyPreprocessIterator(
            base_iterator=dummy_arrow_iterator,
            tokenizer_args=DummyTokenizerArgs(),
            patcher_args=DummyPatcherArgs(),
            add_tokens=True,
            add_patches=True
        )
        
        # Get processed examples.
        processed_examples = preprocess_iterator.create_iter()
        
        # We expect two examples.
        assert len(processed_examples) == 2, f"Expected 2 processed examples, got {len(processed_examples)}"
        
        for example in processed_examples:
            # Expected tokens: ASCII codes for each character.
            expected_tokens = [ord(c) for c in example.text]
            assert example.tokens == expected_tokens, f"Tokens mismatch for text '{example.text}': expected {expected_tokens}, got {example.tokens}"
            # Mask should be list of True.
            assert example.mask == [True] * len(expected_tokens), f"Mask mismatch for text '{example.text}'"
            # Dummy patcher returns [len(expected_tokens)]
            assert example.patch_lengths == [len(expected_tokens)], f"Patch length mismatch for text '{example.text}': expected {[len(expected_tokens)]}, got {example.patch_lengths}"
            # Entropies should be unchanged.
            if example.text == "Hello":
                assert example.entropies == [0.1, 0.2, 0.3, 0.4, 0.5], f"Entropies mismatch for 'Hello'"
            elif example.text == "World":
                assert example.entropies == [0.5, 0.4, 0.3, 0.2, 0.1], f"Entropies mismatch for 'World'"
        
        print("PreprocessIterator with ArrowFileIterator test passed!")

if __name__ == "__main__":
    test_preprocess_iterator_with_arrow()
