# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Any, Generator, Optional

import torch

from bytelatent.data.data_types import BltExample
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator
from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum
from bytelatent.blt_tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.blt_tokenizers.build_tokenizer import TokenizerArgs


class PreprocessIterator:
    """
    Takes BltExamples with fields filled in only from input iterators (like ArrowFileIterator),
    and fills in fields that require preprocessing like tokenization and patching.
    """

    def __init__(
        self,
        base_iterator,
        *,
        tokenizer_args: Optional[TokenizerArgs] = None,
        patcher_args: Optional[PatcherArgs] = None,
        add_tokens: bool = True,
        add_patches: bool = True,
    ):
        """
        Initialize a preprocessing iterator.

        Args:
            base_iterator: The base iterator providing raw examples (like ArrowFileIterator)
            tokenizer_args: Arguments for building the tokenizer
            patcher_args: Arguments for building the patcher
            add_tokens: Whether to add tokenization to examples
            add_patches: Whether to add patches to examples
        """
        self.base_iterator = base_iterator
        self.tokenizer_args = tokenizer_args
        self.patcher_args = patcher_args
        self.add_tokens = add_tokens
        self.add_patches = add_patches
        self.tokenizer = None ## we will import here
        self.patcher = None ## likewise, we import it here

    def get_state(self):
        """
        Get the current state of the iterator for checkpointing.
        """
        return {
            "base_iterator_state": self.base_iterator.get_state(),
            "tokenizer_args": self.tokenizer_args,
            "patcher_args": self.patcher_args,
            "add_tokens": self.add_tokens,
            "add_patches": self.add_patches,
        }

    @classmethod
    def from_state(cls, state, base_iterator_class=None):
        """
        Reconstruct an iterator from a checkpoint state.
        
        Args:
            state: The saved state dictionary
            base_iterator_class: The class of the base iterator (e.g., ArrowFileIterator)
                                 If None, assumes the base_iterator has a from_state method
        """
        if base_iterator_class:
            base_iterator = base_iterator_class.from_state(state["base_iterator_state"])
        else:

            base_iterator = state["base_iterator_state"]["build"]()
            
        return cls(
            base_iterator,
            tokenizer_args=state["tokenizer_args"],
            patcher_args=state["patcher_args"],
            add_tokens=state["add_tokens"],
            add_patches=state["add_patches"],
        )

    def _init_preprocessing_tools(self):
        """Initialize tokenizer and patcher as needed."""
        if self.tokenizer is None and self.add_tokens and self.tokenizer_args:
            self.tokenizer = self.tokenizer_args.build()
            
        if self.patcher is None and self.add_patches and self.patcher_args:
            self.patcher = self.patcher_args.build()

    def create_iter(self) -> Generator[BltExample, Any, None]:
        """Create an iterator that processes examples from the base iterator."""
        self._init_preprocessing_tools()
        
        for example in self.base_iterator.create_iter():
            tokens = self._process_tokens(example)
            
            patch_lengths, entropies = self._process_patches(example, tokens)
            
            # Yield the processed example
            yield BltExample(
                sample_id=example.sample_id,
                text=example.text,
                tokens=tokens,
                mask=[True] * len(tokens) if tokens else None,
                patch_lengths=patch_lengths,
                entropies=entropies,
            )

    def _process_tokens(self, example):
        """Process tokens for an example."""
        if self.add_tokens and self.tokenizer:
            return self.tokenizer.encode(example.text)
        return example.tokens

    def _process_patches(self, example, tokens):
        """Process patches for an example."""
        entropies = example.entropies
        
        # Skip patching if not needed
        if not self.add_patches or self.patcher is None:
            return None, entropies
            
        # Process entropy-based patching
        if self.patcher.patching_mode == PatchingModeEnum.entropy:
            assert entropies is not None, "For entropy-based patching, entropies cannot be None"
            entropies_tensor = torch.tensor(entropies).unsqueeze(0)
        else:
            entropies_tensor = None
            
        # Calculate patch lengths
        if tokens:
            tokens_tensor = torch.tensor(tokens).unsqueeze(0)
            patch_lengths = self.patcher.patch(
                tokens_tensor,
                include_next_token=False,
                entropies=entropies_tensor,
            )[0][0].tolist()
            return patch_lengths, entropies
        
        return None, entropies

    def __iter__(self):
        """Make the class directly iterable."""
        return self.create_iter()