# Copyright (c) Meta Platforms, Inc. and affiliates.
import pandas as pd
from pydantic import BaseModel

from bytelatent.constants import BLT_DATA
from bytelatent.data.data_types import BltExample
from bytelatent.data.iterators.abstract_iterator import IteratorState, StatefulIterator
from bytelatent.data.iterators.preprocess_iterator import PreprocessIterator
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs


class BltTestIteratorState(BaseModel, IteratorState):
    position: int
    total: int

    def build(self):
        blt_iter = BltTestIteratorState(total=self.total)
        blt_iter.position = self.position
        return blt_iter


class BltTestIterator(StatefulIterator):
    def __init__(self, total: int):
        self.position = 0
        self.total = total

    def get_state(self):
        return BltTestIteratorState(position=self.position, total=self.total)

    def create_iter(self):
        for i in range(self.total):
            self.position += 1
            yield BltExample(
                sample_id=f"test_{i}",
                text=f"This is some test {i} text.",
                tokens=None,
                mask=None,
                entropies=None,
                patch_lengths=None,
            )


class BltTestWithEntropiesIteratorState(BaseModel, IteratorState):
    position: int
    total: int

    def build(self):
        blt_iter = BltTestWithEntropiesIteratorState(total=self.total)
        blt_iter.position = self.position
        return blt_iter


class BltTestWithEntropiesIterator(StatefulIterator):
    def __init__(self, total: int):
        self.position = 0
        self.total = total

    def get_state(self):
        return BltTestIteratorState(position=self.position, total=self.total)

    def create_iter(self):
        text = "Daenerys Targaryen is in Game of Thrones, a fantasy epic by George R.R. Martin."
        df = pd.read_json("fixtures/tokens_with_entropies.json")
        tokens = df["token_ids"].tolist()
        entropies = df["entropies"].tolist()
        # BOS and EOS
        assert len(tokens) == len(text) + 2
        for i in range(self.total):
            self.position += 1
            yield BltExample(
                sample_id=f"test_{i}",
                text=text,
                tokens=tokens,
                mask=[True] * len(tokens),
                entropies=entropies,
                patch_lengths=None,
            )


def test_preprocess_iter():
    total = 3
    tokenizer_args = TokenizerArgs(
        name="blt",
        init_kwargs={
            "bpe_tokenizer_path": BLT_DATA / "tokenizer_final_32k.minus_inf_ws.model"
        },
    )
    for mode in [
        PatchingModeEnum.bpe,
        PatchingModeEnum.space,
    ]:
        data_it = BltTestIterator(total)
        patcher_args = PatcherArgs(patching_mode=mode)
        example_it = PreprocessIterator(
            data_it, tokenizer_args=tokenizer_args, patcher_args=patcher_args
        )
        count = 0
        for example in example_it.create_iter():
            assert isinstance(example.tokens, list)
            assert isinstance(example.tokens[0], int)
            # BOS and EOS
            assert len(example.tokens) == len(example.text) + 2
            assert example.mask is not None
            assert len(example.tokens) == len(example.mask)
            count += 1

        assert count == total


def test_non_entropy_patch_iter():
    total = 3
    tokenizer_args = TokenizerArgs(
        name="blt",
        init_kwargs={
            "bpe_tokenizer_path": BLT_DATA / "tokenizer_final_32k.minus_inf_ws.model"
        },
    )
    for mode in [
        PatchingModeEnum.bpe,
        PatchingModeEnum.space,
    ]:
        patcher_args = PatcherArgs(patching_mode=mode)
        data_it = BltTestIterator(total)
        example_it = PreprocessIterator(
            data_it, tokenizer_args=tokenizer_args, patcher_args=patcher_args
        )

        count = 0
        for example in example_it.create_iter():
            assert isinstance(example.patch_lengths, list)
            assert isinstance(example.patch_lengths[0], int)
            assert len(example.tokens) == sum(example.patch_lengths)
            count += 1

        assert count == total


def test_entropy_patch_iter():
    total = 2
    patcher_args = PatcherArgs(
        patching_mode=PatchingModeEnum.entropy, threshold=1.335442066192627
    )
    tokenizer_args = TokenizerArgs(
        name="blt",
        init_kwargs={
            "bpe_tokenizer_path": BLT_DATA / "tokenizer_final_32k.minus_inf_ws.model"
        },
    )
    data_it = BltTestWithEntropiesIterator(total)
    example_it = PreprocessIterator(
        data_it, tokenizer_args=tokenizer_args, patcher_args=patcher_args
    )

    count = 0
    for example in example_it.create_iter():
        assert isinstance(example.patch_lengths, list)
        assert isinstance(example.patch_lengths[0], int)
        assert len(example.tokens) == sum(example.patch_lengths)
        count += 1

    assert count == total
