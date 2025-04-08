from tokenizer import Tokenizer

SEP = " "
BOS_ID: int = 1
EOS_ID: int = 2
PAD_ID: int = 255
BOE_ID: int = 0
BPE_ID: int = 3
OFFSET: int = 4

BYTE_UNITS: int = 256

class BltTokenizer(Tokenizer):
    def __init__(
        self,
        *,
        vocab_size_unit_1: int = BYTE_UNITS,
        bpe_delim: bool = False,
        bpe_tokenizer_path="/home/artidoro/tokenizers/llama_v2.tokenizer.model",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.bpe_id = BPE_ID
        self.bpe_tokenizer_path = bpe_tokenizer_path
        if bpe_delim:
            self.bpe_tokenizer = SentencePieceTokenizer(
                model_path=self.bpe_tokenizer_path
            )
        else:
            self.bpe_tokenizer = None
        self.bpe_delim = bpe_delim
        self.offsetting_special_char = OFFSET
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.n_words = vocab_size_unit_1 + self.offsetting_special_char

    def get_vocab_size(self) -> int:
        return self.n_words

    def encode(
        self, text: str, add_bos: bool | None = None, add_eos: bool | None = None
    ):
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        if self.bpe_delim:
            tokens = text2bytes_bpe_delims(
                text,
                bpe_tokenizer=self.bpe_tokenizer,
                bpe_id=self.bpe_id,
                offsetting_special_char=self.offsetting_special_char,
                add_bos=False,
                add_eos=False,
            )
        else:
            tokens = bytes(text, encoding="utf-8", errors="ignore")

        # Offsetting
        tokens = [int(unit) + self.offsetting_special_char for unit in tokens]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: list[int], cut_at_eos: bool = False):
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        return bytes(
            [
                tok - self.offsetting_special_char
                for tok in tokens
                if tok - self.offsetting_special_char >= 0
            ]
        ).decode("utf-8", errors="ignore")

    def get_token_offsets(self, text: str, tokens: list[int] | None = None):
        # TODO: Figure out what this does
        raise NotImplementedError()