import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import random
from dataclasses import dataclass, field
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator

###############################################
# this is our top down approach, the following methods of course need to be updated, but this is a good first step
###############################################

class SimpleTokenizer: # DONE
    def __init__(self):
        self.n_words = 4 # ACTG
        self.unk_token = 78  # unknown token id = "N"
        # Hyperefficient implementation of StripedHyena Char Tokenizer

    def encode(self, text: str) -> list:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list) -> str:
        return bytes(tokens).decode("utf-8")


@dataclass
class TokenizerArgs: 
    def build(self) -> SimpleTokenizer:
        return SimpleTokenizer()


# --- TODO: need to implement this correctly ofc ---
class SimplePatcher:
    """
    A simple patcher that splits a token sequence into fixed-size patches.
    For each example, we split the sequence into patches of size `patch_size`
    (except possibly the last patch, which is shorter).
    """
    def __init__(self, patch_size: int = 10):
        self.patch_size = patch_size

    def patch(self, tokens_tensor: torch.Tensor, include_next_token: bool = False) -> torch.Tensor:
        # tokens_tensor shape: [batch, seq_len]
        batch, seq_len = tokens_tensor.shape
        seq_len_next = seq_len + 1 if include_next_token else seq_len
        # Calculate number of patches (ceil division)
        num_patches = (seq_len_next + self.patch_size - 1) // self.patch_size
        patch_lengths = []
        for _ in range(batch):
            # Create a list of patch lengths
            lengths = [self.patch_size] * num_patches
            total = sum(lengths)
            if total > seq_len_next:
                # Adjust the last patch length
                lengths[-1] = seq_len_next - self.patch_size * (num_patches - 1)
            patch_lengths.append(lengths)
        # Pad patch lengths so every example has the same number of patches
        max_len = max(len(plist) for plist in patch_lengths)
        padded = [plist + [0] * (max_len - len(plist)) for plist in patch_lengths]
        return torch.tensor(padded, dtype=torch.long)


@dataclass
class PatcherArgs:
    patch_size: int = 10

    def build(self) -> SimplePatcher:
        return SimplePatcher(patch_size=self.patch_size)


# TODO: implement this
@dataclass
class BltExample:
    sample_id: int
    text: str
    tokens: list
    mask: list
    patch_lengths: list = None
    entropies: list = None

# this will be our preproces iterator, will do the arrow file iteration + the patching
class SimplePreprocessIterator:
    """
    A simple iterator that yields BltExample objects.
    It takes a list of raw texts, tokenizes them using the provided tokenizer,
    pads/truncates to a fixed sequence length, and applies patching.
    """
    def __init__(self, arrow_iterator: ArrowFileIterator, seq_len: int, tokenizer_args: TokenizerArgs = None, patcher_args: PatcherArgs = None):
        self.arrow_iterator = arrow_iterator
        self.seq_len = seq_len
        # Build tokenizer; use default if not provided.
        if tokenizer_args is None:
            tokenizer_args = TokenizerArgs()
        self.tokenizer = tokenizer_args.build()
        # Build patcher; use default if not provided.
        if patcher_args is None:
            patcher_args = PatcherArgs()
        self.patcher = patcher_args.build()

    def create_iter(self):
        example_iter = self.arrow_iterator.create_iter()
        for example in example_iter:
            text = example.text
            tokens = self.tokenizer.encode(text) # or just example.tokens
            entropies = torch.tensor(example.entropies).unsqueeze(0)
            patch_lengths = self.patcher.patch(torch.tensor(tokens).unsqueeze(0), entropies=entropies, include_next_token=False)[0][0].tolist()
            yield BltExample(
                sample_id=example.sample_id,
                text=text,
                tokens=tokens,
                mask=[True] * len(tokens),
                patch_lengths=patch_lengths,
                entropies=example.entropies
            )


# this is the iterator, also needs to be implemented
class LightningIterableDataset(IterableDataset):
    def __init__(self, stateful_iterator):
        """
        Wraps an iterator (which implements create_iter) into an IterableDataset.
        """
        self.stateful_iterator = stateful_iterator

    def __iter__(self):
        return self.stateful_iterator.create_iter()


from bytelatent.model.blt import ByteLatentTransformer ## this is the LMtransformer that they used in the train.py


def compute_loss(pred, target, mask, scale=1.0):
    """
    A simple cross-entropy loss function.
    For language modeling, target is typically the input tokens shifted by one.
    """
    batch, seq_len, vocab_size = pred.shape
    pred = pred.reshape(-1, vocab_size)
    target = target.reshape(-1)
    loss = F.cross_entropy(pred, target)
    # Here, we return the same loss as both overall loss and token-level loss.
    return loss * scale, loss * scale

###############################################
# TrainArgs and related configuration
###############################################

@dataclass
class DataloaderArgs:
    batch_size: int = 2  # Note: our iterator yields one example at a time.
    seq_len: int = 50
    # For demonstration, we use a fixed list of texts.
    sources: list = field(default_factory=lambda: [
        "This is the first sample sentence for training.",
        "Another example text is here.",
        "Yet another piece of text to simulate training data.",
    ])
    # Allow attaching tokenizer and patcher arguments.
    tokenizer_args: TokenizerArgs = TokenizerArgs()
    patcher_args: PatcherArgs = PatcherArgs(patch_size=10)

    def build_iterator(self):
        return SimplePreprocessIterator(self.sources, self.seq_len, self.tokenizer_args, self.patcher_args)


@dataclass
class OptimArgs:
    lr: float = 1e-3
    clip: float = 1.0


@dataclass
class CheckpointArgs:
    init_ckpt_path: str = None
    s3_profile: str = None


@dataclass
class DistributedArgs:
    pass


@dataclass
class EnvironmentArgs:
    pass


@dataclass
class ProfilerArgs:
    pass


@dataclass
class LoggingArgs:
    pass


@dataclass
class TrainArgs:
    seed: int = 42
    steps: int = 1000
    grad_acc_steps: int = 1
    train_entropy_model: bool = False
    data: DataloaderArgs = DataloaderArgs()
    optim: OptimArgs = OptimArgs()
    # For simplicity, we define model parameters in a dict.
    model: dict = field(default_factory=lambda: {"d_model": 32, "n_layers": 2, "vocab_size": 100})
    checkpoint: CheckpointArgs = CheckpointArgs()
    distributed: DistributedArgs = DistributedArgs()
    env: EnvironmentArgs = EnvironmentArgs()
    profiling: ProfilerArgs = ProfilerArgs()
    logging: LoggingArgs = LoggingArgs()

###############################################
# PyTorch Lightning Module
###############################################

class MyLightningModule(pl.LightningModule):
    def __init__(self, args: TrainArgs):
        super().__init__()
        self.args = args
        
        self.tokenizer = args.data.tokenizer_args.build()

        if self.tokenizer.n_words < args.model["vocab_size"]:
            raise ValueError("Tokenizer vocab size is smaller than model vocab size.")
        torch.manual_seed(args.seed)
        

        self.model = ByteLatentTransformer(args.model)
        

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.optim.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.95)
        

        iterator = args.data.build_iterator()
        self.train_dataset = LightningIterableDataset(iterator)

    def forward(self, x, patch_lengths=None, ngram_ids=None):
        return self.model(x, patch_lengths=patch_lengths, ngram_ids=ngram_ids)
        
    def training_step(self, batch, batch_idx):
        """
        In our simple example, batch is a BltExample.
        We convert its fields into tensors.
        """

        batch_x = torch.tensor(batch.tokens, dtype=torch.long)

        batch_y = torch.tensor(batch.tokens, dtype=torch.long)
        mask = torch.tensor(batch.mask) if batch.mask is not None else None
        patch_lengths = (torch.tensor(batch.patch_lengths, dtype=torch.long)
                         if batch.patch_lengths is not None else None)

        pred = self.forward(batch_x, patch_lengths=patch_lengths)
        loss, _ = compute_loss(pred, batch_y, mask, scale=1.0)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step", 
            },
        }
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None)  # Each item is a single example



def main(args: TrainArgs):
    model = MyLightningModule(args)
    trainer = pl.Trainer(
        max_steps=args.steps,
        accumulate_grad_batches=args.grad_acc_steps,
        gpus=1 if torch.cuda.is_available() else 0,
    )
    trainer.fit(model)

if __name__ == "__main__":

    args = TrainArgs() ## this is very key, ensure that trainargs is done properly

    main(args)
