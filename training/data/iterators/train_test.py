import os
import shutil
import tempfile
import random
import numpy as np
import torch
import pytorch_lightning as pl

# Import your train routine and TrainArgs.
# (Adjust the import paths as necessary.)
from args import TrainArgs
from lightning_train import train  # assuming your training loop is in train.py

# In many cases, your TrainArgs, DataloaderArgs, OptimArgs, ByteLatentTransformerArgs, etc.
# already have default values, so for testing you may override a few parameters.

def test_training_loop():
    # Create a temporary directory to use as dump_dir
    tmp_dir = tempfile.mkdtemp(prefix="train_loop_test_")
    try:
        # Instantiate TrainArgs with test-friendly configuration.
        # You want the training to run quickly (e.g., a small number of steps) and produce outputs.
        # Here we override:
        #   - dump_dir to be our temporary directory,
        #   - steps to a low number so training runs quickly,
        #   - data.sources with a dummy value so that at least one iterator is created.
        train_args = TrainArgs(
            name="dummy_training_test",
            dump_dir=tmp_dir,
            seed=42,
            steps=5,   # run for just 5 steps
            entropy_model=None,
            data={
                "s3_profile": None,
                "root_dir": tmp_dir,
                "sources": {"dummy_source": 1.0},
                "batch_size": 2,
                "seq_len": 10,
                "seed": 42,
                "add_bos": True,
                "add_eos": True,
                "load_async": False,
                "prefetch_size": 2,
                "preprocess_dir": None,
                "dataset_files": None,  # in testing, you might set up your dummy iterator to not actually read files
                "entropy_model_name": None,
                "arrow_batch_size": 2,
                "buffer_size": 2,
                "file_format": "arrow",
                "pad_to_max_length": True,
                "max_encoder_seq_length": 20,
                "enable_byte_ngrams": False,
                "add_patches": False,  # disable patching for simplicity in a dummy test,
                "tokenizer_args": {
                    "name": "blt",  # Use the mock tokenizer for testing
                    "init_kwargs": {}
                }, # Use defaults (assuming your TokenizerArgs has defaults)
                "patcher_args": {},    # Likewise, defaults for PatcherArgs
            },
            optim={},  # Use default optimizer args
            model={},  # Use default model args; ensure ByteLatentTransformerArgs defaults exist
            checkpoint={
                "dump": {"every": 1, "keep": 1},
                "eval": {"every": 1, "keep": 1},
                "path": None,
                "init_ckpt_path": None,
                "continue_training_from_init": False,
                "s3_profile": None,
            },
            async_eval_gpus=None,
            eval=None,
            eval_on_gpus=None,
            debug_dynamo=False,
            grad_acc_steps=1,
            gc_collect_freq=1000,
            probe_freq=None,
            max_steps=None,
        )
        
        # Optionally, if your TrainArgs expects objects for nested arguments (rather than dictionaries),
        # create them accordingly. The above uses dicts as a shortcut if your Pydantic model supports it.
        
        # Run the training loop.
        # This should instantiate the data pipeline (calling build_from_rank and, via the DataModule,
        # eventually creating iterators via their __iter__() / create_iter() methods).
        print("Starting training loop test...")
        train(train_args, test_mode=True)
        
        # After training, we can check that the checkpoint directory exists and contains files.
        ckpt_dir = os.path.join(train_args.checkpoint.path or os.path.join(tmp_dir, "checkpoints"))
        assert os.path.isdir(ckpt_dir), f"Checkpoint directory not found at {ckpt_dir}"
        ckpt_files = os.listdir(ckpt_dir)
        # With 5 steps and a checkpoint every step, we expect at least one checkpoint file.
        assert len(ckpt_files) > 0, "No checkpoint file was created during training."
        
        print("Training loop test passed!")
    finally:
        # Clean up the temporary directory.
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_training_loop()
