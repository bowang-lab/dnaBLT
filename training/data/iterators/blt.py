# Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum, auto
from typing import Any, Optional

import torch
from pydantic import ConfigDict, model_validator
from torch import nn

from typing_extensions import Self


from pydantic import BaseModel

SEP = " "
BOS_ID: int = 1
EOS_ID: int = 2
PAD_ID: int = 255
BOE_ID: int = 0
BPE_ID: int = 3
OFFSET: int = 4

BYTE_UNITS: int = 256


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    ffn_dim_multiplier: float | None = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED.value

    max_seqlen: int = 1024

    attn_impl: str | None = "xformers"
    attn_bias_type: str | None = None
    # Special token config
    eos_id: int | None = EOS_ID

class ByteLatentTransformerArgs(BaseTransformerArgs):
    # Basic model configuration
    seed: int = 42
    vocab_size: int = 256
    dim: int = 512
    n_layers: int = 9
    n_heads: int = 8
    weight_tying: bool = False
    architecture: str = "vanilla" # For Mamba use "mamba"
    patch_in_forward: bool = False
    max_seqlen: int = 4096

    # Architecture and dimensions
    dim_token: int | None = None
    dim_global: int = 512
    dim_local_decoder: int = 256
    dim_local_encoder: int = 256
    n_layers_global: int = 9
    n_layers_local_decoder: int = 5
    n_layers_local_encoder: int = 1

    # Tokenization and patching
    patch_size: float | None = None
    patching_mode: str | None = "entropy"
    patching_threshold: float | None = None
    patching_threshold_add: float | None = None
    monotonicity: bool = False
    patching_batch_size: int = 1
    patching_device: str = "cuda"
    max_patch_length: int | None = None

    # Encoder/Decoder configuration
    tie_local_encoder_decoder_logits: bool = False
    use_local_encoder_transformer: bool = True
    encoder_lm_loss: bool = False
    max_encoder_seq_length: int | None = 8192
    pad_to_max_length: bool = False
    encoder_enable_byte_ngrams: bool = False
    encoder_enable_byte_group_hash: bool = True
    ngram_vocab_sizes: int | None = 1

    # Cross attention configurations
    cross_attn_encoder: bool = True
    cross_attn_decoder: bool = True
    cross_attn_window_encoder: int | None = 512
    cross_attn_window_decoder: int | None = 512
    cross_attn_k: int | None = 2
    cross_attn_nheads: int | None = 4
    cross_attn_all_layers_decoder: bool = True
    cross_attn_all_layers_encoder: bool = True
    cross_attn_use_flex_attention: bool = True
    cross_attn_init_by_pooling: bool = True

    # Encoder hash configurations
    encoder_hash_byte_group_size: Any | None = [3, 4, 5, 6, 7, 8]
    encoder_hash_byte_group_vocab: int = 100000
    encoder_hash_byte_group_nb_functions: int = 1

    # Model behavior and optimization
    log_patch_lengths: bool = False
    non_linearity: str = "swiglu"
    use_rope: bool = True
    recompute_fc1_out: bool = False
    recompute_fc3_out: bool = False
    recompute_attn: bool = True
    custom_bwd: bool = False
    layer_ckpt: str = "all"

    # Initialization and attention
    init_use_gaussian: bool = True
    init_use_depth: str = "current"
    attn_bias_type: str = "causal"
    alpha_depth: str = "disabled"
    max_length: int = 8192

    # Norm configuration
    norm_eps: float = 1e-5
    norm_affine: bool = True
    pre_norm: bool = True
    norm_type: str = "rmsnorm"

    # Additional configurations
    multiple_of: int = 256
    ffn_dim_multiplier: float = 1 # actually corresponds to ffn_dim=3
    dropout: float = 0
    output_size: int = -1

    # Additional parameters from ModelArgs
    share_encoder_decoder_emb: bool = False
    global_local_decoder_residual_layer: str | None = None

    tokenize_with_bpe_delimiter: bool = False
    patching_thresholds_str: str | None = None
    tie_local_encoder_decoder: bool = False
    encoder_preds_low_entropy_toks: float | None = None
    encoder_preds_random_toks: float | None = None
    dim_token_emb: int | None = None
    dim_patch_emb: int | None = None

    encoder_ngram_table_dir: str | None = None
    encoder_ngram_to_size_str: str | None = "3:16666,4:16666,5:16666,6:16666,7:16666,8:16666"
    
    # Model architecture params
    entropy_model_checkpoint_dir: str | None = None
    entropy_model_is_ngram_model: bool = False
    downsampling_by_pooling: str | None = "max"
    n_heads_global: int = 8
    n_heads_local_decoder: int = 4
    n_heads_local_encoder: int = 4
    n_kv_heads: int | None = None
    n_kv_heads_global: int | None = None
    conv_kernel_size: int | None = None
    local_attention_window_len: int | None = 512

    # Performance optimization
    sequence_parallel: bool = False
    loss_parallel: bool = False
    fuse_sequence_parallel: bool = False
    use_fsdp: bool = False
    attn_to_keep: str = "all"

    # Parameter mixing
    pm_size: int = 0

    # Logging
    full_logging_n_layers: int = 4

    @model_validator(mode="after")
    def check_hash_byte_sizes(self) -> Self:
        if (
            self.encoder_hash_byte_group_size is not None
            and type(self.encoder_hash_byte_group_size) == str
        ):
            self.encoder_hash_byte_group_size = [
                int(x)
                for x in self.encoder_hash_byte_group_size.split(",")
                if len(x) > 0
            ]
        return self

