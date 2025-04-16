# Copyright (c) Meta Platforms, Inc. and affiliates.
import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask
from xformers.ops import AttentionBias

from bytelatent.base_transformer import (
    BaseTransformer,
    BaseTransformerArgs,
    flex_attention_comp,
    repeat_kv,
)
from bytelatent.model.utils import create_causal_mask

logger = logging.getLogger()
try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm

    RMSNorm = FusedRMSNorm
except (ImportError, ModuleNotFoundError):
    logging.debug("Apex not found. Using nn.RMSNorm")
    RMSNorm = nn.RMSNorm

import math

from vortex.model.cache import (
    InferenceParams,
    HyenaCascadeFIRInferenceParams,
    HyenaCascadeIIRInferenceParams,
)
from vortex.model.engine import HyenaInferenceEngine
from vortex.model.layers import (
    ParallelGatedMLP,
    RMSNorm,
    VocabParallelEmbedding,
    VocabParallelUnembedding,
    TELinear,
)
from vortex.model.utils import (
    Lambda,
    column_split,
    interleave,
    print_rank_0,
    move_to_device,
    fixup_fp8_extra_states,
    fixup_te_workspace,
)
from vortex.logging import activations_logger, enable_activations_logging

import logging
from tqdm import tqdm

from bytelatent.base_transformer import Attention, RotaryEmbedding

try:
    from vortex.model.positional_embeddings import swap_mha_rope
except ImportError:
    "could not import swap_mha_rope from src.positional_embeddings"


class GlobalStripedHyena2(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dropout = args.dropout
        self.eos_id = args.eos_id
        self.dim_token_emb = args.dim_token_emb

        self.token_embedding_projection = None
        if args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )

        self.model = StripedHyena(args)

    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        Similar to BaseTransformer.forward, but with an additional embeds argument
        and projection to the token space.
        """
        bs, seqlen = tokens.shape

        h = embeds

        mask = (
            mask
            if mask is not None
            else create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                tokens=tokens,
                eos_id=self.eos_id,
            )
        )

        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.model(h, tok_idx=tok_idx, mask=mask, attn_impl=self.attn_impl)
        return h, cache

    def init_weights(self):
        super().init_weights()
        std = self.dim_token_emb ** (-0.5)
        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )


# Copyright (c) 2024, Michael Poli.


class AttentionBlock(nn.Module):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.pre_norm, self.post_norm = RMSNorm(config), RMSNorm(config)
        self.layer_idx = layer_idx
        self.print_activations = config.get("print_activations", False)
        self.proj_groups = config.get("proj_groups", 1)
        dtype = config.get("attn_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.hidden_size_per_attention_head = (
            config.hidden_size // config.num_attention_heads
        )

        self.counter = 0
        self.inner_mha_cls = Attention(
            dim=config.hidden_size,
            head_dim=config.hidden_size // config.num_attention_heads,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_attention_heads,
            rope_theta=config.get("rotary_emb_base", 1000000),
        ).to(dtype=dtype)

        # check if using interpolated rotary pos emb from config, and swap the rope emb
        if config.get("use_interpolated_rotary_pos_emb", False):
            swap_mha_rope(
                mha=self.inner_mha_cls,
                kwargs_new_rope={
                    "scaling_factor": config.get("rotary_emb_scaling_factor", 1.0)
                },
            )

        if self.config.get("smeared_gqa", False):
            self.inner_mha_cls.num_heads_kv = self.inner_mha_cls.num_heads
        self.inner_mha_cls.rotary_emb.register_buffer(
            "inv_freq", self.inner_mha_cls.rotary_emb.inv_freq
        )

        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

    def forward(self, u, freq_cis, tok_idx, mask, attn_impl, padding_mask):
        if isinstance(
            padding_mask, torch.Tensor
        ):  # workaround for masking bug in FA. This works because Wqkv does not have bias
            # and attention scores will be also automatically zeroed.
            u = u * padding_mask[..., None]

        if self.print_activations:
            activations_logger.info(f"pre mha: {u}")

        u = (
            self.inner_mha_cls(
                self.pre_norm(u),
                freq_cis,
                tok_idx,
                mask,
                attn_impl,
            )
            + u
        )

        if isinstance(padding_mask, torch.Tensor):  # guard against bias
            u = u * padding_mask[..., None]

        u = self.mlp(self.post_norm(u)) + u
        return u, None


class HyenaCascade(nn.Module):
    def __init__(
        self, config, layer_idx, hyena_filter_groups=None, fir_inner_filter_length=None
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hyena_filter_groups = hyena_filter_groups
        self.print_activations = config.get("print_activations", False)
        self.ground_truth_activations_path = config.get(
            "ground_truth_activations_path", None
        )

        self.use_flashfft = config.get("use_flashfft", False)
        self.state_size = config.state_size
        self.hidden_size = config.hidden_size
        self.num_filters = config.num_filters
        self.inference_mode = config.get("inference_mode", True)
        self.counter = 0
        self.column_split_hyena = config.get("column_split_hyena", True)
        self.hyena_flip_x1x2 = config.get("hyena_flip_x1x2", False)

        assert (
            self.hidden_size % self.num_filters == 0
            and self.num_filters <= self.hidden_size
        )

        # attention heads are not used except to split post short_filter
        # projections in the same way as the checkpoint
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = (
            self.hidden_size // self.num_attention_heads
        )

        self.fir_inner_filter_length = fir_inner_filter_length
        self.short_filter_length = config.short_filter_length
        self.short_filter_weight = nn.Parameter(
            torch.randn(3 * config.hidden_size, 1, config.short_filter_length)
        )
        self.short_filter_bias = (
            nn.Parameter(torch.randn(3 * config.hidden_size))
            if config.short_filter_bias
            else None
        )

        self.engine = HyenaInferenceEngine(
            layer_idx=layer_idx,
            ground_truth_activations_path=self.ground_truth_activations_path,
            print_activations=self.print_activations,
            hyena_flip_x1x2=config.get("hyena_flip_x1x2", False),
        )
        self.use_flash_depthwise = config.get("use_flash_depthwise", False)
        self.data_dtype = None

        if self.use_flash_depthwise:
            try:
                from flashfftconv import FlashDepthwiseConv1d

                self.fir_fn = FlashDepthwiseConv1d(
                    channels=3 * self.hidden_size,
                    kernel_size=self.short_filter_length,
                    padding=self.short_filter_length - 1,
                    weights=self.short_filter_weight,
                    bias=self.short_filter_bias,
                    device=None,
                    dtype=self.config.get("depthwise_dtype", torch.bfloat16),
                )
            except ImportError:
                "flashfftconv not installed"
        else:
            self.fir_fn = F.conv1d

            self.fir_inner_fn = F.conv1d

        self.fftconv_fn = None
        self.long_fir_threshold = config.get("long_fir_threshold", None)
        if self.long_fir_threshold is not None:
            assert self.use_flashfft is False, (
                "long_fir_threshold not compatible with fused flashfft"
            )

        self.num_systems = self.hyena_filter_groups
        self.channels_per_group = self.hidden_size // self.hyena_filter_groups

        if self.fir_inner_filter_length:
            self.h = nn.Parameter(
                torch.randn(self.hyena_filter_groups, 1, fir_inner_filter_length)
            )

            if fir_inner_filter_length >= 128:
                self.D = nn.Parameter(torch.zeros(self.hidden_size))

            if fir_inner_filter_length < 128:
                self.D = None

        else:
            log_poles = torch.randn(
                self.num_systems, self.state_size, 1, dtype=torch.float32
            )

            # TODO: bring over init from internals
            # poles[..., 0] = 1e-2 * torch.randn(self.num_systems, self.state_size, 1)
            # poles[..., 1] = 1e-3 * torch.randn(self.num_systems, self.state_size, 1)

            self.log_poles = nn.Parameter(log_poles)
            self.residues = nn.Parameter(
                torch.randn(self.num_systems, self.state_size, dtype=torch.float32)
            )
            self.D = nn.Parameter(torch.zeros(self.hidden_size))
            self.h = None
        self.t = None

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        if (
            inference_params is not None
            and self.layer_idx in inference_params.fir_state_dict.keys()
        ):
            return self.sequential_forward(u, inference_params)

        else:
            return self.parallel_forward(u, inference_params, padding_mask)

    def parallel_forward(self, u, inference_params=None, padding_mask=None):
        L = u.shape[1]
        dims = (
            self.hidden_size,
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.state_size,
            self.hyena_filter_groups,
        )
        if self.print_activations:
            activations_logger.info(f"pre 1 parallel fir: {u}, {u.min()}, {u.max()}")

        z_pre, fir_state = self.engine.parallel_fir(
            self.fir_fn,
            u,
            self.short_filter_weight,
            self.short_filter_bias,
            L,
            dims=dims,
            gate=False,
            column_split_hyena=self.column_split_hyena,
            fir_length=self.short_filter_length,
            inference_params=inference_params,
            padding_mask=padding_mask,
            dim_last=True,
        )

        if inference_params:
            inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.config.interleave:
            z_pre = interleave(z_pre)

        if self.h is None:
            h, _, _, _ = self.compute_filter(L, u.device)
        else:
            h = self.h

        D = self.D

        if self.hyena_filter_groups > 1:
            h = h.repeat_interleave(self.hidden_size // self.hyena_filter_groups, 0)

        # if inference_params is not None, we plan to perform generation:
        # prefilling is handled by the engine.
        if self.fir_inner_filter_length is not None:
            if self.print_activations:
                activations_logger.info(
                    f"pre 2 parallel fir: {z_pre}, {z_pre.min()}, {z_pre.max()}, {self.fir_inner_filter_length}"
                )
            y, fir_inner_state = self.engine.parallel_fir(
                self.fir_inner_fn,
                z_pre,
                h,
                D,
                L,
                dims=dims,
                gate=True,
                gated_bias=self.fir_inner_filter_length >= 128,
                dim_last=False,
                column_split_hyena=self.column_split_hyena,
                fir_length=self.fir_inner_filter_length,
                inference_params=inference_params,
                padding_mask=padding_mask,
                groups=self.hyena_filter_groups,
            )
            if self.print_activations:
                activations_logger.info(
                    f"post 2 parallel fir: {y}, {y.min()}, {y.max()}"
                )
            y = y.permute(0, 2, 1)
            if inference_params:
                inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            if self.print_activations:
                activations_logger.info(
                    f"pre 2 parallel iir: {z_pre}, {z_pre.min()}, {z_pre.max()}"
                )
            y = self.engine.parallel_iir(
                z_pre,
                h,
                D,
                L,
                t=self.t,
                poles=self.log_poles,
                residues=self.residues,
                dims=dims,
                inference_params=inference_params,
                layer_idx=self.layer_idx,
                prefill_style=self.config.get("prefill_style", "fft"),
                use_flashfft=self.use_flashfft,
                fftconv_fn=self.fftconv_fn,
                column_split_hyena=self.column_split_hyena,
                long_fir_threshold=self.long_fir_threshold,
                padding_mask=padding_mask,
            )
            if self.print_activations:
                activations_logger.info(
                    f"post 2 parallel iir: {y}, {y.min()}, {y.max()}"
                )

        return y, inference_params

    def sequential_forward(self, u, inference_params):
        if self.data_dtype is None:
            self.data_dtype = u.dtype

        if len(u.shape) > 2:
            u = u[:, -1]

        z_pre, fir_state = self.engine.step_fir(
            u,
            inference_params.fir_state_dict[self.layer_idx],
            weight=self.short_filter_weight,
            bias=self.short_filter_bias,
        )
        inference_params.fir_state_dict[self.layer_idx] = fir_state

        if self.config.interleave:
            z_pre = interleave(z_pre)

        x2, x1, v = (
            column_split(
                z_pre, self.num_attention_heads, self.hidden_size_per_attention_head
            )
            if self.column_split_hyena
            else z_pre.split(
                [self.hidden_size, self.hidden_size, self.hidden_size], dim=1
            )
        )

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        if self.fir_inner_filter_length is not None:
            if self.hyena_filter_groups > 1:
                h = self.h.repeat_interleave(
                    self.hidden_size // self.hyena_filter_groups, 0
                )
            else:
                h = self.h

            y, fir_inner_state = self.engine.step_fir(
                x1 * v,
                inference_params.fir_inner_state_dict[self.layer_idx],
                weight=h,
                bias=self.D,
                flip_filter=self.fir_inner_filter_length >= 128,
                gated_bias=self.fir_inner_filter_length >= 128,
            )
            y = y * x2
            inference_params.fir_inner_state_dict[self.layer_idx] = fir_inner_state
        else:
            y, iir_state = self.engine.step_iir(
                x2,
                x1,
                v,
                self.D,
                self.residues,
                self.log_poles,
                inference_params.state_dict[self.layer_idx],
                iir_groups=1,
            )
            inference_params.state_dict[self.layer_idx] = iir_state

        y = y.to(dtype=self.data_dtype)
        return y[:, None], inference_params

    def update_time(self, L, device):
        """
        Set [0, 1, ..., L-1] where L is the length of the current batch of inputs.
        If L is greater than the length of the previous batch, then the time vector is
        reinitialized. Otherwise, the time vector is truncated from cache.
        """
        if self.t is None:
            self.t = torch.arange(L, device=device)[None, None]
        elif self.t.shape[-1] < L:
            self.t = torch.arange(L, device=device)[None, None]
        else:
            self.t = self.t[..., :L]

    def compute_filter(self, L, device):
        self.update_time(L, device)
        filter_dtype = torch.float32
        residues, log_poles = (
            self.residues.to(filter_dtype),
            self.log_poles.to(filter_dtype),
        )
        h = (residues[..., None] * (log_poles * self.t).exp()).sum(1)[None]  # B, D, L
        return h, filter_dtype, log_poles, residues


class ParallelGatedConvBlock(nn.Module):
    def __init__(
        self, config, layer_idx, hyena_filter_groups=None, fir_inner_filter_length=None
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.print_activations = config.get("print_activations", False)
        self.ground_truth_activations_path = config.get(
            "ground_truth_activations_path", None
        )
        self.low_mem_mode = config.get("low_mem_mode", False)
        self.fir_inner_filter_length = fir_inner_filter_length
        self.hyena_filter_groups = (
            hyena_filter_groups
            if hyena_filter_groups is not None
            else config.hidden_size
        )
        dtype = config.get("hyena_block_dtype", torch.bfloat16)
        mlp_dtype = config.get("mlp_dtype", torch.bfloat16)
        self.pre_norm, self.post_norm = (
            RMSNorm(config).to(dtype=dtype),
            RMSNorm(config).to(dtype=dtype),
        )
        self.filter = HyenaCascade(
            config,
            layer_idx,
            hyena_filter_groups=self.hyena_filter_groups,
            fir_inner_filter_length=fir_inner_filter_length,
        ).to(dtype=dtype)

        # For posterity/debugging: TELinear can be easily replaced by
        # nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.qkv_proj_bias).to(dtype=dtype)
        # which sometimes is very useful when debugging FP8.
        self.projections = TELinear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.qkv_proj_bias,
            init_method=torch.nn.init.xavier_uniform_,
            use_fp8=config.get("use_fp8_input_projections", False),
        )

        self.out_filter_dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.hyena_out_proj_bias
        ).to(dtype)
        self.mlp = ParallelGatedMLP(config, layer_idx).to(dtype=mlp_dtype)

        # self.proj_norm_fn = self.proj_norm
        # self.res_mlp_norm_fn = self.res_mlp_norm

        if self.config.get("compile", False):
            self.proj_norm_fn = torch.compile(
                self.proj_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )
            self.res_mlp_norm_fn = torch.compile(
                self.res_mlp_norm, fullgraph=True, dynamic=False, mode="reduce-overhead"
            )

    def pad_to_multiple(self, x, multiple=16):
        """Pad input tensor to multiple of 16 only when FP8 is enabled"""
        if not self.config.get("use_fp8_input_projections", False):
            return x

        batch_size, seq_len, hidden_dim = x.size()
        pad_len = (multiple - (seq_len % multiple)) % multiple
        if pad_len == 0:
            return x
        return F.pad(x, (0, 0, 0, pad_len))

    def proj_norm(self, x):
        if self.print_activations:
            activations_logger.info(
                f"pre mixer norm: {x} {x.min()} {x.max()} {self.projections.__class__}"
            )
            activations_logger.info(
                f"post mixer norm: {self.pre_norm(x)} {self.pre_norm(x).min()} {self.pre_norm(x).max()}"
            )

            if self.ground_truth_activations_path:
                pre_norm_savanna = torch.load(
                    f"{self.ground_truth_activations_path}/pre_mixer_norm_{self.layer_idx}.pt"
                )
                post_norm_savanna = torch.load(
                    f"{self.ground_truth_activations_path}/post_mixer_norm_{self.layer_idx}.pt"
                )

                activation_diff = (x.squeeze() - pre_norm_savanna.squeeze()).abs()
                activations_logger.info(
                    f"pre mixer norm activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                )
                activation_diff = (
                    self.pre_norm(x).squeeze() - post_norm_savanna.squeeze()
                ).abs()
                activations_logger.info(
                    f"post mixer norm activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                )
                activations_logger.info(
                    f"pre norm scale: {self.pre_norm.scale}, {self.pre_norm.scale.min()}, {self.pre_norm.scale.max()}"
                )

        normalized = self.pre_norm(x)
        normalized = self.pad_to_multiple(normalized)
        with torch.cuda.device(x.device):
            projected = self.projections(normalized)

        if isinstance(projected, tuple):
            projected = projected[0]

        original_seq_len = x.size(1)
        # Slice back to original sequence length if padding was added
        if projected.size(1) > original_seq_len:
            projected = projected[:, :original_seq_len, :]

        return projected

    def res_mlp_norm(self, x):
        if self.print_activations:
            activations_logger.info(
                f"pre mlp: {x} {x.min()} {x.max()} {self.mlp.__class__}"
            )
            activations_logger.info(
                f"post mlp norm: {self.post_norm(x)} {self.post_norm(x).min()} {self.post_norm(x).max()}"
            )
            activations_logger.info(
                f"post mlp: {self.mlp(self.post_norm(x))} {self.mlp(self.post_norm(x)).min()} {self.mlp(self.post_norm(x)).max()}"
            )
            if self.ground_truth_activations_path:
                pre_mlp_savanna = torch.load(
                    f"{self.ground_truth_activations_path}/pre_mlp_{self.layer_idx}.pt"
                )
                post_mlp_savanna = torch.load(
                    f"{self.ground_truth_activations_path}/post_mlp_norm_{self.layer_idx}.pt"
                )

                activation_diff = (x.squeeze() - pre_mlp_savanna.squeeze()).abs()
                activations_logger.info(
                    f"pre mlp activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                )
                activation_diff = (
                    self.post_norm(x).squeeze() - post_mlp_savanna.squeeze()
                ).abs()
                activations_logger.info(
                    f"post mlp norm activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                )
        return self.mlp(self.post_norm(x)) + x

    def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
        z = self.proj_norm(u)

        if type(padding_mask) == torch.Tensor:  # guard against bias
            z = z * padding_mask[..., None]

        if self.print_activations:
            activations_logger.info(
                f"pre filter: {z} {z.min()} {z.max()} {self.filter.__class__}"
            )
            if self.ground_truth_activations_path:
                z_savanna = torch.load(
                    f"{self.ground_truth_activations_path}/pre_filter_{self.layer_idx}.pt"
                )
                activation_diff = (z - z_savanna.squeeze()).abs()
                activations_logger.info(
                    f"pre filter activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                )
        z, inference_params = self.filter(
            z, inference_params=inference_params, padding_mask=padding_mask
        )

        if self.print_activations:
            activations_logger.info(
                f"post postgate: {z} {z.min()} {z.max()} {self.filter.__class__}"
            )
            activations_logger.info(
                f"post out proj: {self.out_filter_dense(z)} {self.out_filter_dense(z).min()} {self.out_filter_dense(z).max()} {self.out_filter_dense.__class__}"
            )
            activations_logger.info(
                f"post mixer dense and residual: {self.out_filter_dense(z) + u} {(self.out_filter_dense(z) + u).min()} {(self.out_filter_dense(z) + u).max()}"
            )
            activations_logger.info(
                f"post mixer dense: {self.out_filter_dense(z)} {self.out_filter_dense(z).min()} {self.out_filter_dense(z).max()}"
            )
            activations_logger.info(f"post mixer: {z} {z.min()} {z.max()}")
            if self.ground_truth_activations_path:
                z_savanna = torch.load(
                    f"{self.ground_truth_activations_path}/post_filter_{self.layer_idx}.pt"
                )
                activation_diff = (z - z_savanna.squeeze()).abs()
                activations_logger.info(
                    f"post filter activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                )

                z_savanna = torch.load(
                    f"{self.ground_truth_activations_path}/post_out_proj_{self.layer_idx}.pt"
                )
                z_ = F.linear(z, self.out_filter_dense.weight)
                activation_diff = (z_ - z_savanna.squeeze()).abs()
                activations_logger.info(
                    f"post out proj activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                )

        z_in = self.out_filter_dense(z) + u

        # if self.layer_idx == 0:
        #    z_in = z_savanna.squeeze() + u + self.out_filter_dense.bias

        if type(padding_mask) == torch.Tensor:  # guard against bias
            z_in = z_in * padding_mask[..., None]

        y = self.res_mlp_norm(z_in)

        return y, inference_params


def get_block(config, layer_idx, flash_fft=None):
    if layer_idx in config.attn_layer_idxs:
        return AttentionBlock(config, layer_idx)
    elif layer_idx in config.hcl_layer_idxs:
        block = ParallelGatedConvBlock(config, layer_idx)
        if config.get("use_flashfft", "False"):
            block.filter.fftconv_fn = flash_fft
        return block
    elif layer_idx in config.hcm_layer_idxs:
        block = ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcm_filter_groups,
            fir_inner_filter_length=config.hcm_filter_length,
        )
        return block
    elif layer_idx in config.hcs_layer_idxs:
        block = ParallelGatedConvBlock(
            config,
            layer_idx,
            hyena_filter_groups=config.hcs_filter_groups,
            fir_inner_filter_length=config.hcs_filter_length,
        )
        return block
    else:
        raise NotImplementedError


class StripedHyena(nn.Module):
    def __init__(self, config):
        super().__init__()
        fixup_te_workspace()  # Workaround global cublas workspaces in TE

        self.config = config
        self.print_activations = config.get("print_activations", False)

        if self.print_activations:
            enable_activations_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.ground_truth_activations_path = config.get(
            "ground_truth_activations_path", None
        )
        self.logger.info(f"Initializing StripedHyena with config: {config}")

        with torch.device("cuda:0" if torch.cuda.is_available() else "cpu"):
            self.embedding_layer = VocabParallelEmbedding(config)

        if config.get("use_flashfft", "True"):
            try:
                from flashfftconv import FlashFFTConv

                self.flash_fft = FlashFFTConv(config.seqlen, dtype=torch.bfloat16)
            except ImportError:
                "flashfftconv not installed"
        else:
            self.flash_fft = None
        if not self.config.get("evo2_style_activations", False):
            self.logger.warning(
                "⚠️  Not using Evo2 style activations  ⚠️\n"
                "⚠️ Set 'evo2_style_activations: True' in config if you are using Evo 2 checkpoints ⚠️"
            )
        self.logger.info(f"Initializing {config.num_layers} blocks...")
        self.blocks = nn.ModuleList()
        self.block_idx_to_device = {}

        # Calculate layers per GPU
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        layers_per_gpu = math.ceil(config.num_layers / num_gpus)
        self.logger.info(
            f"Distributing across {num_gpus} GPUs, approximately {layers_per_gpu} layers per GPU"
        )

        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=args.head_dim or args.dim // args.n_heads,
            max_seqlen=args.max_seqlen,
            rope_use_fp32_in_outer_product=args.rope_use_fp32_in_outer_product,
        )

        for layer_idx in tqdm(range(config.num_layers)):
            # Determine which GPU should handle this layer
            device_idx = min(layer_idx // layers_per_gpu, num_gpus - 1)
            device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"

            with torch.device(device):
                # TELinear uses `device="cuda"` device to allocate empty bias
                # tensor. This makes sure that the empty tensor is allocated on the
                # correct device. (torch.device(), unlike torch.cuda.device(),
                # doesn't override current CUDA device.)
                with torch.cuda.device(device):
                    block = get_block(config, layer_idx, flash_fft=self.flash_fft)
                    move_to_device(block, device)

            self.blocks.append(block)
            self.block_idx_to_device[layer_idx] = device
            self.logger.info(f"Assigned {layer_idx=} to {device=}")
            self.logger.info(
                f"Parameter count for block {layer_idx}: {sum(p.numel() for p in self.blocks[-1].parameters())}"
            )


    def block_idx_to_name(self, block_idx):
        if block_idx in self.config.attn_layer_idxs:
            return "mha"
        elif block_idx in self.config.hcl_layer_idxs:
            return "hcl"
        elif block_idx in self.config.hcm_layer_idxs:
            return "hcm"
        elif block_idx in self.config.hcs_layer_idxs:
            return "hcs"
        else:
            raise ValueError(f"Block index {block_idx} not found")

    def cross_device_transfer(self, x, block_idx):
        if (
            self.block_idx_to_device[max(block_idx - 1, 0)]
            != self.block_idx_to_device[block_idx]
        ):
            x = x.to(self.block_idx_to_device[block_idx])
        return x

    def forward(self, x, tok_idx, mask, attn_impl, padding_mask=None):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        if isinstance(padding_mask, torch.Tensor):
            x = x * padding_mask[..., None]

        for block_idx, block in enumerate(self.blocks):

            x = self.cross_device_transfer(x, block_idx)
            if block_idx in self.config.attn_layer_idxs:
                x, _ = block(x, freq_cis, tok_idx, mask, attn_impl, padding_mask=padding_mask)
            else:
                x, _ = block(x, padding_mask=padding_mask)

            if self.print_activations:
                activations_logger.info(
                    f"post block {block_idx}: {x}, {x.min()}, {x.max()}"
                )
                if self.ground_truth_activations_path:
                    x_savanna = torch.load(
                        f"{self.ground_truth_activations_path}/post_block_{block_idx}.pt"
                    )
                    activation_diff = (x - x_savanna.squeeze()).abs()
                    activations_logger.info(
                        f"post block {block_idx} activation_diff: {activation_diff.max()}, {activation_diff.mean()}"
                    )

        return x, None

    def initialize_inference_params(self, max_seqlen=None):
        ## Input seqlen takes priority over config!
        ## WARNING: This avoids potential errors but means the model can be used beyond length it was trained at
        config_seqlen = self.config.get("max_seqlen", None)
        if config_seqlen is None:
            print("No max_seqlen found in config!!! using default value of 8192")
            config_seqlen = 8192
        new_max_seqlen = max_seqlen if max_seqlen != None else config_seqlen
        # self.config["max_seqlen"] = new_max_seqlen
        ## Note: changing the stored config max_seqlen will change the max_seqlen used in flash attention, leading to minor logit differences
        print(f"Initializing inference params with max_seqlen={new_max_seqlen}")

        inference_params_dict = {
            "mha": InferenceParams(
                max_seqlen=new_max_seqlen,
                max_batch_size=self.config.get("max_batch_size", 1),
                seqlen_offset=0,
            ),
            "hcl": HyenaCascadeIIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                state_dim=self.config.state_size,
                seqlen_offset=0,
            ),
            "hcm": HyenaCascadeFIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                fir_inner_filter_length=self.config.hcm_filter_length,
                seqlen_offset=0,
            ),
            "hcs": HyenaCascadeFIRInferenceParams(
                fir_filter_length=self.config.short_filter_length,
                fir_inner_filter_length=self.config.hcs_filter_length,
                seqlen_offset=0,
            ),
        }
        return inference_params_dict

    def precompute_filters(self, L, device):
        for block_idx, block in enumerate(self.blocks):
            if type(block) == ParallelGatedConvBlock:
                if type(block.filter) == HyenaCascade:
                    L = block.filter.long_fir_threshold or L
                    print_rank_0(f"Precomputing filters, L={L}...")

                    filter_dtype = torch.float16 if L >= 2048 else torch.float32

                    block.filter._set_time(L, device)
                    residues, poles = (
                        block.filter.residues.to(torch.float16),
                        block.filter.poles.to(torch.float16),
                    )

                    block.filter.h = (residues * poles**block.filter.t).real.sum(1)[
                        None
                    ]
                    block.filter.h = block.filter.h.to(dtype=filter_dtype)

    def load_poles_residues(self, path):
        "Load different poles and residues for each layer."
        for block_idx, block in enumerate(self.blocks):
            if type(block) == ParallelGatedConvBlock:
                if type(block.filter) == HyenaCascade:
                    self.logger.info(
                        f"Loading approximatepoles and residues for block {block_idx}"
                    )
                    poles = torch.load(
                        path + f"/approx_poles_{block_idx + 1}.pt", map_location="cpu"
                    )
                    poles = torch.view_as_real(poles)
                    residues = torch.load(
                        path + f"/approx_residues_{block_idx + 1}.pt",
                        map_location="cpu",
                    )
                    residues = torch.view_as_real(residues)
                    poles = poles.permute(1, 0, 2).unsqueeze(-2)
                    residues = residues.permute(1, 0, 2).unsqueeze(-2)

                    block.filter.poles = nn.Parameter(poles)
                    block.filter.residues = nn.Parameter(residues)

    def custom_load_state_dict(self, state_dict, strict=True):
        """
        Post-processes the state_dict to convert savanna checkpoints to vortex checkpoints.
        """
        self.logger.debug(
            f"Loading state dict: {state_dict}, (ignoring extra keys) with strict: {strict}"
        )
        model_dict = self.state_dict()

        # Find keys that are in model_dict but not in state_dict
        missing_in_state_dict = model_dict.keys() - state_dict.keys()
        # Find keys that are in state_dict but not in model_dict
        extra_in_state_dict = state_dict.keys() - model_dict.keys()

        if missing_in_state_dict:
            print(f"Keys missing in state_dict: {missing_in_state_dict}")
        if extra_in_state_dict:
            print(f"Extra keys in state_dict: {extra_in_state_dict}")

        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}

        if all("._extra_state" in k for k in missing_in_state_dict):
            self.logger.info(
                "Checkpoint has no FP8 extra state, will be using initial state."
            )
            for k in missing_in_state_dict:
                filtered_dict[k] = None

        self.load_state_dict(filtered_dict, strict=strict)
        fixup_fp8_extra_states(self)

        if self.config.get("column_split", True):
            self.logger.info("Adjusting Wqkv for column split (permuting rows)")
            for layer_idx, block in enumerate(self.blocks):
                if type(block) == AttentionBlock:
                    target_device = block.inner_mha_cls.Wqkv.weight.device

                    Wqkv = state_dict[f"blocks.{layer_idx}.inner_mha_cls.Wqkv.weight"]
                    try:
                        bias = state_dict[f"blocks.{layer_idx}.inner_mha_cls.Wqkv.bias"]
                    except:
                        bias = None

                    size_att_head = block.hidden_size_per_attention_head

                    Wqkv = Wqkv.permute(1, 0)
                    Wqkv = Wqkv.reshape(
                        block.hidden_size, block.num_attention_heads, 3, size_att_head
                    )
                    Wq, Wk, Wv = Wqkv.unbind(dim=-2)
                    Wq = Wq.reshape(block.hidden_size, -1)
                    Wk = Wk.reshape(block.hidden_size, -1)
                    Wv = Wv.reshape(block.hidden_size, -1)
                    Wqkv = torch.cat([Wq, Wk, Wv], dim=-1)
                    Wqkv = Wqkv.permute(1, 0)

                    # Single device transfer at the end
                    block.inner_mha_cls.Wqkv.weight.data = Wqkv.to(target_device)

                    if bias is not None:
                        bias = bias.cpu()  # Process on CPU
                        bias = bias.reshape(block.num_attention_heads, 3, size_att_head)
                        bias_q, bias_k, bias_v = bias.unbind(dim=-2)
                        bias_q = bias_q.reshape(block.hidden_size)
                        bias_k = bias_k.reshape(block.hidden_size)
                        bias_v = bias_v.reshape(block.hidden_size)
                        bias = torch.cat([bias_q, bias_k, bias_v], dim=0)
                        try:
                            block.inner_mha_cls.Wqkv.bias.data = bias.to(target_device)
                        except:
                            pass

    def to_bfloat16_except_pr_lc(self, to_float32=False):
        """Convert all parameters to bfloat16 except for the poles and residues.

        Particularly important for longer prompts.
        """
        excluded_shapes = [(4096, 1, 128)]
        for k, p in self.named_parameters():
            if "projections" not in k:  # avoid TE linears
                if (
                    "log_poles" not in k
                    and "residues" not in k
                    and p.shape not in excluded_shapes
                ):
                    p.data = p.data.to(torch.bfloat16)
                else:
                    if to_float32:
                        p.data = p.data.to(torch.float32)
        for k, b in self.named_buffers():
            if "inv_freq" in k:
                if to_float32:
                    b.data = b.data.to(torch.float32)

'''
# Copyright (c) 2024, Michael Poli.

import gc

import torch
import torch.nn.functional as F

try:
    pass
except:
    pass
from vortex.model.utils import column_split
from vortex.logging import activations_logger

IIR_PREFILL_MODES = [
    "recurrence",
    "modal-fft",
    "hybrid-modal-recurrence",
    "modal-scan",
    "canonical-fft",
    "iir-fir-caching",
]


def adjust_filter_shape_for_broadcast(u, h):
    h = h.squeeze()  # Standardize to [D, L] from [1, D, L] and [D, 1, L]

    # Case: u: [B, D, L], k_f: [D, L]
    if len(u.shape) > len(h.shape):
        h = h.unsqueeze(0)

    # Case: u: [B, D1, D2, L], k_f: [B, D, L]
    if len(u.shape) > 3:
        h = h.unsqueeze(1)
    return h


def fftconv_func(
    u,
    k,
    D,
    dropout_mask,
    gelu=True,
    k_rev=None,
    bidirectional=False,
    print_activations=False,
    layer_idx=None,
    **kwargs,
):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen

    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    k_f = adjust_filter_shape_for_broadcast(u, k_f)
    k = k.squeeze()

    if bidirectional:
        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
        k, k2 = k.split(k.shape[1] // 2, dim=1)
        k2_f = torch.fft.rfft(k2, n=fft_size) / fft_size
        y1 = u_f * k_f
        y2 = u_f.conj() * k2_f.conj()

        y = torch.fft.irfft(y1 + y2, n=fft_size, norm="forward")[..., :seqlen]

    else:
        if k_rev is not None:
            k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
            k_f = k_f + k_rev_f.conj()

        u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

        y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    if print_activations:
        activations_logger.info(f"post fftconv pre bias {y} {y.min()} {y.max()}")

    out = y + u * D.unsqueeze(-1)

    if print_activations:
        activations_logger.info(f"post fftconv post bias {out} {out.min()} {out.max()}")

    return out.to(dtype=u.dtype)


def canonicalize_modal_system(poles, residues):
    """Canonicalize a modal system.

    Args:
        poles (Tensor): The poles of the system.
        residues (Tensor): The residues of the system.

    Returns:
        Tuple[Tensor, Tensor]: The canonicalized poles and residues.
    """
    raise NotImplementedError


def list_tensors(idx):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and isinstance(obj, torch.Tensor):
                # dump to log
                print(type(obj), obj.size())
                el = obj[0]
                with open(f"tensors_{idx}.txt", "a") as f:
                    f.write(f"{type(obj)} {obj.size()} {el}\n")
        except Exception:
            pass


class HyenaInferenceEngine:
    def __init__(
        self,
        fir_fn=None,
        iir_prefill_style="modal-fft",
        layer_idx=None,
        ground_truth_activations_path=None,
        print_activations=False,
        hyena_flip_x1x2=False,
    ) -> None:
        self.fir_fn = fir_fn
        assert iir_prefill_style in IIR_PREFILL_MODES, f"iir_prefill_style must be one of {IIR_PREFILL_MODES}"
        self.iir_prefill_style = iir_prefill_style
        self.layer_idx = layer_idx
        self.low_mem_mode = False
        self.ground_truth_activations_path = ground_truth_activations_path
        self.print_activations = print_activations
        self.hyena_flip_x1x2 = hyena_flip_x1x2

    def parallel_fir(
        self,
        fir_fn,
        u,
        weight,
        bias,
        L,
        dims,
        groups=None,
        gated_bias=False,
        column_split_hyena=False,
        dim_last=True,
        fir_length=3,
        gate=False,
        inference_params=None,
        prefill_mode=None,
        padding_mask=None,
    ):
        L = u.shape[1] if dim_last else u.shape[2]
        if gate:
            hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
            # Compatibility with training infra that column splits the projections
            if column_split_hyena:
                x2, x1, v = column_split(u, num_attention_heads, hidden_size_per_attention_head)
            else:
                x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
            if self.hyena_flip_x1x2:
                x1, x2 = x2, x1
            u = x1 * v

            if self.print_activations:
                activations_logger.info(f"q: {x2}, {x2.min()}, {x2.max()}")
                activations_logger.info(f"k: {x1}, {x1.min()}, {x1.max()}")
                activations_logger.info(f"v: {v}, {v.min()}, {v.max()}")
                activations_logger.info(f"post pregate: {u}, {u.min()}, {u.max()}")

        # prepare input layout, dimensions and dispatch to fir kernel
        # Deprecated
        if fir_fn != torch.nn.functional.conv1d:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L
            z = fir_fn(u)[:, :L]  # B, L, D

        elif fir_length >= 128:
            with torch.autocast("cuda"):
                z = fftconv_func(
                    u.to(torch.float32),
                    weight[:, :, :L].to(torch.float32),
                    bias,
                    None,
                    gelu=False,
                    bidirectional=False,
                    print_activations=self.print_activations,
                    groups=groups,
                    layer_idx=self.layer_idx,
                )
                z = z.to(u.dtype)
        else:
            if dim_last:
                u = u.permute(0, 2, 1)  # B, D, L

            if groups is None:
                g = u.shape[1]
            else:
                g = groups

            z = fir_fn(
                u.to(torch.float32),
                weight.to(torch.float32),
                bias=None,
                stride=1,
                padding=fir_length - 1,
                groups=u.shape[1],  # always set to D, regardless of filter grouping
            )[..., :L]
            if self.print_activations:
                activations_logger.info(f"post filter: {z}, {z.min()}, {z.max()}")

            z = z.to(u.dtype)

            if gated_bias is False:
                if self.print_activations:
                    activations_logger.info(f"post dw conv {z} {z.min()} {z.max()}")
                    # if self.ground_truth_activations_path:
                    #     z_savanna = torch.load(f"{self.ground_truth_activations_path}/post_dw_conv_{self.layer_idx}.pt")
                    #     z_savanna = z_savanna.permute(1, 2, 0)
                    #     z_diff = (z.squeeze() - z_savanna.squeeze()).abs().max()
                    #     activations_logger.info(f"dw_conv_diff: {z_diff}")

            if bias is not None:
                if gated_bias:
                    z = z + bias[None, :, None] * u
                else:
                    z = z + bias[None, :, None]

        # handle padding post fir, the only place with biases
        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[:, None]

        if gate:
            # if self.layer_idx == 1:
            #    breakpoint()
            z = x2 * z

            if self.print_activations:
                activations_logger.info(f"hyena filter: {weight}, {weight.min()}, {weight.max()}")
                activations_logger.info(f"post postgate: {z}, {z.min()}, {z.max()}")
                # if self.ground_truth_activations_path:
                #     q_savanna = torch.load(f"{self.ground_truth_activations_path}/q_{self.layer_idx}.pt")
                #     k_savanna = torch.load(f"{self.ground_truth_activations_path}/k_{self.layer_idx}.pt")
                #     v_savanna = torch.load(f"{self.ground_truth_activations_path}/v_{self.layer_idx}.pt")

                #     q_diff = (x2 - q_savanna).abs()
                #     k_diff = (x1 - k_savanna).abs()
                #     v_diff = (v - v_savanna).abs()

                #     activations_logger.info(f"q_diff: {q_diff.max()}, {q_diff.mean()}")
                #     activations_logger.info(f"k_diff: {k_diff.max()}, {k_diff.mean()}")
                #     activations_logger.info(f"v_diff: {v_diff.max()}, {v_diff.mean()}")

                #     h_savanna = torch.load(f"/home/zymrael/checkpoints/evo2/activations/savanna/hyena_filter_{self.layer_idx}.pt")
                #     h_diff = (weight[..., :h_savanna.shape[-1]].squeeze() - h_savanna.squeeze()).abs()

                #     activations_logger.info(f"h_diff: {h_diff.max()}, {h_diff.mean()}")

        if inference_params is not None:
            fir_state = u[..., -fir_length + 1 :]
        else:
            fir_state = None

        return z, fir_state

    def parallel_iir(
        self,
        z_pre,
        h,
        D,
        L,
        poles,
        residues,
        t,
        dims,
        layer_idx,
        inference_params=None,
        prefill_style="fft",
        fftconv_fn=None,
        padding_mask=None,
        use_flashfft=False,
        column_split_hyena=False,
        long_fir_threshold=None,
    ):
        """Compute the output state of the short convolutional filter."""
        fft_size = 2 * L
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        # Compatibility with training infra that column splits the projections
        if column_split_hyena:
            z = z_pre.reshape(
                z_pre.shape[0],
                num_attention_heads,
                3 * hidden_size_per_attention_head,
                z_pre.shape[2],
            )
            x2, x1, v = (
                z[:, :, :hidden_size_per_attention_head],
                z[
                    :,
                    :,
                    hidden_size_per_attention_head : 2 * hidden_size_per_attention_head,
                ],
                z[:, :, 2 * hidden_size_per_attention_head :],
            )
            x2, x1, v = (
                x2.reshape(x2.shape[0], -1, x2.shape[-1]),
                x1.reshape(x1.shape[0], -1, x1.shape[-1]),
                v.reshape(v.shape[0], -1, v.shape[-1]),
            )
        else:
            x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)

        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1

        x1v = x1 * v

        if inference_params is not None and prefill_style == "recurrence":
            y = self.prefill_via_direct_recurrence(
                inference_params=inference_params,
                x1v=x1v,
                L=L,
                poles=poles,
                residues=residues,
            )

        else:
            if use_flashfft and (L % 2) == 0:  # only works with even L
                y = fftconv_fn(
                    x1v.to(dtype=torch.bfloat16).contiguous(),
                    h.to(dtype=torch.float32),
                )
                X_s = None

            elif long_fir_threshold is None:
                H = torch.fft.rfft(h.to(dtype=torch.float32), n=fft_size) / fft_size
                X_s = torch.fft.fft(x1v.to(dtype=torch.float32), n=fft_size)
                X = X_s[..., : H.shape[-1]]
                if len(z_pre.shape) > 3:
                    H = H.unsqueeze(1)
                y = torch.fft.irfft(X * H, n=fft_size, norm="forward")[..., :L]

            else:
                assert h.shape[0] == 1, "batch size must be 1 for long_fir_threshold"
                h = h[0][:, None]  # rearrange to d, 1, l for depthwise conv1d
                h = h[..., :long_fir_threshold]
                y = F.conv1d(
                    x1v,
                    h.to(dtype=x1v.dtype),
                    stride=1,
                    groups=x1v.shape[1],
                    padding=h.shape[-1] - 1,
                )[..., :L]
        # if self.layer_idx == 2:
        #    breakpoint()
        y = y.to(dtype=x1v.dtype)
        y = (y + x1v * D.unsqueeze(-1)) * x2

        if self.print_activations:
            activations_logger.info(f"hyena filter: {h}, {h.min()}, {h.max()}")
            activations_logger.info(f"post hyena iir gate: {y}, {y.min()}, {y.max()}")
            activations_logger.info(f"q: {x2}, {x2.min()}, {x2.max()}")
            activations_logger.info(f"k: {x1}, {x1.min()}, {x1.max()}")
            activations_logger.info(f"v: {v}, {v.min()}, {v.max()}")
            # if self.ground_truth_activations_path:
            #     q_savanna = torch.load(f"{self.ground_truth_activations_path}/q_{self.layer_idx}.pt")
            #     k_savanna = torch.load(f"{self.ground_truth_activations_path}/k_{self.layer_idx}.pt")
            #     v_savanna = torch.load(f"{self.ground_truth_activations_path}/v_{self.layer_idx}.pt")

            #     q_diff = (x2 - q_savanna).abs()
            #     k_diff = (x1 - k_savanna).abs()
            #     v_diff = (v - v_savanna).abs()

            #     activations_logger.info(f"q_diff: {q_diff.max()}, {q_diff.mean()}")
            #     activations_logger.info(f"k_diff: {k_diff.max()}, {k_diff.mean()}")
            #     activations_logger.info(f"v_diff: {v_diff.max()}, {v_diff.mean()}")

            #     h_savanna = torch.load(f"/home/zymrael/checkpoints/evo2/activations/savanna/hyena_filter_{self.layer_idx}.pt")

            #     h_diff = (h[..., :h_savanna.shape[-1]].squeeze() - h_savanna.squeeze()).abs()
            #     activations_logger.info(f"h_diff: {h_diff.max()}, {h_diff.mean()}")

        if inference_params is not None:
            if prefill_style == "fft":
                self.prefill_via_modal_fft(
                    inference_params=inference_params,
                    x1v=x1v,
                    X_s=X_s,
                    L=L,
                    t=t,
                    poles=poles,
                    dims=dims,
                    layer_idx=layer_idx,
                    use_flashfft=use_flashfft,
                    fftconv_fn=fftconv_fn,
                )

            elif prefill_style == "recurrence":
                # recurrent prefill is done before
                pass
            else:
                raise NotImplementedError
            if self.low_mem_mode:
                # TODO: smarter gc
                del z_pre, x2, x1, v, x1v, h, poles, residues
                torch.cuda.empty_cache()

        return y.permute(0, 2, 1)

    def step_fir(self, u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
        """Steps forward FIR filters in the architecture.

        FIR filters generally include truncated convolutions in Hyena with an explicit or hybrid time-domain parametrization:
        * Short FIR filters in Hyena featurizers
        * Short and medium FIR filters in Hyena operators

        Note:
            `fir_state` contains the last FIR filter length - 1 elements of `u`: `u_(L-2), u_{L-1), ...`
            We assume dimensions of `short_filter_weight` to be `[d, 1, short_filter_len]`.
        """
        weight = weight.squeeze()

        cache_size = fir_state.shape[-1]
        filter_length = weight.shape[-1]
        if flip_filter:
            weight = weight.flip(-1)
            weight = weight[..., -cache_size - 1 :].unsqueeze(0)
        else:
            weight = weight[..., : cache_size + 1].unsqueeze(0)

        input_dtype = u.dtype
        weight = weight.to(torch.float32)
        u = u.to(torch.float32)
        fir_state = fir_state.to(torch.float32)
        bias = bias.to(torch.float32) if bias is not None else None

        h0, h = weight[..., -1], weight[..., :-1]
        y = h0 * u + torch.sum(fir_state * h, dim=-1)

        if bias is not None:
            if gated_bias:
                y = y + bias * u
            else:
                y = y + bias

        # Update the state
        if cache_size < filter_length - 1:
            fir_state = torch.cat([fir_state, u[..., None]], dim=-1)
        else:
            fir_state = torch.roll(fir_state, -1, dims=2)
            fir_state[..., -1] = u

        return y.to(input_dtype), fir_state

    def step_iir(self, x2, x1, v, D, residues, poles, iir_state, iir_groups=1):
        # TODO: kernelize
        x1v = x1 * v
        poles = torch.exp(poles)  # poles arg contains log_poles
        poles = poles[..., 0][None]  # squeeze dummy seqlen dim and add dummy batch dim
        residues = residues[None]  # add dummy batch dim
        iir_state = poles * iir_state + x1v[..., None]

        res_state = torch.sum(residues * iir_state, dim=-1)

        if iir_groups > 1:
            raise NotImplementedError
        # if self.layer_idx == 2:
        #    breakpoint()
        y = x2 * (res_state + D * x1v)

        return y, iir_state

    def prefill_via_fir_caching(self, u, inference_params, L, *args, **kwargs):
        """Turns the IIR filter into a FIR and uses a cache for decoding."""
        raise NotImplementedError(":)")

    def prefill_via_direct_recurrence(self, inference_params, x1v, L, residues, poles, *args, **kwargs) -> torch.Tensor:
        """
        Compute the IIR state via explicit recurrence (modal form)

        This is the most memory efficient prefilling method for Hyena filters.

        Note:
            dtypes: [state: float32, poles: float32, x1v: bfloat16, output: bfloat16]
        """
        state_dim = poles.shape[1]
        x1v_ = x1v[..., None, None]  # b, d, l, sdim, reim
        x1v_ = x1v_.repeat(1, 1, 1, state_dim, 2)  # b, d, l, sdim, reim
        x1v_[..., 1] = 0

        state = 0 * x1v_[:, :, 0]
        output = 0 * x1v_[:, :, :, 0, 0]  # b, d, l

        # suppress dummy seqlen dimension
        poles = poles[:, :, 0][None]
        residues = residues[:, :, 0][None].repeat(x1v_.shape[0], 1, 1, 1)  # b, d, sdim, reim

        # state: b, d, sdim, reim
        # poles: 1, d, sdim, reim
        # x1v_: b, d, l, sdim, reim
        for i in range(L):
            state[..., 0] = poles[..., 0] * state[..., 0] - poles[..., 1] * state[..., 1] + x1v_[:, :, i, :, 0]
            state[..., 1] = poles[..., 0] * state[..., 1] + poles[..., 1] * state[..., 0] + x1v_[:, :, i, :, 1]
            output[:, :, i] = torch.sum(residues * state, dim=-2)[..., 0]  # .real

        inference_params.state_dict[self.layer_idx] = state.to(dtype=torch.float32)

        return output

    def prefill_via_hybrid_recurrence(self, inference_params, u, log_poles, x1v_f_a, L, *args, **kwargs):
        """
        Compute the IIR state via hybrid recurrence-convolution over blocks
        """
        raise NotImplementedError(":)")

    def prefill_via_scan(self, u, inference_params=None, *args, **kwargs):
        raise NotImplementedError

    def prefill_via_canonical_fft(self, u, inference_params=None, *args, **kwargs):
        """
        Compute the IIR state via a single FFT

        This is the most memory efficient "parallelized" prefilling method for Hyena.

        From: https://arxiv.org/abs/2310.18780
        """
        raise NotImplementedError(":)")

    def prefill_via_modal_fft(
        self,
        inference_params,
        x1v,
        L,
        poles,
        t,
        dims,
        layer_idx,
        X_s=None,
        use_flashfft=False,
        fftconv_fn=None,
        state_dtype=torch.float32,
        *args,
        **kwargs,
    ):
        """
        Compute the IIR state via a single FFT
        """
        # When the model has a long convolution derived from a recurrence in modal form and prefill_style is "fft",
        # we split the filter into poles and residues and reuse FFT computation on the input.
        hidden_size, _, _, state_size, hyena_filter_groups = dims

        assert X_s is not None
        bs = x1v.shape[0]
        fft_size = 2 * L
        # poles = torch.view_as_complex(poles.to(torch.float32))
        state_s = (poles.to(torch.float32) * t).exp()

        # state_s = poles**t
        state_S = torch.fft.fft(state_s, n=fft_size).repeat(bs, 1, 1, 1)  # B, D, state_dim, 2 * L
        if hyena_filter_groups > 1:
            state_S = state_S.repeat_interleave(hidden_size // hyena_filter_groups, 1)
        state = torch.fft.ifft(X_s[..., None, :] * state_S, n=fft_size)
        inference_params.state_dict[layer_idx] = state[..., L - 1].to(dtype=state_dtype)

    def _compute_state(self, log_poles, u, t, L, *args, **kwargs):
        """
        Compute the IIR state given an input `u` and log_poles of the modal system.
        """
        bs = u.shape[0]
        fft_size = 2 * L
        U = torch.fft.rfft(u.to(torch.float32), n=fft_size)
        fft_size = 2 * L
        x = (log_poles * t).exp()
        # [batch, hidden_size, state_dim, 2 * seqlen]
        X = torch.fft.fft(x, n=fft_size).repeat(bs, 1, 1, 1)
        state = torch.fft.ifft(U[..., None, :] * X, n=fft_size)[..., :L]
        return state


class HyenaFilter:
    """Handles Hyena filter computations including FFT and direct convolution."""

    def __init__(self, use_flash_fft=False):
        self.use_flash_fft = use_flash_fft

    def fft_conv(self, u, k, D, **kwargs):
        """FFT-based convolution implementation."""
        seqlen = u.shape[-1]
        fft_size = 2 * seqlen

        k_f = self._prepare_filter(k, u, fft_size)
        y = self._compute_fft_conv(u, k_f, fft_size, seqlen, **kwargs)

        return y + u * D.unsqueeze(-1)

    def _prepare_filter(self, k, u, fft_size):
        """Prepare filter for FFT convolution."""
        k_f = torch.fft.rfft(k, n=fft_size) / fft_size
        return adjust_filter_shape_for_broadcast(u, k_f)
'''