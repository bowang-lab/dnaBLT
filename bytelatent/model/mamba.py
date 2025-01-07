import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Tuple
from transformers import Mamba2Config, Mamba2Model


class GlobalMamba(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = args.dropout
        self.dim = args.dim

        self.config = Mamba2Config(
            vocab_size=args.vocab_size,
            state_size=args.state_size,
            hidden_size=args.dim,
            num_hidden_layers=args.n_layers,
            num_heads=args.n_heads,
            head_dim=args.dim // args.n_heads,
            use_cache=True,
            rms_norm=True,
            return_dict=True,
        )
        self.model = Mamba2Model(self.config)

        # Match transformer's projection capability
        self.token_embedding_projection = None
        if args.dim_token_emb is not None and args.dim_token_emb != self.dim:
            self.token_embedding_projection = nn.Linear(
                args.dim_token_emb,
                args.dim,
                bias=False,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        """
        Match GlobalTransformer.forward signature and behavior
        """
        bs, seqlen = tokens.shape
        h = embeds

        # Handle mask same way as transformer
        if mask is None:
            attention_mask = torch.ones(
                (bs, seqlen), dtype=torch.bool, device=tokens.device
            )
        elif isinstance(mask, torch.Tensor):
            attention_mask = mask
        elif isinstance(mask, str) and mask == "causal":
            attention_mask = torch.triu(
                torch.ones(bs, seqlen, dtype=torch.bool, device=tokens.device),
                diagonal=1,
            ).logical_not()

        if self.token_embedding_projection is not None and h.shape[-1] != self.dim:
            h = self.token_embedding_projection(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        outputs = self.model(
            inputs_embeds=h,
            use_cache=cache is not None,
            cache_params=cache,
            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state, outputs.cache_params

    def init_weights(self, init_base_std: float):
        if self.token_embedding_projection is not None:
            nn.init.trunc_normal_(
                self.token_embedding_projection.weight,
                mean=0.0,
                std=init_base_std,
                a=-3 * init_base_std,
                b=3 * init_base_std,
            )
