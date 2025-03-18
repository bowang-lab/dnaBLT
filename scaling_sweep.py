"""

SCALING LAWS EXPERIMENTS

- Entropy threshold calibration -> get reasonable token counts
- For fixed FLOP budgets, train sweeping (model, data, entropy) triplets
    - Implement FLOPs based early stopping (plateau)

- Find equation relating optimal (model, data, entropy) triplet to FLOPs
- Train big boi given prospective FLOPs


"""


def feed_forward_flops(layers, hidden_state, feed_forward_mult):
    return 2 * layers * 2 * hidden_state * feed_forward_mult * hidden_state


def qkvo_flops(layers, hidden_state, ratio_q2kv):
    return (ratio_q2kv * 2 + 2) * 2 * layers * hidden_state**2


def attention_flops(layers, hidden_state_key, n_heads, context):
    return 4 * layers * hidden_state_key * n_heads * (context + 1) // 2


def deembedding_flops(hidden_state, vocab):
    return 2 * hidden_state * vocab


def transformer_flops(
    hidden_state, layers, context, vocab, feed_forward_mult, n_heads, ratio_q2k=1
):
    return (
        feed_forward_flops(layers, hidden_state, feed_forward_mult)
        + qkvo_flops(layers, hidden_state, ratio_q2k)
        + attention_flops(layers, hidden_state // n_heads, n_heads, context)
        + deembedding_flops(hidden_state, vocab)
    )


def cross_attention_flops(layers, hidden_state_key, n_heads, patch_size, ratio_q2k):
    return attention_flops(layers, hidden_state_key, n_heads, patch_size) + qkvo_flops(
        layers, hidden_state_key * n_heads, ratio_q2k
    )


def forward_blt_flops(
    seq_len,
    patch_size,
    hidden_state_g,
    layers_g,
    hidden_state_e,
    layers_e,
    window_e,
    hidden_state_d,
    layers_d,
    window_d,
    ratio_patchdim2bytedim,
    n_heads_e,
    n_heads_g,
    n_heads_d,
    vocab,
    feed_forward_mult,
):
    return (
        transformer_flops(
            hidden_state_g,
            layers_g,
            seq_len / patch_size,
            0,
            feed_forward_mult,
            n_heads_g,
        )
        / patch_size
        + transformer_flops(
            hidden_state_e, layers_e, window_e, 0, feed_forward_mult, n_heads_e
        )
        + transformer_flops(
            hidden_state_d, layers_d, window_d, vocab, feed_forward_mult, n_heads_d
        )
        + cross_attention_flops(
            layers_e,
            hidden_state_e // n_heads_e,
            n_heads_e,
            patch_size,
            patch_size / ratio_patchdim2bytedim,
        )
        * (ratio_patchdim2bytedim / patch_size)
        + cross_attention_flops(
            layers_d,
            hidden_state_d // n_heads_d,
            n_heads_d,
            ratio_patchdim2bytedim,
            ratio_patchdim2bytedim / patch_size,
        )
    )


def total_flops(
    tokens,
    seq_len,
    patch_size,
    hidden_state_g,
    layers_g,
    hidden_state_e,
    layers_e,
    window_e,
    hidden_state_d,
    layers_d,
    window_d,
    ratio_patchdim2bytedim,
    n_heads_e,
    n_heads_g,
    n_heads_d,
    vocab=256,
    feed_forward_mult=4,
):
    return (
        3
        * tokens
        * forward_blt_flops(
            seq_len,
            patch_size,
            hidden_state_g,
            layers_g,
            hidden_state_e,
            layers_e,
            window_e,
            hidden_state_d,
            layers_d,
            window_d,
            ratio_patchdim2bytedim,
            n_heads_e,
            n_heads_g,
            n_heads_d,
            vocab,
            feed_forward_mult,
        )
    )


def total_parameters(L, d_model):
    return L * (12 * d_model**2 + 13 * d_model)


# Validate FLOPs fall in expected range with found value
# Tokens: 220B, Params: 8B, ps=4
"""

8B model
-----

seq_len = 8192
layers_e = 1
n_heads_e = 20
hidden_state_e = 1280
layers_g = 32
n_heads_g = 32
hidden_state_g = 4096
layers_d = 6
n_heads_d = 20
hidden_state_d = 1280
ratio_patchdim2bytedim = 4
window_e = 512
window_d = 512

"""

"""
patch_size=[1, 2, 4, 6, 8]
tokens = [1e10, 2e10, 4e10, 6e10]
# BLT optima: 64 encoder/decoder dimension: encoder/decoder heads | 128 global dimension: global heads
decoder_layers = [5, 6, 7, 8, 9, 10]
n_heads_e = [12, 14, 16, 18, 20]
global_layers = [24, 26, 28, 30, 32, 34, 36, 38, 40]
n_heads_g = [10, 16, 20, 24, 32, 40]
"""

FLOP_estimates = [4e18, 8e18, 2e19, 4e19]
patch_size = [1, 2, 4, 6]
#! tokens = [6.66e9, 1e10, 1.33e10]
# BLT optima: 64 encoder/decoder dimension: encoder/decoder heads | 128 global dimension: global heads
decoder_layers = [5, 6, 7, 8, 9]
n_heads_e = [2, 4, 6, 8, 10, 12, 14, 16]
global_layers = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
n_heads_g = [2, 4, 6, 8, 10, 12, 14, 16]

permutation_count = 0
values = []


for p in patch_size:
    for dl in decoder_layers:
        for gl in global_layers:
            for eh in n_heads_e:
                for gh in n_heads_g:
                    if (
                        total_parameters(1, eh * 64) / total_parameters(gl, gh * 128)
                        <= 0.01
                    ):
                        forward_flops = forward_blt_flops(
                            8192,
                            patch_size=p,
                            hidden_state_g=gh * 128,
                            layers_g=gl,
                            hidden_state_e=eh * 64,
                            layers_e=1,
                            window_e=512,
                            hidden_state_d=eh * 64,
                            layers_d=dl,
                            window_d=512,
                            ratio_patchdim2bytedim=2,
                            n_heads_e=eh,
                            n_heads_g=gh,
                            n_heads_d=eh,
                            vocab=256,
                            feed_forward_mult=4,
                        )

                        tokens_s = 4e18 / (3 * forward_flops)
                        tokens_m = 8e18 / (3 * forward_flops)
                        tokens_l = 2e19 / (3 * forward_flops)
                        tokens_xl = 4e19 / (3 * forward_flops)

                        if (
                            tokens_s >= 4e9 and tokens_s <= 7e10
                        ):  # these values were obtained by taking the compute budget and dividing by 6N for tranformer++ on Evo paper, where N is between 20M and 400M
                            values.append(
                                (
                                    "s",
                                    p,
                                    tokens_s,
                                    total_parameters(1, eh * 64),
                                    total_parameters(gl, gh * 128),
                                    total_parameters(dl, eh * 64),
                                    gh * 128,
                                    eh * 64,
                                    gl,
                                    dl,
                                )
                            )

                        if (
                            tokens_m >= 6.66e9 and tokens_m <= 1.33e11
                        ):  # these values were obtained by taking the compute budget and dividing by 6N for tranformer++ on Evo paper, where N is between 5M and 200M
                            values.append(
                                (
                                    "m",
                                    p,
                                    tokens_m,
                                    total_parameters(1, eh * 64),
                                    total_parameters(gl, gh * 128),
                                    total_parameters(dl, eh * 64),
                                    gh * 128,
                                    eh * 64,
                                    gl,
                                    dl,
                                )
                            )

                        if tokens_l >= 1.3e10 and tokens_l <= 6.66e11:
                            values.append(
                                (
                                    "l",
                                    p,
                                    tokens_l,
                                    total_parameters(1, eh * 64),
                                    total_parameters(gl, gh * 128),
                                    total_parameters(dl, eh * 64),
                                    gh * 128,
                                    eh * 64,
                                    gl,
                                    dl,
                                )
                            )

                        if tokens_xl >= 1.3e10 and tokens_xl <= 6.66e11:
                            values.append(
                                (
                                    "xl",
                                    p,
                                    tokens_xl,
                                    total_parameters(1, eh * 64),
                                    total_parameters(gl, gh * 128),
                                    total_parameters(dl, eh * 64),
                                    gh * 128,
                                    eh * 64,
                                    gl,
                                    dl,
                                )
                            )


import pandas as pd

df = pd.DataFrame.from_records(
    values,
    columns=[
        "FLOPs",
        "Patch size",
        "Tokens",
        "Encoder parameters",
        "Global transformer parameters",
        "Decoder parameters",
        "Global transformer dimension",
        "Encoder/Decoder dimension",
        "Global transformer layers",
        "Decoder layers",
    ],
)

df.to_csv("Compute allocations.csv", index=False)

# import plotly.express as px
# fig = px.scatter(estimates)
# fig.show()
# print(estimates)

