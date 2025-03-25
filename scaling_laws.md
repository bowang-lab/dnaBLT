# FLOP Calculations for Striped Hyena

Based on the paper, they say that the Striped Hyena FLOP cost is the sum of FLOPS of Hyena-GLU and MHA-GLU. Now we will compute the flop values of these 2 first and sum them up based on the mixing ratios for the Striped Hyena. 

Recall: $\text{FLOPS}_{\text{StripedHyena}} = \lambda\;\text{FLOPS}_{\text{Hyena-GLU}} + (1-\lambda)\;\text{FLOPS}_{\text{MHA-GLU}}$, where $0 \leq \lambda \leq 1$

## MHA GLU FLOP Calculations

- **Embedding layers:**  
  $$4LDV$$

- **MHA projections:**  
  $$6LD^2$$

- **MHA attention:**  
  Rather than computing over a full $L \times L$ interaction matrix, we now assume that each layer’s attention is computed over a fixed context. The updated attention FLOP cost is: \\

  $4 \times \text{(layers)} \times D \times \frac{(L+1)}{2} = 2 \times \text{(layers)} \times D \times (L+1).$

- **MHA out layer:**  
  $$2LD^2$$

- **GLU:**  
  $$6LD D_{\text{glu}}$$

*Summing the MHA‐GLU parts gives:*

$$
\text{FLOPS}_{\text{MHA-GLU}} = 4LDV + 6LD\,D_{\text{glu}} + (6LD^2 + 2LD^2) + 2 \times \text{(layers)} \times D \times (L+1)
$$

or, after combining the \(D^2\) terms:

$$
\text{FLOPS}_{\text{MHA-GLU}} = 4LDV + 6LD\,D_{\text{glu}} + 8LD^2 + 2 \times \text{(layers)} \times D \times (L+1).
$$

In the paper, they mention the MHA-GLU FLOPs calculations come from the Transformer++ section in the paper.

## Hyena-GLU

Here the embedding and GLU parts are the same as for Transformer++ and the “Sequence Mixer” adds extra FLOPs.

- **Embedding + GLU (same as above):**  
  $$4LDV + 6LD D_{\text{glu}}$$

- **Sequence Mixer – projections:**  
  $$6LD^2$$

- **Sequence Mixer – convs on projections:**  
  $$18LD$$

- **Sequence Mixer – featurization:**  
  $$S_{hyena} LD^9$$  

- **Sequence Mixer – convolution & gates:**  
  $$10L \log_2(L) D + 4LD$$

- **Sequence Mixer – out layer:**  
  $$2LD^2$$

*Combining the Sequence Mixer terms:*

- The \(D^2\) parts:  
  $$6LD^2 + 2LD^2 = 8LD^2.$$

- The \(LD\) parts:  
  $$18LD + 4LD + 10L\log_2(L)D = 22LD + 10L\log_2(L)D.$$

Thus, the total for Hyena‐GLU is:

$$
\text{FLOPS}_{\text{Hyena-GLU}} = 4LDV + 6LD D_{\text{glu}} + 8LD^2 + \bigl(22LD + 10L\log_2(L)D\bigr) + \text{Shyena} LD^9.
$$

## Final Calculation

Using the mixing ratio $\lambda$ for the Hyena branch (and $1-\lambda$ for the MHA branch), we obtain:

$$
\begin{aligned}
\text{FLOPS}_{\text{StripedHyena}} = \; & \lambda\Bigl[4LDV + 6LD D_{\text{glu}} + 8LD^2 + 22LD + 10L\log_2(L)D + \text{Shyena} LD^9\Bigr] \\
& + (1-\lambda)\Bigl[4LDV + 6LD\,D_{\text{glu}} + 8LD^2 + 2 \times \text{(layers)} \times D \times (L+1)\Bigr].
\end{aligned}
$$


# Comparison of Full Attention vs. Causal (Block) Attention FLOP Counts

Below is an explanation that compares the FLOP counts for full attention versus the BLT-style causal (block) attention, along with an explanation of how these counts are derived.

---

## Full Attention FLOP Count: `4L²H`

- **Full Interaction:**  
  In full attention, every token in the sequence attends to every other token. For a sequence of length `L`, this results in an interaction matrix with `L × L = L²` elements.

- **Operation Breakdown:**  
  - **Dot-Product Computation:**  
    The term `4L²H` comes from computing the dot products between queries and keys. The factor `4` is used to account for the arithmetic operations (multiplications and additions) required for each interaction.
  - **Softmax Overhead:**  
    An additional cost (often represented as `2L²H`) is associated with the softmax normalization of these dot products. This overhead includes the operations for exponentiating the scores, summing them, and normalizing—all done over the full `L × L` matrix.
  
- **Key Point:**  
  Since every query token interacts with every key token, the cost scales quadratically with the sequence length (`L²`).

---

## Causal (Block) Attention FLOP Count in BLT: Factor of `(m+1)/2`

- **Notation in BLT:**  
  - `l`: Number of layers  
  - `h`: Hidden dimension  
  - `hk`: Head dimension (with `nheads` heads, so that `h = hk × nheads`)  
  - `m`: Context length (or fixed window size)

- **Causal Attention Mechanism:**  
  In causal (or autoregressive) attention—common in decoder-only architectures like GPT-3—each token at position `i` attends only to tokens `1` through `i`. This forms a triangular (or lower triangular) attention matrix.

- **Deriving the `(m+1)/2` Factor:**  
  - For a block of `m` tokens, the first token attends to 1 token, the second to 2 tokens, and so on, up to the `m`th token which attends to `m` tokens.
  - The total number of attention interactions for the block is:
    $1 + 2 + \dots + m = \frac{m(m+1)}{2}.$
  - **Averaging per Token:**  
    Dividing the total interactions by `m` (the number of tokens) gives an average of:
    $\frac{m(m+1)/2}{m} = \frac{m+1}{2}.$
  - This average value reflects that, on average, each token attends to roughly \(\frac{m+1}{2}\) tokens rather than all `m` tokens. That’s why in the BLT computation, the attention cost is reduced by this factor compared to a full \(m \times m\) interaction.

---

## Decoder-Only Architectures and Causal Attention

- **Causal Nature in Models like GPT-3:**  
  In decoder-only architectures such as GPT-3, every self-attention layer is causal. This means:
  - Each token can only attend to itself and tokens that come before it.
  - This ensures the autoregressive property, so the model does not "peek" into future tokens during training or generation.

---

This explanation highlights why full attention has a `4L²H` cost (plus softmax overhead) due to the complete \(L × L\) interactions, whereas causal attention in BLT uses the $\frac{(m+1)}{2}$ factor to reflect that each token, on average, attends to only a subset of tokens (its past), ensuring efficiency in decoder-only models like GPT-3.
