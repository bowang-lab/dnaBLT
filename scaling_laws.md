# FLOP Calculations for Striped Hyena

Based on the paper, they say that the Striped Hyena FLOP cost is the sum of FLOPS of Hyena-GLU and MHA-GLU. Now we will compute the flop values of these 2 first and sum them up based on the mixing ratios for the Striped Hyena. 

Recall: $\text{FLOPS}_{\text{StripedHyena}} = \lambda\;\text{FLOPS}_{\text{Hyena-GLU}} + (1-\lambda)\;\text{FLOPS}_{\text{MHA-GLU}}$, where $0 \leq \lambda \leq 1$

## MHA GLU FLOP Calculations

- **MHA projections:**  
  $$6LD^2$$

- **MHA attention:**  
  Rather than computing over a full $L \times L$ interaction matrix, we now assume that each layer’s attention is computed over a fixed context. The updated attention FLOP cost is:

  $$4 \times \text{(layers)} \times D \times \frac{(L+1)}{2} = 2 \times \text{(layers)} \times D \times (L+1).$$

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
\text{FLOPS}_{\text{MHA-GLU}} =  6LD\,D_{\text{glu}} + 8LD^2 + 2 \times \text{(layers)} \times D \times (L+1).
$$

In the paper, they mention the MHA-GLU FLOPs calculations come from the Transformer++ section in the paper.

## Hyena-GLU

Here the embedding and GLU parts are the same as for Transformer++ and the “Sequence Mixer” adds extra FLOPs.

- **GLU (same as above):**  
  $$ 6LD D_{\text{glu}}$$

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

- The $D^2$ parts:  
  $$6LD^2 + 2LD^2 = 8LD^2.$$

- The $LD$ parts:  
  $$18LD + 4LD + 10L\log_2(L)D = 22LD + 10L\log_2(L)D.$$

Thus, the total for Hyena‐GLU is:

$$
\text{FLOPS}_{\text{Hyena-GLU}} = 6LD D_{\text{glu}} + 8LD^2 + \bigl(22LD + 10L\log_2(L)D\bigr) + \text{Shyena} LD^9.
$$

## Final Calculation

Using the mixing ratio $\lambda$ for the Hyena branch (and $1-\lambda$ for the MHA branch), we obtain:

$$
\begin{aligned}
\text{FLOPS}_{\text{StripedHyena}} = \; & \lambda\Bigl[6LD D_{\text{glu}} + 8LD^2 + \bigl(22LD + 10L\log_2(L)D\bigr) + \text{Shyena} LD^9\Bigr] \\
& + (1-\lambda)\Bigl[6LD\,D_{\text{glu}} + 8LD^2 + 2 \times \text{(layers)} \times D \times (L+1)\Bigr].
\end{aligned}
$$


# Causal Masking and Layer-wise Operations

## Causal Masking

In transformer architectures—especially in decoder-only models like GPT-3—causal masking is used to enforce the autoregressive property. This means that for any given token at position \(i\) in a sequence, the attention mechanism is restricted so that the token only attends to itself and tokens preceding it (positions $1$ through $i$). This results in a lower triangular attention matrix rather than a full $L \times L$ matrix.

When using a fixed context window of size $m$, the number of attention interactions per token is not $m$ (or $m^2$ for a full block), but instead is averaged to $\frac{m+1}{2}$. For instance, the first token attends to 1 token, the second to 2 tokens, and so on, which sums to $\frac{m(m+1)}{2}$ interactions across the block. Dividing by $m$ gives an average of $\frac{m+1}{2}$ interactions per token. This reduction in interactions directly lowers the FLOP count compared to full attention.

## Layer-wise vs. One-Time Operations

The FLOP counts in the calculations above represent operations that are performed **per layer** of the model. These include:

- **MHA Projections:**  
  Operations to project the input into queries, keys, and values.
- **MHA Attention (with Causal Masking):**  
  The dot-product calculations and softmax normalization, computed over a fixed context window, adjusted by the $\frac{(L+1)}{2}$ factor.
- **MHA Output Layer:**  
  The computations for combiningi the attenton outputs.
- **GLU Computations:**  
  The operations involved in the gated linear unit.
- **Sequence Mixer Operations (for Hyena-GLU):**  
  This includes projections, convolution operations, featurization, and additional convolution and gating steps.

These operations are repeated for each layer of the transformer model. By summing the FLOP counts across all layers, we obtain the total computational cost for the model's forward pass.


