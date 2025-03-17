# FLOP Calculations for Striped Hyena

Based on the paper, they say that the Striped Hyena FLOP cost is the sum of FLOPS of Hyena-GLU and MHA-GLU. Now we will compute the flop values of these 2 first and sum them up based on the mixing ratios for the Striped Hyena. 

Recall: $\text{FLOPS}_{\text{StripedHyena}} = \lambda\;\text{FLOPS}_{\text{Hyena-GLU}} + (1-\lambda)\;\text{FLOPS}_{\text{MHA-GLU}}$, where $0 \leq \lambda \leq 1$

## MHA GLU FLOP Calculations

- **Embedding layers:**  
  $$4LDV$$

- **MHA projections:**  
  $$6LD^2$$

- **MHA attention:**  
  $$4L \times 2D + 2H L^2$$  
  $$= 8LD + 2H L^2$$

- **MHA out layer:**  
  $$2LD^2$$

- **GLU:**  
  $$6LD D_{\text{glu}}$$

*Summing the MHA‐GLU parts gives:*

$$
\text{FLOPS}_{\text{MHA-GLU}} = 4LDV + 6LD D_{\text{glu}} + (6LD^2 + 2LD^2) + 8LD + 2HL^2
$$

or, after combining the \(D^2\) terms:

$$
\text{FLOPS}_{\text{MHA-GLU}} = 4LDV + 6LD D_{\text{glu}} + 8LD^2 + 8LD + 2HL^2.
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
& + (1-\lambda)\Bigl[4LDV + 6LD D_{\text{glu}} + 8LD^2 + 8LD + 2HL^2\Bigr].
\end{aligned}
$$
