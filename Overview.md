Use this space to obtain clarity on how to proceed.

- [x] Compute entropies over tokens
- [ ] Generate experiment set for Byte Latent StripedHyena2 
    - [x] Get FLOPs per token 
    - [ ] Sample uniformly from isoflop parabolas on fixed FLOP budgets
- [ ] Build training loop with Byte Latent StripedHyena2 architecture
- [ ] Find stable HP optima for various model sizes
- [ ] Run experiments

## FLOPs per token

We'll want to use the same high-level structure as the Byte Latent Transformer, but with a StripedHyena2 model in-place of the transformer.

The architecture consists of interleaved Hyena-GLU and MHA-GLU layers. Evo1 uses a mixing ratio of roughly 10%. Hybrid architecture scaling laws paper observes stable optima centred around that region. We oddly have different FLOPs values for the same operations found across various papers. Let’s consolidate the differences.

### Why does the Hybrid paper’s FLOPs differ to those of the Chinchilla paper?

Chinchilla paper

1. KQV projections: $6 \times \text{seq\_len} \times \text{d\_model}^2$
2. Key @ Query logits: $2 \times \text{seq\_len}^2 \times \text{d\_model}$
3. Softmax: $3 \times \text{num\_heads} \times \text{seq\_len}^2$
4. Softmax @ query reductions: $2 \times \text{seq\_len}^2 \times {d\_model}$
5. Final linear: $2 \times \text{seq\_len} \times \text{d\_model}^2$

Dense block (instead of GLU): $4 \times \text{seq\_len} \times \text{d\_model} \times \text{ffw\_size}$

Hybrid scaling laws paper

Transformer++
- MHA
    - projections: $6LD^2$
    - attention: $4L^2D + 2HL^2$
    - out layer: $2LD^2$
- GLU (instead of dense): $6LDD_{\text{glu}}$

If you sum the various Chinchilla MHA operations, they turn out to be equivalent to the Hybrid ones! Except the dense block has different FLOPs to the GLU, let’s unpack that.

Dense: $XW_1 \rightarrow \sigma(\cdot) \rightarrow (\cdot)W_2$

Since $W_1$ is of shape $(D, D_{ff})$, the cost of this operation is $2LDD_{ff}$. Likewise, since $W_2$ is of shape $(D_{ff}, D)$, it has the same FLOP cost. So summing these operations, we get $4LDD_{ff}$. 

SwiGLU: $(XW_1) \rightarrow \text{SwiGLU}(\cdot) \rightarrow (\cdot)W_2$

This function is a little weirder; first we project from $D$ to $2D_{ff}$ to split the result into two halves. So the FLOPs are instantly $4LDD_{ff}$. After undergoing the SwiGLU activation (splitting into two halves, inserting the latter into swish, and then doing elementwise multiplication with the former) there's a trivial addition of FLOPs. But, the result of the gating $(L, D_{ff})$ is then multiplied by $W_2$ of shape $(D_{ff}, D)$ which is the normal linear layer $2LDD_{ff}$. So summing the two major operations, we get $6LDD_{ff}$.

So that covers Hyena vs Chinchilla. In conclusion, everything in the former paper is correct and we simply multiply the Hyena-GLU by the number of layers. Now can we consolidate the differences between the Hyena-GLU/MHA-GLU with the BLT formulation? 

### BLT formulation vs Chinchilla formulation

The differences comes from the fact that the Chinchilla formulation is computing FLOPs per sequence whereas BLT is per token.

*Note*: The total forward pass FLOPs in the Chinchilla paper is computed as $\text{embeddings}+\text{\text{num\_layers}}\times (\text{total\_attention} + \text{dense\_block}) + \text{logits}$. The aforementioned MHA-Dense block is just including the FLOPs for a single transformer layer, whereas the Kaplan/BLT formulation is multiplying by $\text{num\_layers}$ for every operation, effectively giving you the total FLOPs for the transformer forward pass.

Formula to derive FLOPs per token (BLT format) from FLOPs per sequence (Chinchilla format): $\small\text{BLT\_FLOPs}=\frac{\text{chinchilla\_FLOPs}}{\text{seq\_len}}\cdot \text{num\_layers}$

### Extending StripedHyena FLOPs in Hybrid formulation to BLT/Kaplan formulation

Since Hyena FLOPs are computed in the same way as Chinchilla and Evo uses the same parameter scaling regime as the Hybrid scaling law authors (short filter order of 3, filter order of 2):

- Hyena operator:
    -  Projections: $6LD^2$
    -  Convs on projections: $18LD$
    -  Featurization: $2LD$
    -  Convolution and gates: $10L\log_2(L)D+4LD$
    -  Out layer: $2LD^2$

- GLU: $6LDD_{glu}$

Using the derived sequence-to-token FLOPs converter,

- Hyena operator:
    -  Projections: $6 \times h^2 \times l$
    -  Convs on projections: $18 \times h \times l$
    -  Featurization: $2 \times h \times l$
    -  Convolution and gates: $l \times (10 \times \log_2(m) \times h + 4 \times h)$
    -  Out layer: $2 \times h^2 \times l$

*(or as a shorthand with 99% lowerbound accuracy: $2.68 \times 3 \times h \times l \times (\log_2(m) + h)$)*

- GLU: $6 \times l \times h \times d_{ff}h$

## Sample uniformly from isloflop parabolas

How should we choose the initial range of our set of scaling experiments?

The Evo paper shows a parabola of values they assessed, with the optima being located roughly in the middle. Naively, if we simply wanted to reproduce those scaling results, we'd just need to assess those exact parameter vs token allocations.

What does the byte latent formulation change? Larger patches decrease the number of tokens. Given a fixed compute budget, if we ran any of their experiments using the same model size, we'd just have vastly more tokens than they trained on.

So let's compare runs with model sizes (compared to their experiments) [smallest_size, largest_size * patch_size] (it has been determined that proportionally increasing params leads to proportional increases in FLOPs). This way we'll be able to sample across having patch_size x max_tokens, to min_tokens.