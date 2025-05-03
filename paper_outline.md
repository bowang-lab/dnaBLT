# Paper Outline

## Abstract

150–200 words -- talk about novelty in architecture, overview of performance on the different evals

## Introduction

- Why byte-level DNA LM matters
- Shortcomings of Transformers for 1 M-token windows
- Promise of hybrid/latent architectures
- Our contributions (bullet list)

## Related Works

## Byte-Latent StripedHyena2 (BLSH2)

- General talking points about how we designed the architecture
- Any additional design decisions that we may have made (any significant changes to iterators, RoPE, etc?)

## Scaling Laws Section

### FLOPs Calculations

- Use `Overview.md` calculations and put it into this section. 

### Setup 

- Basic set up for FLOP experimenets

### Scaling‑law analysis (four FLOP budgets, varied token : parameter ratios)

- reference `experiments.md`

#### Compute‑optimal ridge

- For each size we train at the Chinchilla‑style optimum. Plot log‑PPL vs log‑FLOPs

#### Off‑optimal robustness

- Re‑run the smallest and mid‑scale models at 10 × and 40 × tokens / param.

#### IsoFLOP frontiers

- At each budget, overlay PPL vs parameter count for all sizes. Identify break‑even where BLSH‑2 matches Evo with fewer params

### Latency Experiments

- We will measure the latency per generation step for long sequences. Since BLT processes in two stages (local byte encoder/decoder and global latent model), we’ll measure the overall end-to-end latency to predict the next token for a given context. StripedHyena2 layers operate in parallel across sequence positions (like convolution), so the new model can leverage GPU parallelism better than the sequential attention mechanism.



