# Experiments



## Evaluation Metrics and Protocol

We will use both **quality metrics** and **efficiency metrics** to evaluate the models:

* **Token-level Perplexity (PPL):** Our primary quality metric is perplexity on a held-out test set of genome sequences. We compute cross-entropy loss on the next-nucleotide prediction and report perplexity = exp(loss). Lower perplexity indicates better predictive modeling of genomic sequences. We will report average perplexity across the test set for each model/experiment, as is standard in language model evaluation. Perplexity will be measured for full-length sequences (e.g. contiguous genome segments) to capture long-range dependencies. We will particularly highlight perplexity improvements of BLT+Hyena2 over Evo (baseline) at equal model size and training budget.

* **Inference Latency:** We measure the time taken to produce outputs, specifically **per-token latency** (e.g. milliseconds per generated nucleotide) and **end-to-end sequence latency** for processing a full context. Latency is measured on a standardized hardware setup with a fixed batch size. We will compare latency as a function of sequence length, expecting StripedHyena2 to have significantly lower growth in latency with context length due to its sub-quadratic time complexity (https://arxiv.org/pdf/2306.15794).



All metrics will be averaged over multiple runs or seeds where appropriate to ensure statistical significance. We will use standard deviations or confidence intervals to report variability. Where relevant, we will follow evaluation protocols from prior genomic language model papers (e.g. using the **Nucleotide Transformer** benchmark suite for regulatory element prediction, as in HyenaDNA).


## Scaling Experiments (4 FLOP Levels, Varied Token\:Parameter Ratios)

**Experiment 1: Scaling Law Analysis.** We will conduct a scaling study across **four different compute budgets (FLOP levels)** to characterize how the new architecture’s performance scales with model size and data, in line with mechanistic scaling law approaches. At each budget, we will train a model from scratch and measure its achieved perplexity. The four budgets will range from **small-scale** to **large-scale**. The **token\:parameter ratio** (amount of training data per model parameter) will be systematically varied across these runs:

* **Compute-Optimal Training:** For one run at each model size, we will allocate training tokens following the compute-optimal paradigm (roughly the Chinchilla strategy). 

* **Undertrained vs Overtrained Regimes:** At two of the four scales (e.g. the smallest and one mid-sized model), we will also train models with **off-optimal** token counts to probe scaling behavior. Specifically, we will **overtrain** a smaller model (providing substantially more tokens than compute-optimal for that size) and **undertrain** a larger model (fewer tokens than optimal) while keeping total compute similar. This yields different token\:parameter ratios. We will measure the **perplexity gap** when training off the optimal frontier, as in recent analyses. According to mechanistic scaling laws, hybrid models often have **flatter** perplexity curves off-optimal (smaller loss increase when overtrained) compared to Transformers. We will verify if BLT+Hyena2 exhibits a smaller optimality gap, indicating it is more robust to overtraining or undertraining than the baseline.

* **Scaling Curves:** For each architecture (BLT+Hyena2 and baseline Evo/Transformer), we will plot **log-perplexity vs log-model-size** and fit power-law curves. We expect to find a lower perplexity for the new model at equal size, and possibly a different scaling exponent. If StripedHyena2 improves efficiency, the new model may attain the same perplexity as a baseline model with significantly fewer parameters or FLOPs. We will also identify the **“break-even” point** where the new architecture overtakes the baseline (e.g. perhaps even a smaller BLT+Hyena2 might match a larger Evo model’s perplexity).

We probe **compute-optimal, under-trained, and over-trained runs at four FLOP budgets** because a single “best-tuned” curve hides two facts reviewers care about: (1) **true efficiency**—where each architecture sits on the loss-vs-compute frontier—and (2) **robustness to real-world mis-allocation of tokens.** By sandwiching every compute point with a run that gets too few tokens and one that gets too many, we map the full envelope of achievable perplexities. If BLT + StripedHyena-2 stays close to its optimal line while Evo’s perplexity rises sharply off-ridge, we demonstrate that the new hybrid is not only *more compute-efficient* at the sweet spot but also *more forgiving* when practitioners inevitably over-train or under-train.
