# Adaptive Low-Rank Residual Allocation for Quantized Language Models Under Fixed Memory Budgets

## Abstract

Large language models are typically compressed with quantization because it offers strong memory savings, practical inference speedups, and mature deployment support. Low-rank decomposition is a second major compression family, but as a standalone method it often struggles to match the practicality of low-bit quantization. A more promising direction is hybrid compression, where quantization handles most of the memory reduction and low-rank structure is used selectively to repair the most damaging errors. This proposal studies a specific version of that idea: given a fixed memory budget, can a small set of targeted low-rank residual corrections recover quantization damage more effectively than spending the same budget on uniformly increasing quantization precision? The project will develop and evaluate a budget-aware layer allocation strategy that decides where low-rank correction should be added, how much rank each corrected layer should receive, and when low-rank correction is preferable to extra bits. The expected contribution is both an algorithmic method and an empirical characterization of where hybrid compression helps most in language models.

## Problem Statement

Current LLM compression practice is dominated by quantization, especially post-training quantization, because it is simple, scalable, and well supported in inference frameworks. However, aggressive quantization introduces uneven error across layers. Some layers degrade little under low precision, while others become bottlenecks for model quality. Existing hybrid work suggests that low-rank corrections can reduce quantization error, but the main open question is allocation: under a strict memory budget, which layers should receive low-rank correction, what rank should they receive, and when is that allocation better than simply increasing the quantization precision of the entire model or a subset of layers?

This project addresses that allocation problem directly. While prior work such as QuIP#, SqueezeLLM, and LoftQ has explored combining low-rank and quantized representations, these methods typically do not frame the combination as a constrained budgeting problem or directly compare spending extra memory on rank versus bits in a controlled setting. This project fills that gap.

## Related Work

Post-training quantization methods such as GPTQ, AWQ, and RTN each produce different error profiles: GPTQ uses approximate second-order information to minimize layerwise reconstruction error, AWQ preserves salient weight channels based on activation magnitudes, and RTN applies simple round-to-nearest without calibration. The choice of quantization method strongly affects the structure of the residual error and therefore how well low-rank correction can recover it.

On the hybrid side, SqueezeLLM combines dense-and-sparse quantization to handle sensitive weights differently, QuIP# uses incoherence processing to make quantization error more uniform, and LoftQ initializes LoRA-style low-rank factors from the quantization residual for downstream fine-tuning. These methods demonstrate that quantization and low-rank structure can be combined, but they generally do not study the allocation question under a strict fixed-memory constraint.

The present work is positioned between these two lines: it takes quantization as given and asks how best to spend a marginal memory increase, treating the choice between more bits and more rank as the core optimization variable.

## Core Research Question

For a fixed total model memory budget, is it better to spend additional capacity on:

- higher quantization precision,
- low-rank residual correction,
- or a learned combination of both?

## Hypothesis

Quantization damage is layer-dependent and partially structured. Because of that, a small number of carefully allocated low-rank residual corrections will recover more model quality per parameter than uniformly increasing precision across all layers. This benefit should be strongest in aggressive compression regimes such as 3-bit to 4-bit weight quantization, where quantization error is large enough to matter but still structured enough to be corrected.

## Research Objectives

1. Measure how quantization error is distributed across transformer layers and projection matrices.
2. Develop a budget-aware policy for selecting layers for low-rank correction.
3. Compare several rank allocation strategies under equal memory budgets.
4. Determine whether targeted low-rank correction is a better use of memory than modestly increasing quantization precision.
5. Produce practical guidance for hybrid compression design.

## Novelty Claim

The novelty is not simply combining quantization with low-rank correction. The contribution is a principled allocation framework that treats extra memory as a scarce resource and asks how it should be spent. The project reframes hybrid compression as a budgeting problem rather than a binary method comparison.

## Proposed Method

Start from a pretrained language model and produce a quantized baseline using post-training quantization. Then add low-rank residual matrices only to selected layers:

`W ≈ Q(W) + A_i B_i`

where `Q(W)` is the quantized weight matrix for layer `i`, and `A_i B_i` is a low-rank correction term with rank `r_i`.

### Choice of Quantization Method

The quantization method affects the structure of the residual error and therefore how correctable it is. This study will explore two paths:

- **Primary path:** fix a single well-established method (e.g., GPTQ) for all core experiments, to control for quantization-side variation and isolate the allocation question.
- **Secondary path:** treat the quantization method itself as an experimental variable, comparing at least GPTQ and RTN, to determine whether the optimal allocation strategy is method-dependent.

The decision on which path receives more emphasis will be made after initial profiling of the residual error structure under each method. Both paths are kept open at this stage.

### Correction Approach

The low-rank correction `A_i B_i` can be obtained in two qualitatively different ways:

- **Calibration-based (SVD):** compute the residual `R_i = W_i − Q(W_i)` and take its best rank-`r_i` approximation via truncated SVD. This is cheap — no gradient computation — and serves as a strong baseline.
- **Gradient-based fitting:** treat `A_i` and `B_i` as trainable parameters and optimize them to minimize output error on a calibration set, similar to LoRA-style adaptation. This is more expensive but may capture correction structure that weight-space SVD misses.

The project will begin with SVD-based correction for rapid iteration and explore gradient-based fitting as an extended comparison. Both approaches will be evaluated; the decision of which to prioritize in analysis will depend on early results.

### Error Measurement: Weight-Space vs. Activation-Space

A critical distinction in guiding the allocation policy is whether error is measured in weight space or activation space:

- **Weight-space error** (`‖W − Q(W)‖_F`) is cheap to compute but can be a poor predictor of task-level degradation, because it ignores how activations amplify or suppress different error components.
- **Activation-space error** measures the deviation in layer outputs on calibration data (e.g., `‖Wₗx − Q(Wₗ)x‖` averaged over calibration samples). This is more expensive but is a much better proxy for downstream quality loss.

The allocation policies will therefore use activation-space error as the primary signal, with weight-space error retained as a cheap fallback. Comparing the two will itself be a useful finding.

### Allocation Policies

The main design choice is how to allocate the residual budget across layers. The proposal will compare several policies:

- **Uniform allocation:** every eligible layer receives the same small rank.
- **Error-based allocation:** layers with larger quantization reconstruction or output error receive more rank.
- **Sensitivity-based allocation:** ranks are assigned using proxies such as activation deviation, gradient-based sensitivity, or second-order approximations.
- **Greedy budget allocation:** rank is added iteratively to the layer yielding the largest quality improvement per additional parameter.
- **Mixed bit-rank allocation:** the same budget can be split between increasing bits for some layers and adding low-rank residuals to others. Because jointly optimizing bits and rank across all layers is combinatorial, this will use a greedy or heuristic search (e.g., layerwise marginal improvement) rather than full joint optimization.

### Scope: Memory-Constrained Scenarios

This work primarily targets **memory-constrained** deployment settings — scenarios where the total model footprint must fit within a hard limit (e.g., a single GPU's VRAM). Low-rank corrections add an additional matrix multiplication at inference time, and without fused kernels the wall-clock savings from smaller quantized weights could be partially offset. Inference latency will be measured where feasible, but the central question is about quality per byte, not quality per FLOP.

## Experimental Design

### Models

Use a small-to-medium model ladder for iteration and scaling:

- one small model (~100M–350M parameters) for fast ablations,
- one medium model (~1B parameters) for stronger validation,
- one larger model (3B–7B) for limited final verification if resources allow.

### Compression Settings

Evaluate multiple compression regimes:

- 4-bit weight-only quantization,
- 3-bit weight-only quantization,
- optionally an 8-bit or mixed-precision reference.

These settings create distinct error regimes and help determine where low-rank correction becomes worthwhile.

### Baselines

Compare against:

- original full-precision model,
- pure quantization (at the base bit-width),
- pure low-rank compression (no quantization),
- naive hybrid with uniform rank allocation,
- proposed adaptive hybrid allocation.

All comparisons are at **equal total memory**, which is the defining constraint of this study.

### Calibration / Adaptation

Use a calibration dataset (e.g., a subset of C4 or WikiText) to estimate quantization error and fit low-rank corrections. The project will begin with post-training low-rank fitting (SVD-based) and extend to lightweight gradient-based fitting as a second stage. The choice between these will be informed by early results rather than fixed in advance.

### Evaluation Metrics

Measure:

- perplexity on held-out text,
- zero-shot or few-shot downstream accuracy on a compact task suite,
- memory footprint (total bytes for all model parameters),
- parameter count added by correction terms,
- inference latency where kernels permit meaningful comparison,
- quality recovered per added megabyte (the primary efficiency metric).

The key metric is not raw accuracy alone, but **quality at equal total memory**.

## Key Analyses

The proposal is strongest if it does more than show one average improvement number. It should answer:

- Which layer types benefit most from correction (attention projections vs. MLP)?
- Does attention or MLP dominate the recoverable quantization error?
- How much rank is enough before returns diminish?
- At what bit-width does low-rank correction become more useful than adding one extra bit?
- Are cheap error proxies (weight-space norms) sufficient for allocation, or do activation-space metrics materially improve allocation quality?
- Does the optimal allocation strategy depend on the quantization method, or is it robust across methods?

## Expected Contributions

1. A budget-aware hybrid compression framework for quantized language models.
2. An empirical map of layerwise quantization sensitivity and correction payoff.
3. A comparison of memory spending strategies: more bits versus more rank.
4. A characterization of how quantization method affects residual correctability.
5. Practical heuristics for selecting correction layers without expensive retraining.
6. A reproducible benchmark setup for fixed-budget compression studies.

## Success Criteria

The project will be successful if it demonstrates at least one of the following:

- targeted low-rank correction consistently outperforms uniform correction at equal memory,
- targeted correction outperforms spending the same budget on higher-bit quantization,
- simple layerwise error metrics predict correction value well enough to be useful in practice.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Quantization error may not be low-rank enough to correct efficiently. | Early residual profiling (singular value decay analysis) will reveal this before heavy experimentation. Even a negative result is publishable guidance. |
| Correction may improve perplexity without meaningful downstream gains. | Include zero-shot task evaluation alongside perplexity from the start. |
| Adaptive allocation may be too expensive relative to its quality improvement. | Track compute cost of each allocation policy and report cost–quality tradeoff explicitly. |
| Inference complexity may offset memory benefits. | Scope the contribution to memory-constrained settings; measure latency where feasible but do not claim latency wins without evidence. |
| Quantization method choice may confound allocation results. | Profile residual structure under multiple methods early; present results conditioned on method where relevant. |

## Timeline

1. **Weeks 1–3:** Set up quantization baselines (GPTQ, RTN), build the fixed-budget evaluation framework, and profile layerwise residual error structure.
2. **Weeks 4–5:** Implement SVD-based low-rank residual correction and uniform-rank hybrid baselines.
3. **Weeks 6–7:** Develop and test error-based and sensitivity-based allocation policies (both weight-space and activation-space).
4. **Weeks 8–9:** Run full comparisons under equal memory budgets; begin gradient-based correction experiments.
5. **Weeks 10–11:** Analyze layer patterns, ablations, and tradeoffs. Explore mixed bit-rank allocation.
6. **Weeks 12–14:** Write results and package reproducible code.

## Deliverables

- A research report or paper draft.
- Code for fixed-budget hybrid compression experiments.
- Plots showing quality-memory tradeoffs across allocation strategies.
- Ablations on allocation strategies, layer sensitivity, and correction method.
- Layerwise residual structure profiles under different quantization methods.
- Recommendations for when to spend memory on bits versus rank.

## One-Sentence Summary

This project asks a sharper and more original question than "SVD vs. quantization": under a fixed memory budget, where should extra capacity go in a quantized LLM, and can targeted low-rank residuals recover more quality than simply using more bits?
