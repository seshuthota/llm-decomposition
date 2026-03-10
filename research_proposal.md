# Adaptive Low-Rank Residual Allocation for Quantized Language Models Under Fixed Memory Budgets

## Abstract

Large language models are typically compressed with quantization because it offers strong memory savings, practical inference speedups, and mature deployment support. Low-rank decomposition is a second major compression family, but as a standalone method it often struggles to match the practicality of low-bit quantization. A more promising direction is hybrid compression, where quantization handles most of the memory reduction and low-rank structure is used selectively to repair the most damaging errors. This proposal studies a specific version of that idea: given a fixed memory budget, can a small set of targeted low-rank residual corrections recover quantization damage more effectively than spending the same budget on uniformly increasing quantization precision? The project will develop and evaluate a budget-aware layer allocation strategy that decides where low-rank correction should be added, how much rank each corrected layer should receive, and when low-rank correction is preferable to extra bits. The expected contribution is both an algorithmic method and an empirical characterization of where hybrid compression helps most in language models.

## Problem Statement

Current LLM compression practice is dominated by quantization, especially post-training quantization, because it is simple, scalable, and well supported in inference frameworks. However, aggressive quantization introduces uneven error across layers. Some layers degrade little under low precision, while others become bottlenecks for model quality. Existing hybrid work suggests that low-rank corrections can reduce quantization error, but the main open question is allocation: under a strict memory budget, which layers should receive low-rank correction, what rank should they receive, and when is that allocation better than simply increasing the quantization precision of the entire model or a subset of layers?

This project addresses that allocation problem directly.

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

Start from a pretrained language model and produce a quantized baseline using a standard post-training quantization pipeline. Then add trainable or calibrated low-rank residual matrices only to selected layers:

`W ~= Q(W) + A_i B_i`

where `Q(W)` is the quantized weight matrix for layer `i`, and `A_i B_i` is a low-rank correction term with rank `r_i`.

The main design choice is how to allocate the residual budget across layers. The proposal will compare several policies:

- Uniform allocation: every eligible layer receives the same small rank.
- Error-based allocation: layers with larger quantization reconstruction or output error receive more rank.
- Sensitivity-based allocation: ranks are assigned using proxies such as activation deviation, gradient-based sensitivity, or second-order approximations.
- Greedy budget allocation: rank is added iteratively to the layer yielding the largest quality improvement per additional parameter.
- Mixed bit-rank allocation: the same budget can be split between increasing bits for some layers and adding low-rank residuals to others.

## Experimental Design

### Models

Use a small-to-medium model ladder for iteration and scaling:

- one small model for fast ablations,
- one medium model for stronger validation,
- one larger model for limited final verification if resources allow.

A practical range would be approximately 100M to 1B parameters for development, with one larger checkpoint used only after the method stabilizes.

### Compression Settings

Evaluate multiple compression regimes:

- 4-bit weight-only quantization,
- 3-bit weight-only quantization,
- optionally an 8-bit or mixed-precision reference.

These settings create distinct error regimes and help determine where low-rank correction becomes worthwhile.

### Baselines

Compare against:

- original full-precision model,
- pure quantization,
- pure low-rank compression,
- naive hybrid with uniform rank,
- proposed adaptive hybrid allocation.

### Calibration / Adaptation

Use a calibration dataset to estimate quantization error and optionally fit low-rank corrections. The project can begin with post-training low-rank fitting and later test lightweight fine-tuning if needed.

### Evaluation Metrics

Measure:

- perplexity on held-out text,
- zero-shot or few-shot downstream accuracy on a compact task suite,
- memory footprint,
- parameter count added by correction terms,
- inference latency if kernels permit meaningful comparison,
- quality recovered per added megabyte.

The key metric is not raw accuracy alone, but quality at equal total memory.

## Key Analyses

The proposal is strongest if it does more than show one average improvement number. It should answer:

- Which layer types benefit most from correction?
- Does attention or MLP dominate the recoverable quantization error?
- How much rank is enough before returns diminish?
- At what bit-width does low-rank correction become more useful than adding one extra bit?
- Are cheap error proxies sufficient, or do stronger sensitivity estimates materially improve allocation?

## Expected Contributions

1. A budget-aware hybrid compression framework for quantized language models.
2. An empirical map of layerwise quantization sensitivity and correction payoff.
3. A comparison of memory spending strategies: more bits versus more rank.
4. Practical heuristics for selecting correction layers without expensive retraining.
5. A reproducible benchmark setup for fixed-budget compression studies.

## Success Criteria

The project will be successful if it demonstrates at least one of the following:

- targeted low-rank correction consistently outperforms uniform correction at equal memory,
- targeted correction outperforms spending the same budget on higher-bit quantization,
- simple layerwise error metrics predict correction value well enough to be useful in practice.

## Risks

The main risks are:

- quantization error may not be low-rank enough to correct efficiently,
- the correction may improve perplexity without meaningful downstream gains,
- adaptive allocation may be too expensive relative to its quality improvement,
- inference complexity may offset memory benefits if deployment support is weak.

These risks are manageable because even negative results would still produce useful guidance about when hybrid compression is not worth it.

## Timeline

1. Weeks 1-2: implement quantization baselines and fixed-budget evaluation setup.
2. Weeks 3-4: implement low-rank residual correction and uniform-rank hybrid baselines.
3. Weeks 5-6: develop and test error-based and sensitivity-based allocation policies.
4. Weeks 7-8: run full comparisons under equal memory budgets.
5. Weeks 9-10: analyze layer patterns, ablations, and tradeoffs.
6. Weeks 11-12: write results and package reproducible code.

## Deliverables

- a research report or paper draft,
- code for fixed-budget hybrid compression experiments,
- plots showing quality-memory tradeoffs,
- ablations on allocation strategies and layer sensitivity,
- recommendations for when to spend memory on bits versus rank.

## One-Sentence Summary

This project asks a sharper and more original question than "SVD vs quantization": under a fixed memory budget, where should extra capacity go in a quantized LLM, and can targeted low-rank residuals recover more quality than simply using more bits?
