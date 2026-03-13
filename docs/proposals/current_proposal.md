# Budget-Aware Allocation of Bits and Low-Rank Residuals in Post-Training Quantized Language Models

## Abstract

Post-training quantization is the dominant practical approach for compressing large language models because it offers substantial memory reduction, straightforward deployment, and mature inference support. However, quantization damage is not uniform across a model: some layers tolerate aggressive compression well, while others become quality bottlenecks. Low-rank residual correction is a promising complementary mechanism because quantization error is often partially structured rather than purely random. The key open question is not whether quantization and low-rank structure can be combined, but how a limited memory budget should be divided between them.

This project studies that allocation problem directly. Given a fixed total model size, it asks whether the next marginal unit of memory should be spent on higher quantization precision or on low-rank residual repair. The proposed framework treats memory as a scarce resource and allocates it adaptively across transformer layers using calibration-driven estimates of quality gain per byte. The central contribution is a post-training, budget-aware allocation method and an empirical characterization of when extra bits are preferable to extra rank, and where low-rank residuals are most effective in restoring quantization-induced damage.

## Problem Statement

The practical success of LLM compression has made quantization the default choice for memory-constrained deployment. Yet quantization error is highly uneven across layers, projection matrices, and compression regimes. A model quantized to 3 or 4 bits often contains a small subset of layers that dominate the resulting quality loss, while many others degrade only marginally.

Low-rank residual correction offers one way to selectively repair this damage by approximating the quantization residual with small trainable or analytically derived matrices. The unresolved issue is allocation. Under a strict memory budget, one must decide:

- which layers should receive additional capacity,
- whether that capacity should take the form of extra quantization bits or low-rank residual terms,
- and how much of either resource each selected layer should receive.

This project addresses that problem as a constrained optimization question. Rather than treating hybrid compression as a simple combination of two known methods, it treats every additional byte as a decision variable and asks how that byte should be spent for maximum quality recovery.

## Motivation and Positioning

This proposal is not based on the claim that combining quantization and low-rank structure is itself new. Prior work has already shown that both non-uniform bit allocation and low-rank correction can improve compressed models. Early experiments in this repo also suggest that naive uniform low-rank repair is weaker than spending the same memory on more bits once the bits-only baseline becomes meaningful. The sharper and more defensible contribution of this project is therefore:

1. it focuses on **post-training, frozen-model compression** rather than fine-tuning-oriented adaptation,
2. it frames the design choice as a **marginal memory allocation problem**,
3. it treats **bit allocation as the mainline baseline** and compares rank against it under equal total model size,
4. and it uses **calibration-driven activation-space signals** to guide that allocation.

The paper therefore asks a more specific question than traditional method comparisons. It is not merely "does hybrid compression help?" but rather:

> Under a fixed memory budget, when is it better to spend additional capacity on quantization precision, and when is it better to spend it on low-rank residual repair?

That framing is the core of the project.

## Core Research Question

For a fixed total model memory budget, how should additional capacity be allocated across transformer layers between:

- higher quantization precision,
- low-rank residual correction,
- or a combination of both,

in order to maximize model quality?

## Hypothesis

Quantization damage in language models is strongly layer-dependent and partly structured. Because of this, additional memory is not equally valuable everywhere. The working hypothesis is now two-part: first, non-uniform bit allocation will usually be the strongest first use of extra memory; second, low-rank residual correction may still provide incremental value after the most important bit-allocation decisions have already been made. This should be most relevant in aggressive regimes such as 3-bit to 4-bit weight quantization, where quantization error is large enough to matter but still structured enough to admit selective repair.

## Research Objectives

1. Measure how quantization error and task degradation are distributed across layers and matrix types in transformer models.
2. Build a calibration-driven framework that estimates the marginal quality gain per byte for candidate compression repairs.
3. Compare the value of spending memory on additional bits versus low-rank residual rank under equal total footprint.
4. Evaluate whether activation-space error is a better allocation signal than cheap weight-space proxies.
5. Produce practical heuristics for fixed-budget hybrid compression in post-training settings.

## Refined Novelty Claim

The contribution of this work is not the generic combination of quantization and low-rank structure. Its novelty lies in a **post-training budget-allocation framework** that explicitly chooses how each marginal unit of memory should be spent under a fixed deployment budget.

More concretely, the project contributes:

- a PTQ-focused formulation of hybrid compression as a **resource allocation problem**,
- an allocator that compares **gain per byte** from extra precision versus extra rank,
- an empirical study of **when bit allocation dominates uniform low-rank repair and where residual repair still adds value**,
- and practical guidance for selecting repair layers using calibration statistics without full retraining.

## Proposed Method

We begin with a pretrained language model and produce a post-training quantized baseline. Let the quantized weight matrix for layer `i` be `Q_i`, derived from the original full-precision weight `W_i`. The model may then be repaired using a low-rank residual term:

`W_i ≈ Q_i + A_i B_i`

where `A_i B_i` has rank `r_i` and is added only for selected layers.

The key question is not simply whether to add `A_i B_i`, but whether doing so is a better use of memory than increasing the precision of `Q_i`.

### Core Design Principle

The mainline method is **post-training and calibration-only**. The goal is to keep the contribution squarely in the PTQ setting and avoid dependence on expensive retraining. Gradient-based correction will be treated as an optional extension, not the main mechanism.

### Quantization Baseline

A single strong PTQ method will be chosen as the primary quantization backbone for the main experiments in order to isolate the allocation question. Additional PTQ methods may be included as secondary comparisons to test whether the learned allocation behavior depends on the quantizer.

### Low-Rank Residual Construction

The project will begin with an analytic residual repair method:

- compute the quantization residual `R_i = W_i - Q_i`,
- construct a rank-`r_i` approximation of `R_i` using truncated SVD,
- attach this approximation only to selected layers.

This provides a strong calibration-free baseline for residual repair. A secondary extension may explore lightweight gradient-based fitting of `A_i` and `B_i` on calibration data, but the main conclusions should not depend on such training.

## Allocation Framework

The project will formulate memory spending as a discrete allocation problem.

For each layer, define a set of candidate actions, such as:

- increase quantization precision by one level,
- add a low-rank residual block of size `Δr`,
- or leave the layer unchanged.

Each action has:

- a **memory cost** in bytes,
- and an estimated **quality gain** measured on calibration data.

The allocator will then choose the set of actions that maximizes predicted quality improvement subject to a total memory budget.

### Candidate Allocation Signals

The project will compare several signals for estimating action value:

- **Weight-space residual norms** such as Frobenius error,
- **Activation-space output deviation** measured on calibration samples,
- **Sensitivity-weighted proxies** based on output amplification or layer importance,
- and **marginal gain curves** obtained by directly testing small repair increments.

Activation-space error is expected to be the most informative because it better reflects downstream degradation, while weight-space norms serve as a cheap baseline.

### Allocation Policies

The following policies will be compared:

- **Uniform rank allocation**: every eligible layer receives the same residual rank.
- **Error-based rank allocation**: rank is assigned according to layerwise quantization damage.
- **Bit-only mixed-precision allocation**: the budget is spent only on non-uniform quantization precision.
- **Greedy hybrid allocation**: at each step, allocate the next memory chunk to the layer-action pair with highest predicted quality gain per byte.
- **Heuristic knapsack-style allocation**: solve the global budget assignment over a discrete action set.

The allocator is the centerpiece of the project. Based on the current evidence, the expected mainline is to allocate bits first and then test whether low-rank repair helps on the remaining hardest layers.

## Error Measurement

A major part of the study is to distinguish between cheap and meaningful allocation signals.

### Weight-Space Error

Weight-space error measures the discrepancy between original and compressed weights. It is computationally simple but may correlate weakly with actual model quality because it ignores the distribution of activations.

### Activation-Space Error

Activation-space error measures how much a compressed layer perturbs its outputs on calibration data. This can be defined as the average deviation between the full-precision layer output and the compressed or repaired layer output over a calibration set. This is more expensive to compute but is expected to be a better proxy for downstream performance loss.

The proposal will explicitly compare whether the extra cost of activation-space profiling is justified by better allocation decisions.

## Scope and Assumptions

This project targets **memory-constrained inference settings** where the central objective is quality at fixed model footprint. It does not assume that low-rank residuals automatically improve latency. In fact, residual branches may increase wall-clock cost if they require additional matrix multiplications or lack fused kernels.

Accordingly:

- **memory footprint** is the primary constraint,
- **quality recovered per added megabyte** is the primary efficiency metric,
- and **latency** is a secondary but reported metric.

The project will not claim inference-speed improvements without explicit evidence.

## Experimental Design

### Model Ladder

The experiments will use a small-to-medium scaling ladder:

- a small model for rapid ablations and profiling,
- a medium model for stable comparisons,
- and at least one larger open-weight model for final validation.

The small model is for iteration only; the main claim should rest on a model size large enough for compression behavior to be meaningful.

### Compression Regimes

The study will focus on aggressive but practical regimes such as:

- 4-bit weight-only quantization,
- 3-bit weight-only quantization,
- and optionally an 8-bit or mixed-precision reference.

These settings create distinct error regimes and allow the project to study when low-rank repair becomes worthwhile.

### Baselines

The baseline set will include:

- full-precision model,
- pure quantized model,
- mixed-precision quantization baseline,
- pure low-rank compression baseline where feasible,
- uniform hybrid baseline with fixed residual rank,
- and the proposed adaptive hybrid allocator.

All comparisons must be made at **equal total memory**.

The practical interpretation of the current pilot results is that the mixed-precision quantization baseline should be treated as the main method to beat, not as a minor ablation.

### Calibration Data

A standard held-out text corpus will be used for calibration and profiling. The same calibration budget will be used across all methods to ensure fair comparison.

### Evaluation Metrics

The evaluation will report:

- perplexity on held-out text,
- compact downstream accuracy or zero-shot task performance,
- total parameter memory in bytes,
- bytes added by each repair mechanism,
- inference latency where implementation permits fair comparison,
- and quality recovered per added megabyte.

The central evaluation metric is **quality at equal memory**, not raw task performance in isolation.

## Key Analyses

The paper should answer more than whether one method wins on average. It should explicitly analyze:

- which layer types receive the most useful repairs,
- whether attention or MLP blocks dominate recoverable damage,
- how quickly marginal returns diminish as rank increases,
- at what compression level extra rank becomes more valuable than extra bits,
- whether activation-space metrics materially outperform weight-space proxies,
- and whether the learned allocation policy is robust across quantization methods.

A particularly strong result would be a set of **gain-per-byte curves** showing the decision boundary between spending memory on bits and spending it on rank.

## Expected Contributions

1. A clear formulation of post-training compression as a fixed-budget resource allocation problem.
2. A calibration-driven allocator that compares extra precision and low-rank repair on equal footing.
3. An empirical map of when mixed-precision allocation dominates uniform low-rank repair.
4. Practical guidance on when residual low-rank correction is preferable as a second-stage hybrid method rather than a first-line intervention.
5. A reproducible benchmark setup for equal-memory compression studies.

## Success Criteria

The project will be successful if it demonstrates one or more of the following:

- adaptive hybrid allocation consistently outperforms uniform hybrid allocation at equal memory,
- hybrid allocation outperforms bit-only mixed-precision allocation in at least one practically relevant compression regime,
- activation-space signals provide meaningfully better allocation than cheap weight-space proxies,
- or the study produces a clear negative result showing that low-rank repair is rarely worth its byte cost under PTQ-only constraints.

A negative result would still be valuable if it is systematic and explains when bits dominate rank.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Quantization residuals may not be sufficiently low-rank to support efficient repair. | Profile singular value decay early and treat this as an explicit empirical question. |
| The hybrid method may underperform strong mixed-precision baselines. | Position the result as a decision study on when bits beat rank rather than assuming hybrid must win. |
| Weight-space metrics may prove misleading for allocation. | Make activation-space profiling part of the mainline evaluation and quantify the cost-benefit tradeoff. |
| Additional residual branches may increase latency too much for practical use. | Keep memory-constrained deployment as the main target and report latency honestly without overclaiming. |
| The method may depend heavily on the chosen quantizer. | Evaluate at least one secondary quantization backbone to test robustness. |
| The search space over bits and rank may become combinatorial. | Use a discrete action space with greedy or knapsack-style heuristics rather than attempting full joint optimization. |

## Timeline

1. **Weeks 1–2:** Build PTQ baselines, evaluation harness, and fixed-memory accounting.
2. **Weeks 3–4:** Profile layerwise residual structure and activation-space degradation.
3. **Weeks 5–6:** Implement SVD-based residual repair and uniform hybrid baselines.
4. **Weeks 7–8:** Implement bit-only and hybrid allocation policies using gain-per-byte estimates.
5. **Weeks 9–10:** Run equal-memory experiments across models and compression regimes.
6. **Weeks 11–12:** Analyze layer patterns, ablations, and sensitivity to allocation signals.
7. **Weeks 13–14:** Write report, finalize plots, and release reproducible code.

## Deliverables

- A research paper draft or technical report.
- Reproducible code for equal-memory hybrid compression experiments.
- Quality-memory tradeoff plots across bit-only, rank-only, and hybrid allocation strategies.
- Layerwise analyses of residual structure and repair payoff.
- Recommendations for when to spend memory on bits versus rank in PTQ deployment.

## One-Sentence Summary

This project studies a sharper question than "quantization versus low-rank compression": under a fixed model-size budget, where should the next byte go in a quantized language model, and can targeted low-rank residual repair recover more quality than spending that byte on extra precision?
