# Budget-Aware Post-Quantization Correction: When Should the Next Byte Buy Bits or Rank?

## Abstract

Post-training quantization is often followed by a small corrective budget, but that budget can be spent in different ways: on selective bit-width upgrades, on low-rank residual repair, or on hybrids of both. We study that marginal-byte decision directly in a bounded matched-budget comparison over RTN and GPTQ, spanning `Qwen3-0.6B`, `Qwen3-1.7B`, `SmolLM3-3B`, and `Qwen3-8B`, with targeted bits and targeted rank policies. The main result is regime dependence rather than a universal winner. Under RTN, rank wins at `0.6B` while bits win at larger scales; under GPTQ, the `1.7B` point becomes ambiguous under calibration resampling, the `3B` point favors bits over rank without improving on baseline, and bits wins at `8B`. Downstream evaluation shows that the qualitative regime picture remains visible beyond perplexity, but that small perplexity gaps do not reliably predict task-level ordering among nearby compressed policies. Multi-seed resampling weakens the original `1.7B` GPTQ rank-over-bits claim, and latency-plus-VRAM benchmarks show that deployment recommendations are workload-dependent. The practical conclusion is conditional rather than absolute: the next byte should be allocated according to quantizer, model scale, and serving workload, not by a global bits-versus-rank rule.

## 1. Introduction

Post-training quantization reduces model memory, but it does not remove the need for quality recovery. In practice, the relevant question is therefore not whether to quantize, but how to spend a small amount of memory after quantization. That extra budget can buy more bits, low-rank residual repair, or some hybrid of both.

We treat that choice as a budget-allocation problem. Under a fixed post-quantization memory budget, should the next byte go to precision upgrades or to low-rank correction? The answer is not obvious. Higher precision reduces quantization error directly in selected matrices, while low-rank repair can target structured residuals without changing the base bit-width. Both strategies are plausible, and both can look favorable depending on quantizer, model scale, and how the extra budget is allocated.

This question matters operationally because deployment teams often already have a quantized model and a small additional memory budget. At that point, the choice is not between full precision and quantization; it is between different corrective actions on top of an already-compressed base model. Prior work studies quantization, mixed precision, and lightweight repair, but usually does not isolate this marginal-byte decision cleanly.

Our study is intentionally bounded. We do not attempt unrestricted policy search or introduce a new repair family. Instead, we compare targeted bits, targeted rank, and a limited hybrid follow-up under matched post-quantization budgets, using the same experiment harness across RTN and GPTQ. The empirical result is regime dependence. Neither targeted bits nor targeted rank is universally dominant. RTN and GPTQ disagree at some scales, scale changes the preferred policy within each quantizer family, and downstream and latency analyses show that perplexity alone is not enough to collapse the study into a single winner. The intended takeaway is therefore not a single global ordering, but that three decision axes materially affect the recommended corrective action: quantizer family, model scale, and serving workload.

Our contributions are:

1. We formulate budget-aware post-quantization correction as a marginal-byte decision problem: should extra memory buy targeted precision upgrades or targeted low-rank repair?
2. We provide a bounded matched-budget comparison across RTN and GPTQ over multiple model scales, using a shared allocator framework and persistent-byte accounting.
3. We show that the preferred corrective action is regime-dependent rather than globally ordered.
4. We show that single-seed perplexity alone is insufficient for close comparisons: multiseed resampling, downstream evaluation, and serving benchmarks materially change the recommendation.

## 2. Related Work

This paper sits at the intersection of post-training quantization, selective precision allocation, and lightweight post-quantization repair. A large body of work studies post-training quantization as a way to reduce model footprint and inference cost, including weight-only and activation-aware schemes such as GPTQ, AWQ, and related PTQ systems `[CITATION]`. Those methods establish that high-quality low-bit compression is possible, but they usually optimize the quantized model itself rather than the decision of how to spend a small corrective budget after quantization.

A second line of work studies lightweight adaptation or repair, including low-rank methods and quantization-aware adapter strategies such as LoftQ, as well as more structured compression or mixed-precision systems such as SqueezeLLM and QuIP# `[CITATION]`. These approaches show that residual structure can be exploited after compression, but they often involve training-time adaptation, broader search spaces, or objectives other than a tightly matched post-hoc memory budget. Our focus is narrower: bounded corrective allocation on top of already-quantized models.

The closest conceptual framing is mixed precision or budget-aware allocation. Prior work has shown that some layers matter more than others and that selective allocation can outperform uniform allocation `[CITATION]`. The distinction in this paper is that we isolate a specific decision problem: given a model that is already quantized, and given a small additional persistent-memory budget, should that budget buy targeted bits or targeted rank? We do not claim a new quantizer or a new repair method. We claim that this decision problem is operationally meaningful, empirically nontrivial, and regime-dependent across quantizer, scale, and serving workload.

## 3. Method

We cast post-quantization correction as greedy action selection under an added-byte budget. Starting from a quantized baseline model, we define a finite candidate action set, assign each action a persistent byte cost, and greedily select actions until the budget is exhausted.

### 3.1 Budget and Action Space

Let the baseline quantized model have persistent storage cost `M_base` bytes. Each experiment specifies an added-byte budget `B`, either directly as `budget_bytes` or indirectly as a percentage of `M_base`. The allocator may choose actions only if the cumulative added cost stays within `B`.

The budget counts only persistent model-state storage added to the compressed checkpoint. It includes quantized weights, low-rank factors, and quantization metadata, but excludes transient inference-time memory such as KV cache, activations, and workspace buffers.

The bounded action space contains three policy families:

- `targeted bits`: selectively increase the bit-width of chosen matrices
- `targeted rank`: add low-rank residual repair to chosen matrices
- `hybrid`: a bounded second-stage follow-up used only as a secondary branch, not as the main headline comparison

In the main paper scope, actions apply at `matrix` granularity to a shared candidate pool of `12` `nn.Linear` matrices derived from the baseline damage profile. The value `12` is fixed across the main targeted runs as a bounded compute choice: it keeps the targeted bits and targeted rank pools aligned while still focusing on the most damaged matrices under the baseline profiling stage. Embeddings and `lm_head` are not part of the targeted action space in the current study. For both `1.7B` and `8B` GPTQ, the candidate pool uses:

- base bit-width `4`
- group size `128`
- symmetric quantization
- candidate bit upgrade `4 -> 8` in the main frontier
- candidate ranks `[4, 8, 16, 32]`

The `4 -> 8` bit action is used in the main frontier to keep the bits-side action space simple and matched across scales. Richer `4 -> 5` and `4 -> 6` variants were explored separately as bounded follow-ups rather than part of the core paper comparison.

For a targeted-bit action on weight matrix `W`, the added persistent cost is:

`Delta C_bits(W, b -> b') = C_q(W, b') - C_q(W, b)`

where `C_q` is the measured quantized storage cost, including both quantized weights and quantization metadata.

For a targeted-rank action that increases a layer from rank `r_prev` to `r`, with matrix shape `m x n` and factor storage width `s` bytes, the added persistent cost is:

`Delta C_rank = (m * Delta r + Delta r * n) * s`

where `Delta r = r - r_prev`. In the main GPTQ targeted-rank runs, `s = 2` bytes because the repair factors are stored in `float16`.

### 3.2 Allocation Signals

Each candidate action is given a scalar proxy score derived from the baseline damage profile. In the current implementation:

- activation proxy: layer `activation_relative_l2`
- weight proxy: layer `relative_fro_error`

For targeted-bit actions at matrix granularity, the greedy score is simply the layer proxy score normalized by the added byte cost:

`score_bits(a) = p_l / Delta C_bits(a)`

where `p_l` is the chosen layer proxy.

For targeted-rank actions, the allocator first profiles the residual energy spectrum of the quantization residual for each candidate matrix. The score of an incremental rank step is:

`score_rank(a) = p_l * Delta E_l(r_prev -> r) / Delta C_rank(a)`

where `Delta E_l` is the incremental explained residual-energy fraction captured by moving from `r_prev` to `r`.

The allocator-proxy ablation later tests whether the more expensive activation-space profiling is justified by better final allocation quality.

### 3.3 Greedy Selection Rule

The main paper uses a simple static greedy allocator.

For targeted bits:

1. build one candidate action for each eligible matrix and target bit-width
2. sort by predicted gain per byte
3. select the highest-scoring action that still fits the remaining budget
4. do not select two matrix-level bit upgrades for the same target
5. stop when no further action fits

For targeted rank:

1. build incremental rank actions for each eligible matrix
2. expose only the next available rank step for each matrix
3. select the highest-scoring available step that fits the remaining budget
4. advance that matrix to its new current rank
5. stop when no further step fits

This procedure is intentionally heuristic and bounded. The paper does not claim global optimality; it compares reproducible matched-budget corrective policies.

The goal of the allocator is not to find a globally optimal compressed model, but to compare bounded corrective policy families under the same persistent-memory budget.

### 3.4 Evaluation Targets

We evaluate:

- perplexity on WikiText-style language modeling
- downstream task accuracy using a six-task `lm-eval` suite
- calibration-resampling stability across seeds
- decode throughput, decode latency, first-token latency, and peak VRAM for serving

This combination matters because a compression policy can look favorable by perplexity yet fail to dominate in downstream task metrics or deployment efficiency.

## 4. Experimental Setup

### 4.1 Quantizers and Models

We study two quantizers:

- `RTN`
- `GPTQ`

Across the following model scales:

- `Qwen3-0.6B`
- `Qwen3-1.7B`
- `SmolLM3-3B`
- `Qwen3-8B`

The full cross-scale regime map uses RTN for `0.6B`, `1.7B`, `3B`, and `8B`, and GPTQ for `1.7B`, `3B`, and `8B`.

### 4.2 Core Metrics

The main quality metric is perplexity. To test whether small perplexity differences are meaningful, we add three additional evaluation branches:

- downstream evaluation on six tasks:
  - `hellaswag`
  - `arc_easy`
  - `arc_challenge`
  - `winogrande`
  - `piqa`
  - `boolq`
- calibration-seed resampling at the closest GPTQ comparison points
- generation latency and peak-VRAM benchmarking

For the main GPTQ targeted runs, calibration uses `WikiText-2 (wikitext-2-raw-v1)` with `128` training sequences of length `512`, sampled with seeded shuffle. Perplexity evaluation uses the test split at sequence length `512` and stride `512`. The multiseed branch reruns the same calibration pipeline with seeds `42`, `123`, and `456`.

Downstream evaluation uses our wrapper around `lm_eval.models.huggingface.HFLM` with batch size `4` and task-specific few-shot settings: `hellaswag` `0`, `arc_easy` `0`, `arc_challenge` `25`, `winogrande` `5`, `piqa` `0`, and `boolq` `0`. The main targeted GPTQ runs were executed on single-GPU Modal workers, while the latency benchmarks use `A10G` for `Qwen3-1.7B` and `A100-40GB` for `Qwen3-8B`.

### 4.3 Latency Benchmark Contract

Latency uses greedy generation with KV cache enabled, prompt length `512`, decode length `128`, `3` warmup runs, and `10` timed repetitions. We report:

- decode tokens/sec
- decode ms/token
- first-token latency
- peak VRAM

The primary serving comparison points are:

- `Qwen3-1.7B` GPTQ on `A10G`
- `Qwen3-8B` GPTQ on `A100`
- baseline 4-bit, best bits-only, and best rank-only
- batch sizes `1` and `8`

## 5. Results

### 5.1 Cross-Quantizer Regime Map

The project-level result is a regime map rather than a universal winner.

| Quantizer | Model | Baseline | Best Bits | Best Rank | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| RTN | `Qwen3-0.6B` | `30.5169` | `30.2506` | `29.7223` | rank-favoring |
| RTN | `Qwen3-1.7B` | `21.3102` | `21.1505` | `21.2971` | bits-favoring |
| RTN | `SmolLM3-3B` | `47.9169` | `47.4955` | `47.9833` | bits-favoring |
| RTN | `Qwen3-8B` | `16.1939` | `16.1429` | `16.2035` | bits-favoring |
| GPTQ | `Qwen3-1.7B` | `15.9137` | `15.8914` | `15.8823` | single-seed rank, multiseed ambiguous |
| GPTQ | `SmolLM3-3B` | `11.5366` | `11.5483` | `11.6482` | bits over rank, but no corrective win over baseline |
| GPTQ | `Qwen3-8B` | `11.7970` | `11.7823` | `11.7962` | bits-favoring |

Two points matter most. First, RTN and GPTQ do not induce the same frontier: `Qwen3-1.7B` flips from bits-favoring under RTN to a single-seed rank-favoring GPTQ result that later becomes ambiguous under resampling. Second, scale changes the answer within each quantizer family. This prevents a single global bits-versus-rank conclusion and motivates the rest of the robustness analysis.

### 5.2 Multi-Seed Resampling Separates Stable From Unstable Claims

The closest GPTQ comparisons were too narrow to trust on single-seed evidence alone, so we reran the critical policy pairs across calibration seeds `42`, `123`, and `456`.

| Scale | Policy | Mean PPL | Std PPL |
| --- | --- | ---: | ---: |
| `Qwen3-1.7B` | bits | `15.8285` | `0.1235` |
| `Qwen3-1.7B` | rank | `15.8200` | `0.1227` |
| `SmolLM3-3B` | bits | `11.6211` | `0.0740` |
| `SmolLM3-3B` | rank | `11.7337` | `0.0988` |
| `Qwen3-8B` | bits | `11.5975` | `0.1660` |
| `Qwen3-8B` | rank | `11.6154` | `0.1622` |

The `1.7B` GPTQ ordering is not robust enough to support a strong rank-over-bits claim: the mean gap is only `0.0085` PPL, far smaller than either policy’s observed spread. By contrast, the `8B` GPTQ comparison is modest but directionally stable: bits beats rank on all three seeds, with a mean advantage of `0.0178` PPL. The `3B` point is cleaner still, with bits ahead on every seed.

This reclassifies `GPTQ / 1.7B` from an apparent single-seed rank win to an ambiguous regime under calibration resampling. `GPTQ / 8B`, however, supports a stable bits-favoring interpretation.

### 5.3 Downstream Evaluation Shows Where Nearby Policy Rankings Become Less Stable

Downstream results preserve the broad regime picture but weaken any claim that small perplexity gaps cleanly determine policy ordering.

At `GPTQ / Qwen3-1.7B`, rank is best by single-seed perplexity (`15.8823`), but targeted bits slightly leads the downstream comparison: it has the best mean downstream score (`0.6769`) and wins more task-level comparisons. At `GPTQ / Qwen3-8B`, targeted bits is best by perplexity (`11.7823`), but downstream is effectively tied among baseline, targeted rank, and the secondary hybrid branch, with targeted rank having the highest mean downstream score by a very small margin (`0.7758` vs `0.7756` baseline and `0.7749` bits). At `GPTQ / SmolLM3-3B`, baseline remains best by perplexity, while targeted bits slightly improves average downstream score over baseline. The RTN `1.7B` anchor preserves the cross-quantizer contrast: targeted bits is best by perplexity, while targeted rank slightly edges the downstream mean.

The aggregated correlation result makes the same point more formally. Across the full quality range, including full-precision anchors, perplexity improvement and downstream improvement remain positively aligned (`r ≈ 0.77`). Within the compressed-policy regime, however, the correlation becomes weak and unstable (`r ≈ -0.21`). This prevents over-interpreting small perplexity gaps as reliable downstream differences.

### 5.4 Latency and Peak VRAM Add a Deployment Axis

The latency matrix adds the main deployment-facing result of the study: the serving recommendation is workload-dependent.

For `Qwen3-8B` on `A100` at batch size `1`, bits is essentially baseline-speed while rank is much slower:

| Policy | Decode tok/s | Decode ms/token | Peak VRAM MB |
| --- | ---: | ---: | ---: |
| Baseline (`R3_Q8B`) | `13.9960` | `71.4508` | `6178.06` |
| Bits (`G2B02_Q8B`) | `13.7762` | `72.5901` | `6255.19` |
| Rank (`G2R02_Q8B`) | `8.5127` | `117.4797` | `6444.69` |

Relative to baseline, bits adds only about `1.6%` decode latency while rank adds about `64.4%`. This is the strongest deployment-facing result in the study: at `8B` for interactive decode, bits dominates rank on both quality stability and serving latency.

At `Qwen3-8B` batch size `8`, the picture changes:

| Policy | Decode tok/s | Decode ms/token | Peak VRAM MB |
| --- | ---: | ---: | ---: |
| Baseline (`R3_Q8B`) | `69.4134` | `14.4075` | `6958.13` |
| Bits (`G2B02_Q8B`) | `112.0511` | `8.9306` | `22663.13` |
| Rank (`G2R02_Q8B`) | `115.5812` | `8.6520` | `7230.34` |

Here, both targeted policies outperform the baseline on decode throughput, and rank is slightly faster than bits while using far less peak VRAM. The recommendation therefore changes under higher-batch throughput serving: low-batch interactive decode and higher-batch throughput serving can favor different policies.

The `Qwen3-1.7B` latency results are also mixed. At batch size `1`, rank is the fastest policy; at batch size `8`, bits is fastest and rank becomes the slowest. Bits also carries a large serving-memory penalty at `1.7B`, while rank remains close to baseline in peak VRAM. Together with the multiseed result, this makes `1.7B` a genuinely ambiguous deployment regime rather than a clean policy win.

### 5.5 Allocator Ablation: Weight-Space Proxy Is the Better Default in the Current GPTQ Scope

The allocator ablation asks whether activation-space profiling justifies its extra cost relative to a cheap weight-space proxy. In the current `Qwen3-1.7B` GPTQ scope, it does not.

For targeted bits, activation and weight proxies converge to the same final target set and the same perplexity (`15.8993`), while activation profiling adds roughly `10-15` seconds of profiling and selection work. For targeted rank, activation-space selection is worse: activation yields `15.9224`, while the cheaper weight proxy yields `15.8823`, even though both end at the same repaired layer set and the same final rank caps. The difference comes from incremental action ordering rather than a different final target set.

This yields a bounded practical lesson: in the present GPTQ setup, activation profiling is not justified by allocation quality.

## 6. Discussion

### 6.1 Robustness of Claims

The empirical claims should be tiered by stability.

Stable:

- RTN is not rank-dominated; bits win at `1.7B`, `3B`, and `8B`
- GPTQ `8B` is directionally bits-favoring under multiseed resampling
- GPTQ `3B` is clearly not rank-favoring
- `8B` interactive decode favors bits over rank in practical latency terms

Conditional or unstable:

- GPTQ `1.7B` rank-over-bits by perplexity
- downstream ordering among nearby compressed policies
- global latency ordering without specifying batch size and serving regime

This is the scientific core of the paper. Some differences are clearly meaningful; others are not. The value of the study is that it separates those cases instead of forcing them into one ordering.

### 6.2 Practical Guidance

The most useful practical output of the study is a deployment-oriented regime guide.

For `8B` GPTQ:

- if the workload is interactive or low-batch, prefer targeted bits
- if the workload is higher-batch throughput serving and peak VRAM is tight, rank remains viable despite losing on perplexity

For `1.7B` GPTQ:

- do not claim a universal winner
- the quality gap is within noise under seed resampling
- the latency ordering flips with batch size
- bits and rank should be selected according to serving constraints, not by a blanket rule

For RTN:

- very small scale (`0.6B`) can favor rank
- once the model is large enough in the present matrix-level setup, bits becomes the stronger default

These recommendations are intentionally bounded. They are decision rules for the present study rather than a universal theory of all quantization-repair systems.

### 6.3 Limitations

This work is intentionally narrow in several ways. First, it studies bounded action spaces rather than open-ended search over all possible correction policies. Second, it considers only two quantizer families, RTN and GPTQ. Third, it uses a simple matrix-level low-rank repair family rather than broader adapter or jointly optimized correction methods. Fourth, it evaluates a limited task suite and model scale range. Fifth, its fairness claims depend on the chosen persistent-byte accounting for mixed-precision and low-rank actions. These constraints are a feature for the decision problem studied here, but they limit generality.

Several results point to clear follow-on work. Larger models could test whether the `8B` bits-favoring GPTQ result persists or flips again. New quantizer families could reveal different frontiers. More expressive shared-family or jointly optimized rank methods might outperform the bounded low-rank repair studied here. Those are natural next phases, but they should be framed as extension projects rather than missing pieces of the present one.

## 7. Appendix Notes

The appendix should carry the material that supports reproducibility without distracting from the main empirical line:

- full run inventory by quantizer, scale, and policy
- config naming conventions and run identifiers
- Modal and local execution notes
- saved artifact locations for downstream, multiseed, and latency branches
- any extended tables that would otherwise crowd the main text

## 8. Draft Figure and Table Plan

Must-have figures and tables for the current draft:

1. Cross-quantizer regime-map summary table
2. Multi-seed error-bar figure for GPTQ `1.7B`, `3B`, and `8B`
3. Latency and peak-VRAM table for `1.7B / A10G` and `8B / A100`
4. Activation-vs-weight allocator ablation table

Optional additions if time permits:

1. Gain-per-byte curves
2. Layer-type heatmaps

## 9. Canonical Supporting Artifacts

- project synthesis:
  - `docs/experiments/final_quantization_vs_svd_synthesis.md`
- downstream analysis:
  - `docs/experiments/downstream_item1_analysis.md`
- multiseed analysis:
  - `docs/experiments/item3_multiseed_analysis.md`
- latency analysis:
  - `docs/experiments/item4_latency_analysis.md`
- allocator ablation:
  - `docs/experiments/activation_vs_weight_ablation.md`
- latency summary table:
  - `results/analysis/latency_item4_summary.csv`
- multiseed summary table:
  - `results/analysis/multiseed_stability_all_summary.csv`
- generated paper assets:
  - `docs/experiments/assets/item3_multiseed_errorbar.csv`
  - `docs/experiments/assets/item4_latency_table.csv`
  - `docs/experiments/assets/item4_latency_overheads.csv`
