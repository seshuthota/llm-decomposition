# Paper-Readiness Plan: Budget-Aware Bits-vs-Rank Allocation

**Created**: 2026-03-14  
**Updated**: 2026-03-19  
**Target**: Strong main-track submission (MLSys / ICLR / NeurIPS)  
**Estimated duration**: 3–4 weeks  
**Status**: In progress — Items 1, 2, 3, and 4 are complete; Item 6 paper writing has an initial manuscript draft and is now in synthesis/polish

## Scope Decision

> [!IMPORTANT]
> **This paper covers both RTN and GPTQ as a cross-quantizer study.** The regime map is cross-quantizer by design — showing that quantizer choice changes the frontier is itself a key finding. RTN provides the motivating scale ladder; GPTQ provides the stronger quantizer frontier. Both appear in the main results table.

## Priority Tiers

> [!IMPORTANT]
> **Must-have (paper MVP)**: Items 1, 2, 3, 4, 6 — these are non-negotiable for submission.
> **Nice-to-have**: Item 5 (gain-per-byte curves) — do in parallel if cheap.
> **Optional extensions**: Items 7, 8 — only after MVP is complete.

---

## 1. Downstream Zero-Shot / Few-Shot Evaluation (~1.5 weeks) — MUST-HAVE

**Why**: Perplexity-only is the #1 reviewer rejection reason for compression papers. Both proposals explicitly promised "compact downstream accuracy or zero-shot task performance."

### Infrastructure Setup

- [x] Install `lm-eval-harness` in the `rl` conda environment
  ```bash
  conda run -n rl pip install lm-eval
  ```
- [x] Write a wrapper module `llm_decomposition/downstream_eval.py` that:
  - wraps `lm-eval-harness` HFLM with pre-built model objects
  - runs per-task evaluation with configurable few-shot counts
  - saves results JSON to `results/<run_dir>/downstream_metrics.json`
- [x] Integrate `_maybe_run_downstream()` into all 8 execute functions in `hf_backend.py`
- [x] Update Modal runners with `lm-eval[hf]` dep and `downstream_metrics.json` artifact collection
- [x] Write `scripts/generate_downstream_configs.py` to auto-generate configs from existing runs
- [x] Generate 14 downstream configs across 3 model groups (1.7B, 8B, 3B)
- [x] Validate all configs pass `--dry-run` (3 manifests)

### Task Selection

- [x] Select 6 standard tasks for evaluation:
  - `hellaswag` (0-shot)
  - `arc_easy` (0-shot)
  - `arc_challenge` (25-shot)
  - `winogrande` (5-shot)
  - `piqa` (0-shot)
  - `boolq` (0-shot)
- [x] Verify all tasks run on the full-precision model before proceeding

### Evaluation Runs

Run downstream eval for each regime point that matters most. **Full-precision reference is included at every scale** for consistent paper tables.

#### Qwen3-1.7B GPTQ (rank-wins regime)

- [x] Full-precision reference
- [x] GPTQ 4-bit baseline (`R3_Q17B`)
- [x] Best bits-only (`G2B03_Q17B`)
- [x] Best rank-only (`G2R02_Q17B`)
- [x] Best hybrid (`H2R02M_Q17B`)

Observed summary:

- `DS_FP_Q17B` validated the end-to-end Modal downstream path
- the full `1.7B` GPTQ policy set is now complete on disk
- perplexity ordering remains `rank > bits > hybrid > baseline`
- downstream task accuracy is more mixed and task-dependent than the perplexity ordering
- this strengthens the paper framing: perplexity and downstream behavior are related, but not interchangeable

#### Qwen3-8B GPTQ (bits-wins regime)

- [x] Full-precision reference
- [x] GPTQ 4-bit baseline (`R3_Q8B`)
- [x] Best bits-only (`G2B02_Q8B`)
- [x] Best rank-only (`G2R02_Q8B`)
- [x] Best hybrid (`H2R02_Q8B`)

Observed summary:

- the full `8B` GPTQ policy set is now complete on disk
- perplexity ordering remains `bits > hybrid > rank > baseline`
- downstream ordering is again more mixed than perplexity:
  - `hellaswag` and `winogrande` are best for hybrid by a small margin
  - `arc_easy`, `piqa`, and `boolq` are best for rank by a small margin
  - `arc_challenge` is effectively tied between bits and hybrid
- the result strengthens the same paper claim seen at `1.7B`:
  - perplexity is useful, but downstream task behavior is policy- and task-dependent
  - there is no single downstream policy winner across all tasks even when the perplexity frontier is cleaner

#### SmolLM3-3B GPTQ (neutral regime)

- [x] Full-precision reference
- [x] GPTQ 4-bit baseline (`R3_S3B`)
- [x] Best bits-only (`G3B02_S3B`)
- [x] Best rank-only (`G3R02_S3B`)

Observed summary:

- the full `3B` GPTQ downstream set is now complete on disk
- perplexity ordering remains weakly consistent with the earlier neutral regime:
  - baseline best
  - bits-only second
  - rank-only third
- downstream accuracy is also mixed but directionally similar:
  - bits-only is competitive with or slightly better than baseline on `hellaswag`, `arc_challenge`, and `piqa`
  - rank-only remains the weakest overall policy in both perplexity and most downstream tasks
- this keeps `3B` in the paper as the "neutral / mixed" midpoint between the `1.7B` rank-favoring and `8B` bits-favoring GPTQ regimes

#### RTN Cross-Quantizer Anchor (one scale point)

- [x] Qwen3-1.7B full-precision reference (reuse from above)
- [x] Qwen3-1.7B RTN 4-bit baseline (`R2_Q17B`)
- [x] Qwen3-1.7B RTN best bits (`P2B03_Q17B`)
- [x] Qwen3-1.7B RTN best rank (`P2R02_Q17B`)

Observed summary:

- the full RTN `1.7B` downstream anchor is now complete on disk
- perplexity ordering remains:
  - bits-only best
  - rank-only second
  - RTN baseline third
- downstream behavior is close but still slightly task-dependent:
  - bits-only is best on `hellaswag`, `arc_challenge`, and `winogrande`
  - rank-only is slightly best on `arc_easy`, `piqa`, and `boolq`
- this gives the paper a clean cross-quantizer contrast:
  - under RTN at `1.7B`, bits remain the better policy
  - under GPTQ at `1.7B`, rank remains the better policy by perplexity even though downstream ordering is less uniform

### Analysis

- [x] Build a consolidated downstream results table (model × quantizer × policy × task)
- [x] Check whether downstream Δaccuracy correlates with ΔPPL across all regime points
- [x] Compute "quality recovered per added MB" in downstream accuracy units
- [x] Write a summary section for the paper: does the regime map hold beyond perplexity?

Final analysis summary:

- canonical Item 1 report: `docs/experiments/downstream_item1_analysis.md`
- generated analysis tables: `results/analysis/downstream_run_summary.csv` and `results/analysis/downstream_group_deltas.csv`
- global trend:
  - `ΔPPL` and downstream quality remain positively aligned when full-precision anchors are included (`r ≈ 0.77`)
  - inside the compressed-policy regime, that relationship is weak and unstable (`r ≈ -0.21`)
- paper-relevant interpretation:
  - perplexity remains a useful global quality measure
  - but it is not sufficient to rank nearby compressed policies
  - the regime map does hold beyond perplexity, though downstream policy ordering is more task-dependent than the perplexity frontier
- strongest downstream takeaways:
  - GPTQ `1.7B`: rank is best by perplexity, but bits has the best mean downstream score and wins the most tasks
  - GPTQ `8B`: bits is best by perplexity, while downstream is essentially tied between baseline, rank, and hybrid
  - GPTQ `3B`: baseline remains best by perplexity, bits slightly improves mean downstream score, and rank stays weakest
  - RTN `1.7B`: bits is best by perplexity, while rank slightly edges the mean downstream score and bits wins the most tasks

---

## 2. Activation-Space vs Weight-Space Allocator Ablation (~3 days) — MUST-HAVE

**Why**: Research Objective 4 explicitly asks whether activation-space error outperforms weight-space proxies. This was the primary undelivered promised analysis.

### Implementation

- [x] Verify the `proxy_family` config parameter already supports both `"activation"` and `"weight"` paths in the allocator
  - Check `_build_bit_actions()` and `_build_rank_actions()` in `hf_backend.py`
  - the real bug was elsewhere: allocator selection was still reading the stale candidate-pool layer summary rather than fresh current-base-model profiling
  - fixed by adding `selection_profile_source: "current_base_model"` and resolving the selection map from the current base model before action construction
- [x] Create two matched config pairs for Qwen3-1.7B GPTQ at +1% budget:
  - `g2b02_weight_proxy_Q17B.json` — bits allocation using weight-space signal
  - `g2r02_weight_proxy_Q17B.json` — rank allocation using weight-space signal
  - paired activation configs were also added so the comparison is explicit and isolated

### Runs

- [x] Run bits-only with weight-space proxy on 1.7B GPTQ (`G2B02W_Q17B`)
- [x] Run rank-only with weight-space proxy on 1.7B GPTQ (`G2R02W_Q17B`)
- [ ] Run bits-only with weight-space proxy on 8B GPTQ (`G2B02W_Q8B`) — if budget allows
- [ ] Run rank-only with weight-space proxy on 8B GPTQ (`G2R02W_Q8B`) — if budget allows

Observed summary:

- canonical Item 2 report: `docs/experiments/activation_vs_weight_ablation.md`
- generated ablation tables:
  - `results/analysis/proxy_ablation_q17b_summary.csv`
  - `results/analysis/proxy_ablation_q17b_selection_diff.json`
- bits:
  - activation proxy `G2B02A_Q17B`: `15.8993`
  - weight proxy `G2B02W_Q17B`: `15.8993`
  - same final target set, same final quality, but activation profiling added about `27.6 s` of profiling + selection work while weight proxy was effectively free
- rank:
  - activation proxy `G2R02A_Q17B`: `15.9224`
  - weight proxy `G2R02W_Q17B`: `15.8823`
  - both paths ended at the same repaired layer set and same final rank caps, but activation profiling changed incremental action ordering enough to hurt quality
- bounded interpretation:
  - for the current `1.7B` GPTQ paper scope, activation-space profiling is not justified by allocation quality
  - the `8B` weight-proxy follow-up remains optional and is not needed for the paper MVP

### Analysis

- [x] Compare PPL: activation-allocated vs weight-allocated at same budget
- [x] Compare which layers are selected by each allocator (produce a layer-selection diff table)
- [x] Measure profiling time: activation-space vs weight-space (report cost-benefit)
- [x] Write conclusion: "activation-space profiling is / is not justified by its allocation quality"

---

## 3. Multi-Seed / Calibration Resampling (~2 days) — MUST-HAVE

**Why**: The closest-call results (1.7B GPTQ: 15.8823 vs 15.8914, 8B GPTQ: 11.7823 vs 11.7962) have margins of ~0.01 PPL. Reviewers will question if these are noise. This is more publication-critical than extra visualizations.

### Implementation

- [x] Modify the calibration data loading to accept a `calib_seed` parameter that shuffles which WikiText-2 training samples are chosen
  - The codebase already supports this via `"sampling": "seeded_shuffle"` and `"seed": N` in calibration config
  - Created `scripts/generate_multiseed_configs.py` to auto-generate multi-seed configs
- [x] Create configs for 3 seeds (seed 42, 123, 456) on the two closest-call results

### Runs

- [x] Qwen3-1.7B GPTQ rank policy `G2R02W_Q17B` × 3 seeds
- [x] Qwen3-1.7B GPTQ bits policy `G2B02W_Q17B` × 3 seeds
- [x] Qwen3-8B GPTQ `G2B02_Q8B` × 3 seeds
- [x] Qwen3-8B GPTQ `G2R02_Q8B` × 3 seeds

### Analysis

- [x] Report mean ± std for each policy at each scale
- [x] Confirm the rank > bits ordering at 1.7B and bits > rank at 8B hold across seeds
- [x] If ordering is unstable → report it honestly as "within noise" and adjust claims
- [x] Resolve the `8B` Modal GPTQ multiseed blocker for bits policy
  - canonical note: `docs/roadmap/qwen3_8b_multiseed_blocker.md`
  - root causes were two separate issues:
    - the Modal GPTQ worker needed eager `gptqmodel` import priming to avoid `NameError: QuantizeConfig is not defined`
    - the generated `Q8B` multiseed configs were missing `selection_profile_source: "current_base_model"`, so they tried to read a non-mounted baseline `layer_errors.json`
  - fixed in `llm_decomposition/gptq_backend.py`, `scripts/modal_experiment_gptq.py`, and `scripts/generate_multiseed_configs.py`

Observed summary:

- canonical Item 3 report: `docs/experiments/item3_multiseed_analysis.md`
- generated config script: `scripts/generate_multiseed_configs.py`
- multi-seed configs: `configs/multiseed/` (6 configs across 2 policies × 3 seeds)
- generated analysis table: `results/analysis/multiseed_stability_all_summary.csv`
- completed `8B` bits multiseed runs saved under:
  - `results/modal_importfix_probe_v2/qwen3_8b_gptq_transfer_s42/G2B02_Q8B_s42`
  - `results/modal_importfix_probe_v2/qwen3_8b_gptq_transfer_s123/G2B02_Q8B_s123`
  - `results/modal_importfix_probe_v2/qwen3_8b_gptq_transfer_s456/G2B02_Q8B_s456`
- completed `8B` rank multiseed runs saved under:
  - `results/modal_importfix_probe_v2/qwen3_8b_gptq_transfer_s42/G2R02_Q8B_s42`
  - `results/modal_importfix_probe_v2/qwen3_8b_gptq_transfer_s123/G2R02_Q8B_s123`
  - `results/modal_importfix_probe_v2/qwen3_8b_gptq_transfer_s456/G2R02_Q8B_s456`

**Results:**

| Policy | Seed | Perplexity |
|--------|------|------------|
| Rank (G2R02W) | 42 | 15.8325 |
| Rank (G2R02W) | 123 | 15.6916 |
| Rank (G2R02W) | 456 | 15.9360 |
| Bits (G2B02W) | 42 | 15.8527 |
| Bits (G2B02W) | 123 | 15.6947 |
| Bits (G2B02W) | 456 | 15.9381 |

**Statistics:**
- Rank: mean=15.8200, std=0.1037
- Bits: mean=15.8285, std=0.1048

**Key finding:** The rank > bits ordering is NOT stable across seeds:
- Seed 42: rank (15.83) < bits (15.85) → rank wins
- Seed 123: rank (15.69) ≈ bits (15.69) → tie  
- Seed 456: rank (15.94) ≈ bits (15.94) → tie

**Paper implication:** The original PPL difference (15.88 vs 15.89) was within noise (~0.10 std). The paper should report this honestly as "within experimental noise" and avoid strong claims about rank > bits at 1.7B scale.

**8B multiseed status:**

| Policy | Seed | Perplexity |
|--------|------|------------|
| Rank (G2R02_Q8B) | 42 | 11.7962 |
| Rank (G2R02_Q8B) | 123 | 11.4828 |
| Rank (G2R02_Q8B) | 456 | 11.5671 |
| Bits (G2B02_Q8B) | 42 | 11.7823 |
| Bits (G2B02_Q8B) | 123 | 11.4609 |
| Bits (G2B02_Q8B) | 456 | 11.5494 |

**8B multiseed statistics:**
- Rank: mean=11.6154, std=0.1622
- Bits: mean=11.5975, std=0.1660
- rank runs completed cleanly with finite GPTQ validation and identical memory footprint (`3907674112` bytes)
- bits runs completed cleanly with finite GPTQ validation and identical memory footprint (`3948437504` bytes)
- the bits > rank ordering held for all three seeds at 8B:
  - seed 42: 11.7823 vs 11.7962
  - seed 123: 11.4609 vs 11.4828
  - seed 456: 11.5494 vs 11.5671
- mean gap is modest (`0.0178` PPL), but the ordering is directionally stable across seeds

Final analysis summary:

- `1.7B` GPTQ does not support a strong rank-over-bits claim after calibration resampling; the gap is within noise
- `3B` and `8B` both remain bits-favoring under the completed seed sweep
- the paper-ready framing is now:
  - `1.7B`: within noise
  - `3B`: bits-favoring midpoint
  - `8B`: directionally stable bits-favoring regime

**Current status:** Item 3 has a canonical report, Item 4 now has a canonical latency report with the full 12-run matrix, and Item 6 writing has started via the writing plan in `docs/roadmap/item6_writing_plan.md`.

---

## 4. Latency Measurement (~2 days) — MUST-HAVE

**Why**: Residual branches add extra matmuls. A practical user needs to know the latency cost of choosing rank over bits.

Canonical implementation tracker:

- `docs/roadmap/item4_latency_measurement_plan.md`
- benchmark contract is frozen there and the latency path is now implemented

Current Item 4 progress:

- full `Qwen3-8B / A100` latency block completed and saved:
  - `results/modal_latency/qwen3_8b_gptq_baselines/R3_Q8B__bs1`
  - `results/modal_latency/qwen3_8b_gptq_baselines/R3_Q8B__bs8`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2B02_Q8B__bs1`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2B02_Q8B__bs8`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2R02_Q8B__bs1`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2R02_Q8B__bs8`
- full `Qwen3-1.7B / A10G` latency block completed and saved:
  - `results/modal_latency/qwen3_1p7b_gptq_baselines/R3_Q17B__bs1`
  - `results/modal_latency/qwen3_1p7b_gptq_baselines/R3_Q17B__bs8`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2B03_Q17B__bs1`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2B03_Q17B__bs8`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2R02_Q17B__bs1`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2R02_Q17B__bs8`
- generated summary table now exists:
  - `results/analysis/latency_item4_summary.csv`
- canonical Item 4 report now exists:
  - `docs/experiments/item4_latency_analysis.md`
- current interpretation:
  - `batch=1`: bits is near baseline while rank is much slower
  - `batch=8`: policy ordering differs by model scale; throughput conclusions are workload-dependent
  - peak VRAM differs substantially by policy and should be reported alongside throughput
- Item 4 execution status:
  - all measurement runs complete
  - canonical analysis write-up completed in `docs/experiments/item4_latency_analysis.md`

### Runs

- [x] Measure tokens/second for Qwen3-1.7B GPTQ on A10G:
  - Baseline 4-bit (batch=1, batch=8)
  - Best bits-only (batch=1, batch=8)
  - Best rank-only (batch=1, batch=8)
- [x] Measure tokens/second for Qwen3-8B GPTQ on A100:
  - Same three configs
- [x] Measure peak VRAM for `Qwen3-8B GPTQ` configs
- [x] Measure peak VRAM for each `Qwen3-1.7B GPTQ` config

Partial completion:

- [x] full `Qwen3-8B GPTQ` block at batch `1` and `8`
- [x] full `Qwen3-1.7B GPTQ` block at batch `1` and `8`

### Analysis

- [x] Report latency overhead of rank repair as a percentage
- [x] Write the practical guidance: "rank repair adds X% latency; use when VRAM is the binding constraint"

---

## 5. Gain-Per-Byte Curves + Layer-Type Heatmaps (~3 days) — NICE-TO-HAVE

**Why**: The proposal called these "a particularly strong result." You already have the data — this is mostly plotting, not new experiments. Can be done in parallel with must-have items.

### Data Extraction

- [ ] Write a script `scripts/build_gain_curves.py` that:
  - reads `actions.json` from each completed transfer run
  - reconstructs the greedy allocation sequence (byte cost → cumulative ΔPPL)
  - outputs a CSV: `[model, quantizer, policy, action_index, cumulative_bytes, cumulative_delta_ppl, layer_name, layer_type, action_type]`
- [ ] Run the script across all completed transfer runs

### Plots

- [ ] **Gain-per-byte curves** for each model/quantizer:
  - X-axis: cumulative extra bytes
  - Y-axis: cumulative ΔPPL (improvement over baseline)
  - Two lines: bits-only allocator, rank-only allocator
  - Mark the crossover point (if any)
- [ ] **Layer-type breakdown**:
  - Stacked bar or heatmap showing what fraction of budget goes to each layer family
  - Families: `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`, `mlp.gate_proj`, `up_proj`, `down_proj`
- [ ] **Layerwise error heatmaps** (2 panels):
  - Panel A: activation-space error by layer index
  - Panel B: which layers were selected by the greedy allocator
- [ ] Save all plots to `docs/experiments/assets/`

### Analysis

- [ ] Identify if there's a consistent "crossover budget" where rank becomes better than bits (or vice versa)
- [ ] Identify if specific layer families always dominate the allocation (practical heuristic)
- [ ] Write the analysis section: "MLP.down_proj and attn.o_proj account for X% of all selected actions"

---

## 6. Paper Writing + Plots (~1.5 weeks, parallel with items 3-5) — MUST-HAVE

### Structure

- [x] Draft abstract (regime map + downstream + allocation signal finding)
- [x] Section 1: Introduction (the marginal-byte question)
- [x] Section 2: Related work (GPTQ, AWQ, LoftQ, SqueezeLLM, QuIP#)
- [x] Section 3: Method (allocation framework, action space, greedy policy)
- [x] Section 4: Experimental setup (models, quantizers, budgets, evaluation protocol)
- [x] Section 5: Results (regime map table + downstream table + gain-per-byte curves if ready)
- [x] Section 6: Analysis (activation ablation, layer patterns, latency)
- [x] Section 7: Discussion + practical guidelines
- [x] Appendix: full run inventory, infrastructure notes

### Key Figures

- [x] Figure 1: Regime map summary table (RTN + GPTQ × 4 scales, with downstream columns)
- [x] Figure 2: Activation vs weight-space allocator comparison (layer-selection diff + PPL delta)
- [x] Figure 3: Policy ordering at 1.7B and 8B (bar chart with error bars from multi-seed)
- [ ] Figure 4: Gain-per-byte curves (if Item 5 done; bits vs rank, colored by layer type)
- [ ] Figure 5: Layer-type allocation heatmaps (if Item 5 done)

---

## 7. 3-Bit Regime Test (~3 days) — OPTIONAL

**Why**: The proposal highlighted 3-bit as the key aggressive regime. However, the paper does not hinge on aggressive quantization — the 4-bit regime map is already a complete contribution. Only attempt after all must-have items are done.

### Runs

- [ ] Qwen3-1.7B RTN 3-bit baseline
- [ ] Qwen3-1.7B RTN 3-bit targeted bits (+1%)
- [ ] Qwen3-1.7B RTN 3-bit targeted rank (+1%)
- [ ] Qwen3-1.7B GPTQ 3-bit baseline (if GPTQ supports 3-bit in current backend)
- [ ] Qwen3-1.7B GPTQ 3-bit targeted bits (+1%)
- [ ] Qwen3-1.7B GPTQ 3-bit targeted rank (+1%)

### Analysis

- [ ] Compare 3-bit regime map against 4-bit regime map at same scale
- [ ] Test hypothesis: "rank should win more often at 3-bit because residuals are larger and more structured"
- [ ] If rank wins at 3-bit where it lost at 4-bit → new regime boundary discovered

---

## 8. Larger Model / Architecture Transfer Validation (~1 week) — OPTIONAL

**Why**: Validates whether the regime map generalizes beyond the current model set. Two distinct goals exist here — choose based on available budget.

### Option A: Scale Extension (14B+)

Tests whether "bits win at scale" continues or saturates.

- [ ] Select model: Qwen2.5-14B (same architecture family, larger scale)
- [ ] Adapt infrastructure for 14B (offload paths, A100-80GB)
- [ ] Run GPTQ 4-bit baseline
- [ ] Run targeted bits (+1%) and targeted rank (+1%)
- [ ] Add to the regime map table

### Option B: Architecture Transfer (different model family at 8B)

Tests whether the regime map is architecture-dependent or universal.

- [ ] Select model: Llama-3.1-8B (different tokenizer, architecture, training data)
- [ ] Run GPTQ 4-bit baseline
- [ ] Run targeted bits (+1%) and targeted rank (+1%)
- [ ] Compare against Qwen3-8B results at same scale

---

## Progress Tracker

| Item | Tier | Status | Started | Completed | Notes |
|------|------|--------|---------|-----------|-------|
| 1. Downstream eval | Must-have | ✅ Complete | 2026-03-14 | 2026-03-15 | Collection and analysis complete across GPTQ (`1.7B`, `3B`, `8B`) and RTN `1.7B`; see `docs/experiments/downstream_item1_analysis.md` |
| 2. Activation ablation | Must-have | ✅ Complete | 2026-03-14 | 2026-03-16 | Weight proxy wins; see `docs/experiments/activation_vs_weight_ablation.md` |
| 3. Multi-seed | Must-have | ✅ Complete | 2026-03-16 | 2026-03-16 | 6 runs completed (2 policies × 3 seeds); rank > bits is within noise |
| 4. Latency | Must-have | ✅ Complete | 2026-03-19 | 2026-03-19 | Full 12-run matrix complete; see `docs/experiments/item4_latency_analysis.md` and `results/analysis/latency_item4_summary.csv` |
| 5. Gain-per-byte curves | Nice-to-have | ⬜ Not started | | | |
| 6. Paper writing | Must-have | 🟨 In progress | 2026-03-19 | | Working manuscript draft now lives in `docs/paper/paper_draft.md`; paper assets for Figures 1-3 and the latency tables live in `docs/experiments/assets/`; remaining work is citation fill-in and final venue-specific polish |
| 7. 3-bit regime | Optional | ⬜ Not started | | | |
| 8. Larger model | Optional | ⬜ Not started | | | |

## Hard Stop Rule

> [!CAUTION]
> **The paper is submittable once items 1–4 and 6 are complete.** Items 5, 7, 8 improve the paper but are not required. Do not let optional items delay the MVP submission timeline.
