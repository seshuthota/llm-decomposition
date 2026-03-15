# Paper-Readiness Plan: Budget-Aware Bits-vs-Rank Allocation

**Created**: 2026-03-14  
**Updated**: 2026-03-15  
**Target**: Strong main-track submission (MLSys / ICLR / NeurIPS)  
**Estimated duration**: 3–4 weeks  
**Status**: In progress — Item 1 downstream evaluation and analysis are complete; activation-vs-weight ablation is next

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

- [ ] Verify the `proxy_family` config parameter already supports both `"activation"` and `"weight"` paths in the allocator
  - Check `_build_bit_actions()` and `_build_rank_actions()` in `hf_backend.py`
  - If weight-space path is missing, implement it using `relative_fro_error` from `layer_errors.json` as the proxy signal
- [ ] Create two matched config pairs for Qwen3-1.7B GPTQ at +1% budget:
  - `g2b02_weight_proxy_Q17B.json` — bits allocation using weight-space signal
  - `g2r02_weight_proxy_Q17B.json` — rank allocation using weight-space signal

### Runs

- [ ] Run bits-only with weight-space proxy on 1.7B GPTQ (`G2B02W_Q17B`)
- [ ] Run rank-only with weight-space proxy on 1.7B GPTQ (`G2R02W_Q17B`)
- [ ] Run bits-only with weight-space proxy on 8B GPTQ (`G2B02W_Q8B`) — if budget allows
- [ ] Run rank-only with weight-space proxy on 8B GPTQ (`G2R02W_Q8B`) — if budget allows

### Analysis

- [ ] Compare PPL: activation-allocated vs weight-allocated at same budget
- [ ] Compare which layers are selected by each allocator (produce a layer-selection diff table)
- [ ] Measure profiling time: activation-space vs weight-space (report cost-benefit)
- [ ] Write conclusion: "activation-space profiling is / is not justified by its allocation quality"

---

## 3. Multi-Seed / Calibration Resampling (~2 days) — MUST-HAVE

**Why**: The closest-call results (1.7B GPTQ: 15.8823 vs 15.8914, 8B GPTQ: 11.7823 vs 11.7962) have margins of ~0.01 PPL. Reviewers will question if these are noise. This is more publication-critical than extra visualizations.

### Implementation

- [ ] Modify the calibration data loading to accept a `calib_seed` parameter that shuffles which WikiText-2 training samples are chosen
- [ ] Create configs for 3 seeds (seed 42, 123, 456) on the two closest-call results

### Runs

- [ ] Qwen3-1.7B GPTQ `G2R02_Q17B` × 3 seeds
- [ ] Qwen3-1.7B GPTQ `G2B03_Q17B` × 3 seeds
- [ ] Qwen3-8B GPTQ `G2B02_Q8B` × 3 seeds (if budget allows)
- [ ] Qwen3-8B GPTQ `G2R02_Q8B` × 3 seeds (if budget allows)

### Analysis

- [ ] Report mean ± std for each policy at each scale
- [ ] Confirm the rank > bits ordering at 1.7B and bits > rank at 8B hold across seeds
- [ ] If ordering is unstable → report it honestly as "within noise" and adjust claims

---

## 4. Latency Measurement (~2 days) — MUST-HAVE

**Why**: Residual branches add extra matmuls. A practical user needs to know the latency cost of choosing rank over bits.

### Runs

- [ ] Measure tokens/second for Qwen3-1.7B GPTQ on A10G:
  - Baseline 4-bit (batch=1, batch=8)
  - Best bits-only (batch=1, batch=8)
  - Best rank-only (batch=1, batch=8)
- [ ] Measure tokens/second for Qwen3-8B GPTQ on A100:
  - Same three configs
- [ ] Measure peak VRAM for each config

### Analysis

- [ ] Report latency overhead of rank repair as a percentage
- [ ] Write the practical guidance: "rank repair adds X% latency; use when VRAM is the binding constraint"

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

- [ ] Draft abstract (regime map + downstream + allocation signal finding)
- [ ] Section 1: Introduction (the marginal-byte question)
- [ ] Section 2: Related work (GPTQ, AWQ, LoftQ, SqueezeLLM, QuIP#)
- [ ] Section 3: Method (allocation framework, action space, greedy policy)
- [ ] Section 4: Experimental setup (models, quantizers, budgets, evaluation protocol)
- [ ] Section 5: Results (regime map table + downstream table + gain-per-byte curves if ready)
- [ ] Section 6: Analysis (activation ablation, layer patterns, latency)
- [ ] Section 7: Discussion + practical guidelines
- [ ] Appendix: full run inventory, infrastructure notes

### Key Figures

- [ ] Figure 1: Regime map summary table (RTN + GPTQ × 4 scales, with downstream columns)
- [ ] Figure 2: Activation vs weight-space allocator comparison (layer-selection diff + PPL delta)
- [ ] Figure 3: Policy ordering at 1.7B and 8B (bar chart with error bars from multi-seed)
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
| 2. Activation ablation | Must-have | ⬜ Not started | | | |
| 3. Multi-seed | Must-have | ⬜ Not started | | | |
| 4. Latency | Must-have | ⬜ Not started | | | |
| 5. Gain-per-byte curves | Nice-to-have | ⬜ Not started | | | |
| 6. Paper writing | Must-have | ⬜ Not started | | | |
| 7. 3-bit regime | Optional | ⬜ Not started | | | |
| 8. Larger model | Optional | ⬜ Not started | | | |

## Hard Stop Rule

> [!CAUTION]
> **The paper is submittable once items 1–4 and 6 are complete.** Items 5, 7, 8 improve the paper but are not required. Do not let optional items delay the MVP submission timeline.
