# Paper-Readiness Plan: Budget-Aware Bits-vs-Rank Allocation

**Created**: 2026-03-14  
**Target**: Strong main-track submission (MLSys / ICLR / NeurIPS)  
**Estimated duration**: 4–5 weeks  
**Status**: Not started

> [!IMPORTANT]
> This checklist is ordered by **impact-per-effort**. Items 1–3 are non-negotiable for any submission. Items 4–6 significantly strengthen. Items 7–8 are bonus.

---

## 1. Downstream Zero-Shot / Few-Shot Evaluation (~1.5 weeks)

**Why**: Perplexity-only is the #1 reviewer rejection reason for compression papers. Both proposals explicitly promised "compact downstream accuracy or zero-shot task performance."

### Infrastructure Setup

- [ ] Install `lm-eval-harness` in the `rl` conda environment
  ```bash
  conda run -n rl pip install lm-eval
  ```
- [ ] Write a wrapper script `scripts/run_downstream_eval.py` that:
  - accepts a model path or HF model name + a quantization config
  - applies the quantization + repair pipeline from `hf_backend.py`
  - runs `lm_eval.evaluator.simple_evaluate()` on the resulting model
  - saves results JSON to `results/<run_dir>/downstream_metrics.json`
- [ ] Verify the wrapper produces valid results on the full-precision Qwen3-1.7B (sanity check)
- [ ] Create a Modal runner `scripts/modal_downstream_eval.py` (matching existing Modal patterns)

### Task Selection

- [ ] Select 6 standard tasks for evaluation:
  - `hellaswag` (0-shot)
  - `arc_easy` (0-shot)
  - `arc_challenge` (25-shot)
  - `winogrande` (5-shot)
  - `piqa` (0-shot)
  - `boolq` (0-shot)
- [ ] Verify all tasks run on the full-precision model before proceeding

### Evaluation Runs

Run downstream eval for each regime point that matters most:

#### Qwen3-1.7B GPTQ (rank-wins regime)

- [ ] Full-precision reference
- [ ] GPTQ 4-bit baseline (`R3_Q17B`)
- [ ] Best bits-only (`G2B03_Q17B`)
- [ ] Best rank-only (`G2R02_Q17B`)
- [ ] Best hybrid (`H2R02M_Q17B`)

#### Qwen3-8B GPTQ (bits-wins regime)

- [ ] GPTQ 4-bit baseline (`R3_Q8B`)
- [ ] Best bits-only (`G2B02_Q8B`)
- [ ] Best rank-only (`G2R02_Q8B`)
- [ ] Best hybrid (`H2R02_Q8B`)

#### SmolLM3-3B GPTQ (neutral regime)

- [ ] GPTQ 4-bit baseline (`R3_S3B`)
- [ ] Best bits-only (`G3B02_S3B`)
- [ ] Best rank-only (`G3R02_S3B`)

### Analysis

- [ ] Build a consolidated downstream results table (model × policy × task)
- [ ] Check whether downstream Δaccuracy correlates with ΔPPL across all regime points
- [ ] Compute "quality recovered per added MB" in downstream accuracy units
- [ ] Write a summary section for the paper: does the regime map hold beyond perplexity?

---

## 2. Activation-Space vs Weight-Space Allocator Ablation (~3 days)

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

## 3. Gain-Per-Byte Curves + Layer-Type Heatmaps (~3 days)

**Why**: The proposal called these "a particularly strong result." You already have the data — this is mostly plotting, not new experiments.

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

## 4. Multi-Seed / Calibration Resampling (~2 days)

**Why**: The closest-call results (1.7B GPTQ: 15.8823 vs 15.8914, 8B GPTQ: 11.7823 vs 11.7962) have margins of ~0.01 PPL. Reviewers will question if these are noise.

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

## 5. 3-Bit Regime Test (~3 days)

**Why**: The proposal highlighted 3-bit as the key aggressive regime where "quantization error is large enough to matter but still structured enough to admit selective repair." All current experiments are 4-bit only.

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

## 6. Latency Measurement (~2 days)

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

## 7. One Larger Model Validation (~1 week, stretch goal)

**Why**: Validates whether "bits win at scale" continues at 14B+.

- [ ] Select model: Qwen2.5-14B or Llama-3.1-8B (different architecture)
- [ ] Adapt infrastructure for larger model (offload paths, A100-80GB)
- [ ] Run GPTQ 4-bit baseline
- [ ] Run targeted bits (+1%) and targeted rank (+1%)
- [ ] Add to the regime map table

---

## 8. Paper Writing + Plots (~1.5 weeks, parallel with items 5-7)

### Structure

- [ ] Draft abstract (regime map + downstream + allocation signal finding)
- [ ] Section 1: Introduction (the marginal-byte question)
- [ ] Section 2: Related work (GPTQ, AWQ, LoftQ, SqueezeLLM, QuIP#)
- [ ] Section 3: Method (allocation framework, action space, greedy policy)
- [ ] Section 4: Experimental setup (models, quantizers, budgets, evaluation protocol)
- [ ] Section 5: Results (regime map table + downstream table + gain-per-byte curves)
- [ ] Section 6: Analysis (activation ablation, layer patterns, 3-bit regime, latency)
- [ ] Section 7: Discussion + practical guidelines
- [ ] Appendix: full run inventory, infrastructure notes

### Key Figures

- [ ] Figure 1: Regime map summary table (RTN + GPTQ × 4 scales, with downstream)
- [ ] Figure 2: Gain-per-byte curves (bits vs rank, colored by layer type)
- [ ] Figure 3: Layer-type allocation heatmaps
- [ ] Figure 4: Activation vs weight-space allocator comparison
- [ ] Figure 5: Policy ordering at 1.7B and 8B (bar chart with error bars from multi-seed)

---

## Progress Tracker

| Item | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 1. Downstream eval | ⬜ Not started | | | |
| 2. Activation ablation | ⬜ Not started | | | |
| 3. Gain-per-byte curves | ⬜ Not started | | | |
| 4. Multi-seed | ⬜ Not started | | | |
| 5. 3-bit regime | ⬜ Not started | | | |
| 6. Latency | ⬜ Not started | | | |
| 7. Larger model | ⬜ Not started | | | |
| 8. Paper writing | ⬜ Not started | | | |
