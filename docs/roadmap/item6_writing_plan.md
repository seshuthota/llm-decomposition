# Item 6 Writing Plan

This document translates the completed paper-readiness experiments into a concrete writing order for the paper draft.

Current status:

- Items `1`, `2`, `3`, and `4` are complete
- Item `5` remains optional
- the paper is no longer bottlenecked by experiments or core drafting
- the main remaining work is citation insertion and final venue-specific polish

## Writing Goal

Produce a submission-ready draft that uses the completed evidence to support one clean thesis:

under a fixed post-quantization memory budget, the better use of the next byte depends on quantizer, model scale, and deployment workload; neither extra bits nor low-rank repair is universally dominant.

Canonical draft location:

- `docs/paper/paper_draft.md`

## Recommended Draft Order

### 1. Results section first

Start with the sections whose evidence is now fixed:

- cross-quantizer regime-map result
- Item 1 downstream result
- Item 3 multi-seed stability result
- Item 4 latency result

This gives the paper its core empirical spine before spending time on framing and prose polish.

### 2. Analysis section second

Once the results section is written, add the analysis sections that explain why the practical recommendation is conditional:

- Item 2 activation-vs-weight allocator result
- Item 4 workload-dependent latency interpretation
- practical guidance by scale / workload

### 3. Introduction and abstract after the evidence is written

Write these only after the results and analysis language is stable. That reduces churn and keeps the claims tightly coupled to the finished evidence.

## Section-to-Evidence Map

### Abstract

Use:

- `docs/experiments/final_quantization_vs_svd_synthesis.md`
- `docs/experiments/downstream_item1_analysis.md`
- `docs/experiments/item3_multiseed_analysis.md`
- `docs/experiments/item4_latency_analysis.md`

Required claims:

- the marginal-byte question
- regime dependence across quantizer / scale
- downstream does not always mirror perplexity
- latency and VRAM make deployment recommendations workload-dependent

### Introduction

Core framing:

- the user does not choose quantization in the abstract
- the user chooses how to spend a small extra memory budget after quantization
- that extra budget can buy:
  - more bits
  - low-rank repair
  - hybrid repair

Suggested source docs:

- `docs/proposals/current_proposal.md`
- `docs/experiments/final_quantization_vs_svd_synthesis.md`

### Method

Cover:

- action-space framing
- greedy allocation under a byte budget
- bits-only / rank-only / hybrid policies
- proxy families for allocation
- evaluation protocol

Suggested source docs:

- `docs/reference/config_harness.md`
- `docs/reference/phase2_action_schema.md`
- `llm_decomposition/hf_backend.py`
- `llm_decomposition/methods.py`

### Experimental Setup

Cover:

- RTN and GPTQ
- model scales
- budget policy
- WikiText calibration / eval
- downstream setup
- latency setup

Suggested source docs:

- `docs/experiments/downstream_item1_analysis.md`
- `docs/experiments/item3_multiseed_analysis.md`
- `docs/experiments/item4_latency_analysis.md`

### Results

Subsections:

- regime map across RTN and GPTQ
- downstream validation
- multi-seed stability
- latency and VRAM tradeoffs

Canonical sources:

- `docs/experiments/final_quantization_vs_svd_synthesis.md`
- `docs/experiments/downstream_item1_analysis.md`
- `docs/experiments/item3_multiseed_analysis.md`
- `docs/experiments/item4_latency_analysis.md`

### Analysis

Subsections:

- activation-space vs weight-space allocation
- where the conclusions are stable vs within noise
- workload-sensitive deployment guidance

Canonical sources:

- `docs/experiments/activation_vs_weight_ablation.md`
- `docs/experiments/item3_multiseed_analysis.md`
- `docs/experiments/item4_latency_analysis.md`

### Discussion

Focus:

- bits vs rank is not a single global ordering
- downstream, perplexity, latency, and VRAM each constrain the recommendation differently
- practical guidance should be scale-aware and workload-aware

## Required Figures / Tables

Must-have first:

1. Regime map summary table
2. Multi-seed error-bar figure
3. Latency + peak-VRAM comparison table
4. Activation-vs-weight ablation figure/table

Optional if time permits:

1. Gain-per-byte curves
2. Layer-type heatmaps

## Immediate Writing Tasks

1. [x] Draft `docs/paper/paper_draft.md` with Abstract, Introduction, Method, Experimental Setup, Results, and Analysis sections
2. [x] Tighten manuscript prose from working draft into paper-draft style
3. [x] Turn Item `3` and Item `4` summary tables into paper figure/table assets:
   - `docs/experiments/assets/table_figure1_regime_map_summary.md`
   - `docs/experiments/assets/figure_item3_multiseed_errorbars.svg`
   - `docs/experiments/assets/table_item2_activation_weight_ablation.md`
   - `docs/experiments/assets/item3_multiseed_errorbar.csv`
   - `docs/experiments/assets/item4_latency_table.csv`
   - `docs/experiments/assets/item4_latency_overheads.csv`
   - `docs/experiments/assets/table_item4_latency.md`
   - `docs/experiments/assets/table_item4_latency_overheads.md`
4. [x] Add a citation-independent abstract and introduction polish pass
