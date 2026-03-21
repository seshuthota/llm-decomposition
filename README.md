# llm-decomposition

`llm-decomposition` is an experiment harness for one bounded research question:

> After post-training quantization, if you have a small extra memory budget, should you spend it on more bits, low-rank residual repair, or a hybrid of both?

The repo contains the code, configs, experiment outputs, analysis notes, and paper-draft assets used to answer that question across `RTN` and `GPTQ`, multiple model scales, multi-seed runs, downstream evaluation, and latency benchmarking.

## What This Repo Does

The project compares three corrective policy families under matched post-quantization budgets:

- targeted bits: upgrade selected matrices to higher precision
- targeted rank: add low-rank residual repairs to selected matrices
- limited hybrids: combine both within a fixed byte budget

The key point is not to build a new quantizer. The point is to understand which corrective action is the better use of a constrained memory budget, and when that answer changes with:

- quantizer family
- model scale
- downstream task behavior
- serving workload

## Current Status

As of `2026-03-21`, the main paper-readiness experiment set is complete:

- Item 1: downstream evaluation is complete
- Item 2: activation-vs-weight proxy ablation is complete
- Item 3: multi-seed stability is complete
- Item 4: latency and peak-VRAM benchmarking is complete
- Item 6: manuscript draft and paper assets are in place, with citation filling still pending

High-level empirical picture so far:

- `RTN` is regime-dependent across scale
- `GPTQ` is also regime-dependent, but not in the same way as `RTN`
- `8B GPTQ` favors targeted bits over targeted rank on both quality and low-batch latency
- `1.7B GPTQ` is more favorable to targeted rank on quality, while latency depends on workload

For the canonical synthesis, start with [docs/experiments/final_quantization_vs_svd_synthesis.md](docs/experiments/final_quantization_vs_svd_synthesis.md).

## Start Here

If you are new to the repo, use this reading order:

1. [docs/README.md](docs/README.md)
2. [docs/experiments/final_quantization_vs_svd_synthesis.md](docs/experiments/final_quantization_vs_svd_synthesis.md)
3. [docs/paper/paper_draft.md](docs/paper/paper_draft.md)
4. [docs/experiments/item3_multiseed_analysis.md](docs/experiments/item3_multiseed_analysis.md)
5. [docs/experiments/item4_latency_analysis.md](docs/experiments/item4_latency_analysis.md)

If you want the planning view instead of the narrative view, start with [docs/roadmap/paper_readiness_plan.md](docs/roadmap/paper_readiness_plan.md).

## Repo Layout

Core code:

- `llm_decomposition/`: experiment harness, quantization/repair logic, evaluation backends, profiling, latency benchmarking
- `scripts/`: local runners, Modal wrappers, summary builders, asset generators
- `configs/`: experiment definitions and manifests

Documentation:

- `docs/experiments/`: canonical analyses and synthesis notes
- `docs/roadmap/`: execution plans, blockers, and progress trackers
- `docs/paper/`: manuscript draft and review feedback
- `docs/reference/`: harness and config notes

Outputs:

- `results/modal/`: canonical remote experiment outputs for main RTN/GPTQ runs
- `results/modal_importfix_probe_v2/`: canonical saved `Qwen3-8B GPTQ` multiseed recovery runs
- `results/modal_latency/`: canonical Item 4 latency benchmark runs
- `results/analysis/`: generated summary tables used by the docs and paper assets

## Main Documents

- [docs/experiments/final_quantization_vs_svd_synthesis.md](docs/experiments/final_quantization_vs_svd_synthesis.md): project-level synthesis
- [docs/experiments/downstream_item1_analysis.md](docs/experiments/downstream_item1_analysis.md): downstream results
- [docs/experiments/activation_vs_weight_ablation.md](docs/experiments/activation_vs_weight_ablation.md): allocator proxy ablation
- [docs/experiments/item3_multiseed_analysis.md](docs/experiments/item3_multiseed_analysis.md): multi-seed stability
- [docs/experiments/item4_latency_analysis.md](docs/experiments/item4_latency_analysis.md): latency and peak-VRAM analysis
- [docs/paper/paper_draft.md](docs/paper/paper_draft.md): current manuscript draft

## Quick Start

Use the `rl` Conda environment:

```bash
conda activate rl
```

Validate a manifest without running models:

```bash
python scripts/run_manifest.py \
  --manifest configs/phase2/phase2_matched_frontier_manifest.json \
  --dry-run
```

Prepare run directories only:

```bash
python scripts/run_manifest.py \
  --manifest configs/phase2/phase2_matched_frontier_manifest.json \
  --prepare-only
```

Run one experiment from a manifest:

```bash
python scripts/run_manifest.py \
  --manifest configs/phase2/phase2_matched_frontier_manifest.json \
  --run-id P2R03
```

Run a remote Modal experiment:

```bash
./scripts/run_modal_experiment_rl.sh \
  P2R03 \
  configs/phase2/phase2_matched_frontier_manifest.json \
  qwen3-0.6b-base
```

## How Experiments Are Organized

The repo evolved in phases:

- `phase1`: baseline quantization and early repair experiments
- `phase2`: matched-budget targeted bits vs targeted rank
- `phase3`: scale-up and regime mapping
- `multiseed`: stability checks for paper claims
- `downstream`: task-level evaluation
- `latency`: serving-oriented decode and VRAM benchmarking

The most important current configs live under:

- `configs/phase2/`
- `configs/scaleup_1p7b_gptq/`
- `configs/scaleup_smollm3_3b_gptq/`
- `configs/scaleup_qwen3_8b_gptq/`
- `configs/multiseed/`
- `configs/latency/`

## Notes For Readers

- `results/` contains saved artifacts and generated summaries; it is not hand-maintained prose
- `docs/` is the curated interpretation layer
- some roadmap files are historical planning records rather than the current project state
- the cleanest single-entry documentation hub is [docs/README.md](docs/README.md)
