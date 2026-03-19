# Results

This directory stores raw experiment outputs, generated summaries, and execution logs.

Do not hand-edit canonical run directories in this tree. Treat them as saved artifacts.

## Layout

- `analysis/`: generated cross-run summary tables used by the docs and paper draft
- `modal/`: canonical remote experiment outputs for RTN/GPTQ/downstream runs
- `modal_importfix_probe_v2/`: saved canonical `Qwen3-8B GPTQ` multiseed recovery runs
- `modal_latency/`: canonical Item 4 latency benchmark artifacts
- `downstream/`: local downstream evaluation exports
- `logs/`: execution and dependency logs
- `phase1/`, `phase2/`: original local run outputs for earlier phases

## Canonical Reading Order

For interpretation, use the curated docs first:

1. [../docs/experiments/final_quantization_vs_svd_synthesis.md](../docs/experiments/final_quantization_vs_svd_synthesis.md)
2. [../docs/experiments/item3_multiseed_analysis.md](../docs/experiments/item3_multiseed_analysis.md)
3. [../docs/experiments/item4_latency_analysis.md](../docs/experiments/item4_latency_analysis.md)
4. [analysis/multiseed_stability_all_summary.csv](analysis/multiseed_stability_all_summary.csv)
5. [analysis/latency_item4_summary.csv](analysis/latency_item4_summary.csv)

## Canonical Artifact Families

- `results/modal_importfix_probe_v2/...`: authoritative saved `Qwen3-8B GPTQ` multiseed runs
- `results/modal_latency/...`: authoritative latency benchmark runs
- `results/modal/...`: authoritative remote run outputs for the main experiment families

The `results/` tree is the raw source of truth for saved run outputs. The `docs/` tree is the curated narrative layer built on top of those outputs.
