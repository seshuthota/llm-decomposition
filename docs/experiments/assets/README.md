# Paper Assets

This directory holds paper-facing table and figure inputs derived from the completed paper-readiness experiments.

Current generated assets:

- `figure1_regime_map_summary.csv`
  - compact regime-map summary table input
- `table_figure1_regime_map_summary.md`
  - paper-facing regime-map summary table
- `figure_item3_multiseed_errorbars.png`
- `figure_item3_multiseed_errorbars.svg`
  - rendered multi-seed error-bar figure
- `item3_multiseed_errorbar.csv`
  - plotting input for the GPTQ multi-seed error-bar figure
  - columns: `scale`, `policy`, `n`, `mean_ppl`, `std_ppl`, `min_ppl`, `max_ppl`
- `item2_activation_weight_ablation.csv`
  - compact allocator-ablation source table
- `table_item2_activation_weight_ablation.md`
  - paper-facing activation-vs-weight ablation table
- `item4_latency_table.csv`
  - compact latency + peak-VRAM table for the main paper text
  - covers `Qwen3-1.7B / A10G` and `Qwen3-8B / A100` at batch sizes `1` and `8`
- `item4_latency_overheads.csv`
  - baseline-relative latency / throughput / VRAM deltas for the practical-guidance section
- `table_item4_latency.md`
  - paper-facing latency + peak-VRAM table
- `table_item4_latency_overheads.md`
  - baseline-relative latency/throughput/VRAM comparison table

Regeneration:

- script: `scripts/build_item6_assets.py`
- source tables:
  - `results/analysis/multiseed_stability_all_summary.csv`
  - `results/analysis/latency_item4_summary.csv`
