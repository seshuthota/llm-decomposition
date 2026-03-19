# Experiments

Canonical experiment reports live here. Use this folder for human-written experiment summaries and conclusions; use `results/` for per-run artifacts.

## Project-Level Reports

- [final_quantization_vs_svd_synthesis.md](final_quantization_vs_svd_synthesis.md): canonical final synthesis across RTN and GPTQ
- [quantization_vs_svd_detailed_report.md](quantization_vs_svd_detailed_report.md): full phase-by-phase report with tables, equations, and plots
- [phase1_to_now_report.md](phase1_to_now_report.md): consolidated report from Phase 1 through the current GPTQ and RTN results
- [downstream_item1_analysis.md](downstream_item1_analysis.md): completed Item 1 downstream evaluation analysis for paper readiness
- [activation_vs_weight_ablation.md](activation_vs_weight_ablation.md): completed Item 2 allocator-proxy ablation for paper readiness
- [item3_multiseed_analysis.md](item3_multiseed_analysis.md): completed Item 3 multi-seed stability analysis for paper readiness
- [item4_latency_analysis.md](item4_latency_analysis.md): completed Item 4 latency and peak-VRAM analysis for paper readiness
- [experiment_journal.md](experiment_journal.md): detailed chronological run journal
- [gptq_pre_final_branch_snapshot.md](gptq_pre_final_branch_snapshot.md): frozen GPTQ frontier snapshot before the final bounded branch stop

## Phase Summaries

- [phase1_results.md](phase1_results.md): consolidated Phase 1 outcome and decision
- [phase2_conclusion.md](phase2_conclusion.md): Phase 2 closeout for the local RTN targeted-bits-vs-rank study
- [phase2_working_summary.md](phase2_working_summary.md): working comparison sheet for the main Phase 2 transition
- [phase3_rtn_regime_map.md](phase3_rtn_regime_map.md): final RTN cross-scale regime-map result

## Model/Quantizer-Specific Reports

- [qwen3_1p7b_transfer_conclusion.md](qwen3_1p7b_transfer_conclusion.md): final trusted `1.7B` RTN transfer result
- [qwen3_1p7b_gptq_bringup_status.md](qwen3_1p7b_gptq_bringup_status.md): GPTQ bring-up history, fixes, and current validated status

## Generated Summaries

- [../../results/phase1/phase1_summary.md](../../results/phase1/phase1_summary.md): generated Phase 1 summary from run metrics
- [assets/README.md](assets/README.md): paper-facing table and figure inputs generated from the completed analyses
