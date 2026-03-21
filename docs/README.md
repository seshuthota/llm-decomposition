# Documentation Map

This directory is the curated documentation layer for the project. Use it for the research narrative, experiment interpretation, planning history, and paper draft. Use `results/` for raw saved artifacts.

## Start Here

If you want the shortest path to understanding the project:

1. [experiments/final_quantization_vs_svd_synthesis.md](experiments/final_quantization_vs_svd_synthesis.md)
2. [paper/paper_draft.md](paper/paper_draft.md)
3. [experiments/item3_multiseed_analysis.md](experiments/item3_multiseed_analysis.md)
4. [experiments/item4_latency_analysis.md](experiments/item4_latency_analysis.md)
5. [roadmap/paper_readiness_plan.md](roadmap/paper_readiness_plan.md)

## Main Sections

### Experiments

- [experiments/README.md](experiments/README.md): index of canonical experiment analyses
- [experiments/final_quantization_vs_svd_synthesis.md](experiments/final_quantization_vs_svd_synthesis.md): project-level synthesis across RTN and GPTQ
- [experiments/downstream_item1_analysis.md](experiments/downstream_item1_analysis.md): downstream evaluation analysis
- [experiments/activation_vs_weight_ablation.md](experiments/activation_vs_weight_ablation.md): allocator proxy ablation
- [experiments/item3_multiseed_analysis.md](experiments/item3_multiseed_analysis.md): multi-seed stability analysis
- [experiments/item4_latency_analysis.md](experiments/item4_latency_analysis.md): latency and peak-VRAM analysis
- [experiments/experiment_journal.md](experiments/experiment_journal.md): detailed historical run log

### Paper

- [paper/paper_draft.md](paper/paper_draft.md): current manuscript draft
- [paper/draft_feedback.md](paper/draft_feedback.md): reviewer-style feedback and revision notes

### Roadmaps

- [roadmap/paper_readiness_plan.md](roadmap/paper_readiness_plan.md): main status tracker for the paper-ready experiment set
- [roadmap/item4_latency_measurement_plan.md](roadmap/item4_latency_measurement_plan.md): detailed latency execution plan
- [roadmap/item6_writing_plan.md](roadmap/item6_writing_plan.md): writing and asset-generation plan
- [roadmap/execution_roadmap.md](roadmap/execution_roadmap.md): broader execution roadmap

### Proposals

- [proposals/current_proposal.md](proposals/current_proposal.md): active research framing
- [proposals/proposal_history.md](proposals/proposal_history.md): framing changes over time
- [proposals/initial_proposal.md](proposals/initial_proposal.md): original proposal draft

### Reference

- [reference/config_harness.md](reference/config_harness.md): config-driven harness notes
- [reference/modal_runner.md](reference/modal_runner.md): Modal execution notes
- [reference/phase2_action_schema.md](reference/phase2_action_schema.md): action-schema reference

### Archive

- [archive/README.md](archive/README.md): archived notes, feedback, and superseded planning documents

## Current Status

As of `2026-03-21`:

- the main paper-readiness experiment block is complete through downstream, ablation, multiseed, and latency
- the manuscript draft exists and the main paper assets are generated
- the remaining core writing task is citation completion and final polish

## Notes

- treat `docs/experiments/` as the canonical interpretation layer
- treat `docs/roadmap/` as a mix of active trackers and preserved planning history
- treat `docs/archive/` as provenance, not current project state
