# Documentation Map

This repo now keeps planning and experiment notes in one place instead of spreading them across root-level markdown files.

## Start Here

- [experiments/final_quantization_vs_svd_synthesis.md](experiments/final_quantization_vs_svd_synthesis.md): canonical project-level synthesis across RTN and GPTQ
- [proposals/current_proposal.md](proposals/current_proposal.md): active research framing
- [roadmap/execution_roadmap.md](roadmap/execution_roadmap.md): current execution plan, evaluation setup, and next milestones
- [roadmap/phase2_plan.md](roadmap/phase2_plan.md): detailed next-phase execution document
- [roadmap/phase2_remaining_checklist.md](roadmap/phase2_remaining_checklist.md): compact finish-line checklist for Phase 2
- [roadmap/phase3_plan.md](roadmap/phase3_plan.md): canonical Phase 3 regime-mapping plan
- [roadmap/qwen3_1p7b_modal_plan.md](roadmap/qwen3_1p7b_modal_plan.md): next scale-up path on Modal for `Qwen/Qwen3-1.7B-Base`
- [roadmap/qwen3_1p7b_gptq_plan.md](roadmap/qwen3_1p7b_gptq_plan.md): GPTQ transfer path on Modal for `Qwen/Qwen3-1.7B-Base`
- [roadmap/gptq_recovery_plan.md](roadmap/gptq_recovery_plan.md): backend-recovery plan for restoring a trustworthy GPTQ baseline
- [roadmap/gptq_richer_action_space_plan.md](roadmap/gptq_richer_action_space_plan.md): next GPTQ implementation branch after the valid matrix-level baselines
- [roadmap/next_steps.md](roadmap/next_steps.md): current restart plan and blocker summary
- [roadmap/kaggle_plan.md](roadmap/kaggle_plan.md): fresh Kaggle notebook setup and execution plan
- [experiments/phase1_results.md](experiments/phase1_results.md): consolidated Phase 1 results and project decision
- [experiments/experiment_journal.md](experiments/experiment_journal.md): detailed run-by-run log
- [experiments/phase2_working_summary.md](experiments/phase2_working_summary.md): live Phase 2 comparison sheet
- [experiments/phase2_conclusion.md](experiments/phase2_conclusion.md): final local Phase 2 decision and next-step recommendation
- [experiments/qwen3_1p7b_transfer_conclusion.md](experiments/qwen3_1p7b_transfer_conclusion.md): final `1.7B` Modal transfer result
- [experiments/phase3_rtn_regime_map.md](experiments/phase3_rtn_regime_map.md): final Phase 3 `RTN` cross-scale result
- [experiments/gptq_pre_final_branch_snapshot.md](experiments/gptq_pre_final_branch_snapshot.md): frozen GPTQ frontier before the bounded endgame stop

## Proposal History

- [proposals/proposal_history.md](proposals/proposal_history.md): how the project framing changed
- [proposals/initial_proposal.md](proposals/initial_proposal.md): first full proposal draft
- [archive/README.md](archive/README.md): archived source notes, feedback, and superseded drafts
- [archive/original_idea_notes.txt](archive/original_idea_notes.txt): original seed notes and literature summary

## Reference

- [reference/config_harness.md](reference/config_harness.md): config-driven experiment harness notes
- [../results/phase1/phase1_summary.md](../results/phase1/phase1_summary.md): generated phase summary from run outputs

## Current Status

As of `2026-03-14`:

- `R1`, `R2`, and `R4-R8`, `R10-R11` are complete on `Qwen/Qwen3-0.6B-Base`
- Phase 1 established that uniform low-rank repair helps at higher rank, but the more important question is targeted bits vs targeted rank under matched budgets
- local RTN Phase 2 is complete: targeted rank beat the current matrix-level targeted bits frontier at both `+1.0%` and `+2.0%` budget points
- the `Qwen/Qwen3-1.7B-Base` Modal transfer study is also complete: targeted bits beat targeted rank at both `+1.0%` and `+2.0%`
- Phase 3 RTN expansion is complete: `0.6B` favored targeted rank, while `1.7B`, `3B`, and `8B` all favored targeted bits
- GPTQ is now operational on Modal and the matrix-level validation set is complete:
  - `1.7B`: rank wins
  - `3B`: neither helps; bits regress less than rank
  - `8B`: bits win
- the bounded GPTQ endgame branch is also complete:
  - the final `1.7B` multi-bit bits gate did not change the ordering
  - GPTQ is ready for synthesis / write-up rather than more runs
