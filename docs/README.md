# Documentation Map

This repo now keeps planning and experiment notes in one place instead of spreading them across root-level markdown files.

## Start Here

- [proposals/current_proposal.md](proposals/current_proposal.md): active research framing
- [roadmap/execution_roadmap.md](roadmap/execution_roadmap.md): current execution plan, evaluation setup, and next milestones
- [roadmap/phase2_plan.md](roadmap/phase2_plan.md): detailed next-phase execution document
- [roadmap/phase2_remaining_checklist.md](roadmap/phase2_remaining_checklist.md): compact finish-line checklist for Phase 2
- [roadmap/qwen3_1p7b_modal_plan.md](roadmap/qwen3_1p7b_modal_plan.md): next scale-up path on Modal for `Qwen/Qwen3-1.7B-Base`
- [roadmap/qwen3_1p7b_gptq_plan.md](roadmap/qwen3_1p7b_gptq_plan.md): GPTQ transfer path on Modal for `Qwen/Qwen3-1.7B-Base`
- [roadmap/next_steps.md](roadmap/next_steps.md): current restart plan and blocker summary
- [experiments/phase1_results.md](experiments/phase1_results.md): consolidated Phase 1 results and project decision
- [experiments/experiment_journal.md](experiments/experiment_journal.md): detailed run-by-run log
- [experiments/phase2_working_summary.md](experiments/phase2_working_summary.md): live Phase 2 comparison sheet
- [experiments/phase2_conclusion.md](experiments/phase2_conclusion.md): final local Phase 2 decision and next-step recommendation
- [experiments/qwen3_1p7b_transfer_conclusion.md](experiments/qwen3_1p7b_transfer_conclusion.md): final `1.7B` Modal transfer result

## Proposal History

- [proposals/proposal_history.md](proposals/proposal_history.md): how the project framing changed
- [proposals/initial_proposal.md](proposals/initial_proposal.md): first full proposal draft
- [archive/original_idea_notes.txt](archive/original_idea_notes.txt): original seed notes and literature summary

## Reference

- [reference/config_harness.md](reference/config_harness.md): config-driven experiment harness notes
- [../results/phase1/phase1_summary.md](../results/phase1/phase1_summary.md): generated phase summary from run outputs

## Current Status

As of `2026-03-13`:

- `R1`, `R2`, and `R4-R8`, `R10-R11` are complete on `Qwen/Qwen3-0.6B-Base`
- `R3` GPTQ is deferred to another machine because the current GPTQ dependency path requires newer GPU architecture than the local `GTX 1660 SUPER`
- Phase 1 established that uniform low-rank repair helps at higher rank, but the more important question is targeted bits vs targeted rank under matched budgets
- local RTN Phase 2 is complete: targeted rank beat the current matrix-level targeted bits frontier at both `+1.0%` and `+2.0%` budget points
- the `Qwen/Qwen3-1.7B-Base` Modal transfer study is also complete: targeted bits beat targeted rank at both `+1.0%` and `+2.0%`
- the active next step is to resume `GPTQ` from a smaller smoke baseline, not from the stale full baseline
