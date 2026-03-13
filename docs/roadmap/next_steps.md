# Next Steps

This file is the current restart point for the project.

## Stable Results

- Local `RTN` work is complete.
- `Qwen/Qwen3-0.6B-Base` under local `RTN` favored targeted rank over the current matrix-level targeted bits frontier.
- `Qwen/Qwen3-1.7B-Base` under Modal `RTN` favored targeted bits over targeted rank at both tested budgets.

These results are already documented and should be treated as trustworthy.

## Current Blocker

The active blocker is the `GPTQ` transfer track on `Qwen/Qwen3-1.7B-Base`.

Current status:

- the first full GPTQ baseline completed but was invalid
- later reruns did not produce a trustworthy replacement baseline
- the repo should currently treat GPTQ as `paused / bring-up blocked`

Canonical references:

- `docs/experiments/qwen3_1p7b_gptq_bringup_status.md`
- `docs/roadmap/qwen3_1p7b_gptq_plan.md`
- `docs/experiments/experiment_journal.md`

## Immediate Resume Plan

When work resumes, do this in order:

1. Verify and stop any stale live GPTQ Modal app from the Modal dashboard.
2. Do not trust the current `results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B` metrics as a valid baseline.
3. Add and run a true GPTQ smoke baseline:
   - smaller calibration sample
   - smaller evaluation sample
   - no profiling
   - explicit `float16` path
4. Confirm the smoke run produces plausible perplexity and clean artifacts.
5. Only then rerun the full GPTQ baseline `R3_Q17B`.
6. Only after a valid GPTQ baseline exists:
   - build `qwen3_1p7b_gptq_top12_candidate_pool.json`
   - run `G2B02_Q17B`
   - run `G2R02_Q17B`

## Decision Rule After GPTQ Resumes

If GPTQ baseline becomes stable:

- continue the `GPTQ` transfer comparison first
- do not start unrelated new experiment branches until `G2B02_Q17B` and `G2R02_Q17B` are resolved

If GPTQ remains blocked after a smoke baseline:

- stop the full GPTQ track for now
- document the backend limitation explicitly
- optionally revisit the unimplemented hybrid second-stage path only after choosing whether that is worth the implementation cost

## Practical Summary

The project is not blocked scientifically on RTN; it is blocked operationally on GPTQ bring-up.

The next session should start from GPTQ smoke validation, not from new research branching.
