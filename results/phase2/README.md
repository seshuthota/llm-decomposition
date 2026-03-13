# Phase 2 Results

This directory stores the raw outputs for Phase 2 allocator experiments.

## Scope

Phase 2 is the first fair frontier comparison on top of `R2`:

- targeted bits
- targeted rank
- optional later hybrid second-stage runs

All Phase 2 runs stay on:

- model: `Qwen/Qwen3-0.6B-Base`
- quantizer: `RTN`
- base anchor: `R2`

## Main References

- [run_index.md](run_index.md)
- [../../docs/experiments/phase2_working_summary.md](../../docs/experiments/phase2_working_summary.md)
- [../../docs/experiments/phase2_conclusion.md](../../docs/experiments/phase2_conclusion.md)

Phase 2 is complete for the current local `RTN` setup. The current conclusion is that targeted rank beat the current matrix-level targeted bits frontier at both matched budget points that mattered.

## Manifests

- first-pass manifest: `configs/phase2/phase2_first_pass_manifest.json`
- matched-frontier manifest: `configs/phase2/phase2_matched_frontier_manifest.json`

## Completed Runs

- `P2B01`: matrix-level bits, uniform policy, `+0.25%`
- `P2B02`: matrix-level bits, activation greedy, `+1.0%`
- `P2R01`: matrix-level rank, uniform policy, `+0.25%`
- `P2R02`: matrix-level rank, activation greedy, `+1.0%`
- `P2B03`: matrix-level bits, activation greedy, `+2.0%`
- `P2R03`: matrix-level rank, activation greedy, `+2.0%`

## Shared Inputs

- candidate pool: `configs/phase2/qwen3_0p6b_r2_top12_candidate_pool.json`
- action schema: `configs/phase2/phase2_action_schema.json`
- human-readable schema reference: `docs/reference/phase2_action_schema.md`

## Expected Run Contents

Each run folder should contain:

- `resolved_config.json`
- `execution_status.json`
- `metrics.json`
- `layer_errors.json`
- `residual_profiles.json`
- `notes.md`
