# Qwen3 1.7B Modal Plan

## Objective

Test whether the Phase 2 allocator result transfers from `Qwen/Qwen3-0.6B-Base` to `Qwen/Qwen3-1.7B-Base` using the Modal-backed execution path.

The question is not "rerun everything." The question is:

- does targeted rank still beat targeted bits at matched budgets on a larger model?

## Current Status

Completed:

- `R2_Q17B` baseline
- `P2B02_Q17B` targeted bits at `+1.0%`
- `P2R02_Q17B` targeted rank at `+1.0%`
- `P2B03_Q17B` targeted bits at `+2.0%`
- `P2R03_Q17B` targeted rank at `+2.0%`

Current comparison:

- `R2_Q17B`: perplexity `21.3102`
- `P2B02_Q17B`: perplexity `21.2415`
- `P2R02_Q17B`: perplexity `21.3001`
- `P2B03_Q17B`: perplexity `21.1505`
- `P2R03_Q17B`: perplexity `21.2971`

Current interpretation:

- at both tested `1.7B` transfer points, targeted bits beat targeted rank
- this does not match the earlier `0.6B` local RTN result
- the transfer question is now resolved for the current `RTN` + matrix-level action setup:
  - `0.6B` favored targeted rank
  - `1.7B` favored targeted bits

## Execution Order

1. Cache the `1.7B` model locally.
2. Upload the cached snapshot to the Modal Volume.
3. Run the `RTN 4-bit` baseline `R2_Q17B`.
4. Build the shared top-12 candidate pool from `R2_Q17B` layer errors.
5. Run the first meaningful matched pair:
   - `P2B02_Q17B`
   - `P2R02_Q17B`
6. Analyze the completed second budget pair:
   - `P2B03_Q17B`
   - `P2R03_Q17B`

## Why This Order

- `R2_Q17B` is the anchor for all later memory budgets.
- The candidate pool should come from actual `1.7B` activation damage, not from the `0.6B` pool.
- The `+1.0%` pair is the first real transfer test.
- The `+2.0%` pair confirmed that the `1.7B` bits advantage is stable across the tested budget range.

## Commands

### 1. Cache locally

```bash
./scripts/cache_qwen3_1p7b_base_rl.sh
```

### 2. Upload to Modal

```bash
./scripts/modal_upload_qwen3_1p7b_base_rl.sh
```

### 3. Run the baseline on Modal

```bash
./scripts/run_qwen3_1p7b_r2_modal.sh
```

### 4. Build the candidate pool

```bash
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl
python scripts/build_candidate_pool.py \
  --layer-errors results/modal/qwen3_1p7b_baselines/R2_Q17B/layer_errors.json \
  --output configs/scaleup_1p7b/qwen3_1p7b_r2_top12_candidate_pool.json \
  --model-name Qwen/Qwen3-1.7B-Base \
  --base-run-id R2_Q17B \
  --source-layer-errors-path results/modal/qwen3_1p7b_baselines/R2_Q17B/layer_errors.json \
  --pool-name qwen3_1p7b_r2_top12_matrix_pool \
  --top-k 12 \
  --control-layer model.layers.0.mlp.down_proj \
  --control-layer model.layers.0.self_attn.o_proj
```

### 5. Run the first matched pair

```bash
./scripts/run_qwen3_1p7b_p2b02_modal.sh
./scripts/run_qwen3_1p7b_p2r02_modal.sh
```

### 6. Second matched pair result

- `P2B03_Q17B`: completed on `A100` after an `A10G` stall, perplexity `21.1505`
- `P2R03_Q17B`: completed on `A100`, perplexity `21.2971`

## Success Criteria

- `R2_Q17B` completes on Modal without OOM.
- `results/modal/qwen3_1p7b_baselines/R2_Q17B/` contains valid `metrics.json` and `layer_errors.json`.
- the `+1.0%` pair completes and gives a clear bits-vs-rank comparison.
- the `+2.0%` pair confirmed that the `1.7B` bits advantage is stable rather than a one-budget effect.

## Notes

- `T4` is acceptable for the `R2_Q17B` baseline.
- `A10G` is the default for the `1.7B` transfer runs.
- `A100` was the pragmatic recovery path for the heavier `+2.0%` runs when `A10G` stopped being reliable.
