# Qwen3 1.7B GPTQ Plan

## Objective

Port the completed `1.7B` transfer comparison from `RTN` to `GPTQ` on Modal in order to test whether the bits-vs-rank result is quantizer-dependent.

## Run Order

1. Run the GPTQ baseline:
   - `R3_Q17B`
2. Build a GPTQ-specific candidate pool from the baseline `layer_errors.json`
3. Run the first matched transfer pair:
   - `G2B02_Q17B`
   - `G2R02_Q17B`
4. Only if that first pair is ambiguous, run:
   - `G2B03_Q17B`
   - `G2R03_Q17B`

## Commands

### Baseline

```bash
MODAL_GPU=A100 ./scripts/run_qwen3_1p7b_r3_gptq_modal.sh
```

### Build candidate pool

```bash
source /home/seshu/anaconda3/etc/profile.d/conda.sh
conda activate rl
python scripts/build_candidate_pool.py \
  --layer-errors results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B/layer_errors.json \
  --output configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_top12_candidate_pool.json \
  --model-name Qwen/Qwen3-1.7B-Base \
  --base-run-id R3_Q17B \
  --source-layer-errors-path results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B/layer_errors.json \
  --pool-name qwen3_1p7b_gptq_top12_candidate_pool \
  --quantizer GPTQ \
  --bit-width 4 \
  --group-size 128 \
  --symmetric true \
  --top-k 12
```

### First matched pair

```bash
MODAL_GPU=A100 ./scripts/run_qwen3_1p7b_g2b02_modal.sh
MODAL_GPU=A100 ./scripts/run_qwen3_1p7b_g2r02_modal.sh
```

## Notes

- The GPTQ Modal path uses a dedicated runner image that installs `optimum` and `gptqmodel`.
- The first target is `A100` for stability; downgrade later only if the baseline and first pair are stable.
- The targeted bits and targeted rank runs reuse the existing action-selection framework but build their candidate pool from GPTQ baseline damage, not RTN damage.
- Modal bring-up required two image fixes:
  - split `gptqmodel` into a separate install step after `torch`
  - add `pip`, `setuptools`, and `wheel` explicitly so `gptqmodel` metadata generation can find `bdist_wheel`

## Current Bring-up Status

- One GPTQ baseline completed end-to-end but is not scientifically usable:
  - `R3_Q17B`
  - perplexity was catastrophically high (`16103625.67`)
  - the resulting baseline should not be used for transfer comparisons
- Follow-up Modal bring-up work fixed several code-path issues:
  - forced GPTQ runs toward `float16`
  - separated FP reference profiling from the in-place GPTQ source model
  - added detached execution and results-volume fetch support
  - changed the wrapper to invoke the remote function directly
- Even after those fixes, the full corrected baseline still has not produced a trustworthy committed artifact.
- Current status should be treated as: `GPTQ bring-up paused / blocked`.

See:

- `docs/experiments/qwen3_1p7b_gptq_bringup_status.md`

## Immediate Next Step

Do **not** proceed to the full GPTQ matched pair yet.

Instead:

1. add a small GPTQ smoke baseline
2. verify sane perplexity under the corrected `float16` path
3. only then rerun the full `R3_Q17B`
4. only after a trustworthy GPTQ baseline exists, resume:
   - `G2B02_Q17B`
   - `G2R02_Q17B`

Pause point:

- the latest `R3_Q17B` Modal wrapper stayed alive in a sleeping state for hours
- refreshed local artifacts still showed the old broken baseline
- next session should start by checking for and stopping any stale live Modal app from the dashboard before another rerun
