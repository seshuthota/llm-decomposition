# Qwen3 1.7B GPTQ Bring-up Status

## Current Status

The GPTQ transfer phase is **not ready to proceed** to `G2B02_Q17B` or `G2R02_Q17B`.

The RTN transfer work is complete and trustworthy, but the GPTQ baseline on `Qwen/Qwen3-1.7B-Base` is still blocked by a bring-up issue.

## What Was Attempted

### 1. First end-to-end GPTQ baseline

- Run: `R3_Q17B`
- Modal app: `ap-NYaV7jQSaTk8CM8gINsYc3`
- Outcome: completed and wrote artifacts

Observed result:

- perplexity: `16103625.670221116`
- dtype in metrics: `bfloat16`
- `layer_errors.json` was not usable for a real candidate pool

Interpretation:

- this was not a fair GPTQ baseline
- the quantized model was numerically broken enough that no transfer experiment should be built on top of it

### 2. Attached rerun after GPTQ code fixes

- Modal app: `ap-FxF9LMfb1deIOUNy2u0b1U`
- Outcome: app stayed live for a long run but stopped without updating the results volume

Observed issue:

- Modal logs ended with `Runner terminated.`
- no trustworthy new baseline artifacts were committed

### 3. Detached rerun through the script local entrypoint

- Modal app: `ap-MfYDty8wketjhOSu3p2i9L`
- Outcome: detached execution avoided client disconnect, but still ended with `Runner terminated`

Observed issue:

- the stale baseline artifacts remained in `results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B`
- the new run did not commit an updated baseline

### 4. Detached rerun calling the remote function directly

- Modal app: `ap-cLA6emAAfjUg29tsW2Jk6e`
- Outcome: the run used the corrected config inside the live container, but the final committed artifacts still did not replace the stale baseline

What was confirmed inside the live container:

- `resolved_config.json` showed `dtype_preference: ["float16"]`
- `metrics.json` in the live run directory stayed `pending` during execution
- the app eventually stopped, but the persisted artifacts still reflected the old broken baseline

Interpretation:

- the corrected config is reaching the live run directory
- but the run is still not producing a trustworthy committed baseline artifact
- the current blocker is now a combination of:
  - GPTQ numerical validity under the current backend path
  - and/or termination before final artifact commit

## Code Changes Already Made

- forced GPTQ CUDA runs toward `float16` instead of the general `bfloat16` preference path
- added `hf_device_map` before packing so GPTQ quantizer selection could proceed
- separated the baseline into:
  - a full-precision reference model for layer stats
  - a separate GPTQ source model for in-place quantization
- changed the GPTQ Modal wrapper to:
  - support detached execution
  - fetch results from a Modal volume
  - invoke the remote function directly instead of depending on the local entrypoint
- fixed the detached results path so GPTQ runs now write into the mounted results volume
  instead of the container-local filesystem
- added a smaller smoke baseline `R3S_Q17B` for `Qwen/Qwen3-1.7B-Base`
  with:
  - 16 calibration texts
  - 16 evaluation sequences
  - no activation profiling
  - no residual profiling

## What Is Safe To Conclude Now

- `RTN` conclusions remain valid
- `GPTQ` transfer conclusions are **not** ready
- the current `R3_Q17B` artifacts in `results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B` should still be treated as stale / not scientifically usable

## Recommended Next Debug Step

The next step should be a **small GPTQ smoke baseline** before another full `WikiText-2` pass.

Recommended smoke run changes:

- same model: `Qwen/Qwen3-1.7B-Base`
- same quantizer: `GPTQ 4-bit`
- much smaller evaluation:
  - fewer calibration texts
  - fewer evaluation sequences
  - no profiling
- explicit runtime logging written incrementally to disk

Success criterion:

- perplexity is at least in a plausible language-model range
- committed artifacts reflect the corrected `float16` config

Only after that smoke run is sane should the project return to the full `R3_Q17B` baseline and then the `G2B02_Q17B` / `G2R02_Q17B` transfer pair.

## Pause Point

As of the latest pause:

- the local GPTQ Modal wrapper for `R3_Q17B` remained alive for multiple hours in a sleeping state
- local artifacts were refreshed again on `2026-03-13 11:31:49 IST`
- but the refreshed files still contained the same invalid baseline:
  - perplexity `16103625.67`
  - `dtype: bfloat16`
  - `layer_errors.json` containing only `lm_head`

This means the GPTQ track should be treated as **paused and unresolved**, not as a completed transfer phase.

## Recommended Resume Plan

When resuming this track:

1. Verify no stale Modal GPTQ app is still running in the dashboard.
2. Do **not** resume from the current `R3_Q17B` artifacts.
3. Add and run a true smoke config:
   - much smaller calibration sample
   - much smaller evaluation sample
   - no profiling
   - explicit `float16` in config
4. Confirm that the smoke run lands a plausible perplexity and trustworthy local artifacts.
5. Only then rerun the full GPTQ baseline:
   - `R3_Q17B`
6. After a valid GPTQ baseline exists:
   - build `qwen3_1p7b_gptq_top12_candidate_pool.json`
   - run `G2B02_Q17B`
   - run `G2R02_Q17B`

## Safe Current Conclusion

- RTN conclusions are complete and trustworthy.
- GPTQ conclusions are **not ready**.
- The repo should currently be read as:
  - `RTN`: complete through transfer
  - `GPTQ`: bring-up paused, baseline not yet valid
