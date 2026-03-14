# Qwen3 1.7B GPTQ Bring-up Status

## Current Status

The GPTQ bring-up is now **successful enough to run the `1.7B` transfer study on Modal**.

The old failure modes were fixed in stages:

- injected `hf_device_map` before Optimum packing in the Transformers GPTQ path
- added finite-logit validation before perplexity evaluation
- stopped treating packed GPTQ tensors as directly writable when applying targeted updates
- replaced selected GPTQ layers with ordinary floating `nn.Linear` modules for:
  - targeted 8-bit upgrades
  - targeted low-rank repairs

The remaining caveat is scientific, not bring-up:

- the current GPTQ rank action space saturates the candidate pool by `+1.0%`
- so the `+2.0%` rank point does not move beyond the `+1.0%` rank point

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

### 5. Kaggle GPTQ smoke baseline

- Run: `R3S_Q17B`
- Platform: `Kaggle`
- Outcome: completed with valid artifacts, but invalid numerics

Observed result:

- dtype: `float16`
- perplexity: `NaN`
- layer summaries were populated across real layers

Interpretation:

- Kaggle solved the old package/build barrier well enough to execute the GPTQ path
- but the quantized model remained numerically unstable at evaluation time
- this means the current blocker is no longer just installation or artifact persistence
- the blocker is now confirmed to be GPTQ runtime numerical validity

### 6. Kaggle GPTQ second smoke baseline

- Run: `R3S2_Q17B`
- Platform: `Kaggle`
- Outcome: completed with safer settings, but still invalid numerics

Observed result:

- dtype: `float16`
- perplexity: `NaN`
- smaller evaluation still produced `NaN`

Interpretation:

- changing to a safer smoke configuration did not fix the evaluation instability
- so the current GPTQ backend path should be treated as blocked across both Modal and Kaggle

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
- the Kaggle smoke artifacts confirm that the problem is not just environment bring-up; GPTQ numerics are still unstable

## Recommended Next Debug Step

Do **not** retry the full GPTQ baseline immediately.

The smoke-baseline stage has now been attempted on Kaggle and still failed numerically.

So the next GPTQ work, when resumed, should be a backend-level debugging pass, not just another run retry. That likely means one of:

- changing GPTQ backend/library
- instrumenting the quantized model outputs to localize the first `NaN`
- or reducing the GPTQ path to a simpler internal validation before perplexity evaluation

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

## Final Bring-up Outcome

The following runs are now valid on Modal:

- `R3_Q17B`
- `G2B02_Q17B`
- `G2R02_Q17B`
- `G2B03_Q17B`
- `G2R03_Q17B`

Key metrics:

- `R3_Q17B`: perplexity `15.9137`
- `G2B02_Q17B`: perplexity `15.8993`
- `G2R02_Q17B`: perplexity `15.8823`
- `G2B03_Q17B`: perplexity `15.8914`
- `G2R03_Q17B`: perplexity `15.8823`

Interpretation:

- GPTQ transfer is now runnable and produces stable finite metrics
- targeted rank beats targeted bits at the first matched point
- the current rank frontier saturates by `+1.0%`, so the `+2.0%` rank point is unchanged

## Safe Current Conclusion

- RTN conclusions are complete and trustworthy.
- GPTQ bring-up is no longer blocked.
- GPTQ transfer conclusions are now available for the current `1.7B` matrix-level action space.
