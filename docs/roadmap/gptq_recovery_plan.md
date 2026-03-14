# GPTQ Recovery Plan

This file defines the next implementation path for `GPTQ` after repeated Modal and Kaggle runs reached execution but still produced invalid numerics.

## Current Diagnosis

The repo is past the old setup blockers:

- `gptqmodel` can be installed
- GPTQ smoke runs execute
- artifacts are written
- layer summaries are populated

But the current baseline path is still not scientifically usable because evaluation outputs are numerically invalid:

- Modal full baseline once produced absurd perplexity
- Kaggle smoke baselines completed with `perplexity = NaN`

So the blocker is no longer environment bring-up. The blocker is the current GPTQ backend/runtime path.

## Goal

Replace repeated blind reruns with a controlled backend recovery sequence that can answer one question cleanly:

> Can this repo produce a trustworthy finite-perplexity GPTQ baseline at all?

## Implementation Strategy

### 1. Add a second GPTQ implementation path

Keep the old direct `optimum.gptq.GPTQQuantizer` path available, but stop treating it as the default.

Add a new default implementation using the documented Transformers integration:

- build `GPTQConfig`
- call `AutoModelForCausalLM.from_pretrained(..., quantization_config=gptq_config)`

This keeps the project inside the officially documented Hugging Face path and makes it easier to distinguish:

- backend/runtime instability
- versus bugs in the repo’s older direct quantizer flow

### 2. Fail fast on non-finite outputs

Do not allow GPTQ runs to silently write `NaN` perplexity as if they were valid completed experiments.

Before the main perplexity loop:

- run a tiny validation forward pass
- check whether `loss` and `logits` are finite
- write a debug artifact
- abort the run early if values are not finite

This turns the GPTQ track from “numerically broken but marked completed” into an explicit validation failure.

### 3. Recover in three stages

Run GPTQ in this order:

1. tiny smoke validation on a very small evaluation sample
2. full smoke validation on the same target model
3. only then full baseline

Do not proceed to transfer runs until the full baseline has finite perplexity.

## Execution Order

### Stage A: backend validation

1. run tiny smoke config
2. inspect finite-output debug artifact
3. if non-finite, stop and debug backend

### Stage B: baseline validation

1. run full `R3_Q17B`
2. confirm:
   - finite perplexity
   - plausible memory
   - layer summaries across real transformer layers

### Stage C: only after a valid baseline

1. build GPTQ candidate pool
2. run `G2B02_Q17B`
3. run `G2R02_Q17B`

## Decision Rules

- If the new backend still fails finite-output validation on smoke runs:
  - stop retrying full baselines
  - treat the current GPTQ backend path as blocked
- If smoke succeeds but the full baseline fails:
  - debug scale/model-specific instability
- If the full baseline succeeds:
  - resume the paused GPTQ transfer plan

## Scope Guard

This plan is for restoring a trustworthy GPTQ baseline.

It is not yet:

- a hybrid experiment plan
- an ExLlama serving migration
- a vLLM benchmark path
- a broad quantizer bake-off

Those only become relevant after the baseline itself is valid.
