# Qwen3-8B GPTQ Multiseed Blocker

This note records the exact issues encountered while trying to complete Item 3 (`8B` GPTQ multiseed stability) and the recommended next actions.

Status update:

- this blocker is now resolved as of `2026-03-19`
- the completed recovery and final multiseed outcome are summarized in `docs/experiments/item3_multiseed_analysis.md`
- keep this file as a historical debugging record of the failure modes and fixes

## Scope

Blocked runs:

- `G2B02_Q8B_s42`
- `G2B02_Q8B_s123`
- `G2B02_Q8B_s456`
- `G2R02_Q8B_s42`
- `G2R02_Q8B_s123`
- `G2R02_Q8B_s456`

The blocker applies specifically to the `Qwen/Qwen3-8B-Base` GPTQ multiseed branch on Modal.

## What Is Already Working

These `8B` GPTQ runs were completed successfully earlier in the project:

- `R3_Q8B`
- `G2B02_Q8B`
- `G2R02_Q8B`
- `H2R02_Q8B`

Those successful configs used:

- `implementation: "transformers_gptq_config"`
- `device_map: "single"`
- `desc_act: true`
- `act_group_aware: false`
- `A100-80GB`

Relevant configs:

- `configs/scaleup_qwen3_8b_gptq/r3_qwen3_8b_gptq_4bit.json`
- `configs/scaleup_qwen3_8b_gptq/g2b02_qwen3_8b_gptq_bits_activation_1p0pct.json`
- `configs/scaleup_qwen3_8b_gptq/g2r02_qwen3_8b_gptq_rank_activation_1p0pct.json`
- `configs/scaleup_qwen3_8b_gptq/h2r02_qwen3_8b_gptq_hybrid_activation_1p0pct.json`

## Issues Encountered

### 1. OOM on smaller GPUs

Earlier `8B` attempts failed on smaller GPUs because the build path temporarily held both the full-precision reference model and the GPTQ model in memory.

Mitigation:

- moved the reference model to CPU during the build phase
- standardized `8B` runs on `A100-80GB`

This fixed the original VRAM bottleneck, but it was not the final blocker.

### 2. Unnecessary full-model serialization in the GPTQ hot path

The old GPTQ path serialized the whole quantized model with `save_pretrained(...)` only to estimate artifact size.

That was a bad fit for `8B`.

Mitigation already applied:

- removed hot-path `save_pretrained(...)`-based byte estimation
- switched to analytical byte accounting in:
  - `llm_decomposition/gptq_backend.py`
  - `llm_decomposition/hf_backend.py`

This was a real issue, but removing it did not fully unblock `8B` multiseed.

### 3. Oversized Modal repo mount

The original Modal GPTQ runner was mounting too much local repo state, including large result trees and fetch directories.

Mitigation already applied:

- trimmed the mount ignore list in `scripts/modal_experiment_gptq.py`
- reduced mount size substantially

This improved startup hygiene, but it was not the root cause of the `8B` failures.

### 4. Missing `8B` model in the Modal model volume

The multiseed branch initially failed because `/vol/models/qwen3-8b-base` was not present.

This was confirmed by the diagnostic preflight stage.

Mitigation already applied:

- restaged `Qwen/Qwen3-8B-Base` into the Modal model volume
- verified presence of:
  - `config.json`
  - tokenizer files
  - all safetensor shards

This fixed one hard prerequisite, but did not resolve the full runtime issue.

### 5. Remote budget lookup could not find `R3_Q8B`

The multiseed configs depend on `base_run_id = R3_Q8B` for budget resolution.

The remote code path failed because:

- `_load_target_memory_bytes(...)` did not originally search the `qwen3_8b_gptq_*` directories
- the Modal runner mount also excluded `results/`, so depending on historical result trees inside the remote mount was fragile

Mitigations already applied:

- added `8B` GPTQ result directories to `_load_target_memory_bytes(...)`
- added explicit `budget_bytes` to all six `8B` multiseed configs

This removed the remote dependency on old result-tree discovery.

### 6. `gptqmodel` removed: Optimum/Transformers GPTQ fails at runtime

When `gptqmodel` was removed from the Modal image to avoid earlier crashes, the `8B` run advanced farther but failed with:

- `NameError: QuantizeConfig is not defined`

This occurred inside:

- `transformers` GPTQ quantizer path
- via `optimum.gptq.quantizer.GPTQQuantizer`

Interpretation:

- `transformers_gptq_config` on this stack still has a hard runtime dependency on `gptqmodel`
- removing `gptqmodel` is therefore not compatible with the current `8B` path

### 7. `gptqmodel` restored: early Modal termination returns

After restoring `gptqmodel` to match the previously successful `8B` environment, fresh retry runs terminated early again.

Observed behavior:

- Modal app/container shows `Terminated`
- no normal local artifact bundle is written
- no Python traceback is surfaced through the normal local path
- this resembles the earlier hard-failure behavior seen with exit code `132`

Interpretation:

- current `8B` multiseed runs appear to be trapped between two incompatible states:
  - without `gptqmodel`: Optimum GPTQ fails at runtime
  - with `gptqmodel`: the container can terminate early before normal artifact packaging

## Current Root Problem

The current `8B` multiseed blocker is an environment/runtime compatibility conflict:

- `transformers_gptq_config` for `8B` still expects `gptqmodel`
- but enabling `gptqmodel` in the current Modal GPTQ image appears to reintroduce early termination / hard native failure

So the issue is no longer:

- model staging
- budget resolution
- or simple result writing

It is now a deeper compatibility problem in the `8B` Modal GPTQ stack.

## Recommended Next Actions

### Recommended mainline choice

De-scope `8B` multiseed from the paper MVP and continue the paper-readiness plan with:

- `1.7B` multiseed complete
- `3B` multiseed complete
- `8B` multiseed marked as blocked by runtime incompatibility

Then proceed to:

- Item 4: latency
- tables / figures / write-up

Why:

- this preserves momentum
- avoids open-ended infra work inside the paper critical path
- still leaves Item 3 with valid multi-seed evidence at two scales

### If `8B` multiseed must be recovered

Treat it as a separate runtime-recovery branch.

Best recovery path:

1. Build a dedicated `8B` Modal image just for this branch.
2. Reproduce the previously successful `8B` package stack as closely as possible.
3. Avoid mixing in newer diagnostic/cleanup changes one by one without a fresh baseline.
4. Validate a single fresh retry run before launching all six seeds.

This should be treated as:

- a separate infrastructure effort
- not the main paper-readiness path

## Bottom Line

Current status:

- `1.7B` multiseed: complete
- `3B` multiseed: complete
- `8B` multiseed: blocked

Recommended interpretation for the roadmap:

> `8B` GPTQ multiseed stability is currently blocked by a Modal runtime compatibility conflict between the working `8B` GPTQ stack and the current multiseed execution image. The paper MVP should proceed with `1.7B` and `3B` multi-seed evidence, while `8B` multiseed is tracked as an optional recovery branch rather than a blocking requirement.
