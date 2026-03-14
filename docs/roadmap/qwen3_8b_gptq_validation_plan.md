# Qwen3-8B GPTQ Validation Plan

## Goal

Use one larger GPTQ model to test whether the `1.7B` and `3B` GPTQ results generalize.

## Current State

The `8B` GPTQ path is now valid on Modal under:

- `A100-80GB`
- `device_map: "single"`

Validated checkpoints:

- smoke baseline `R3S_Q8B`
- full baseline `R3_Q8B`
- first matched transfer pair:
  - `G2B02_Q8B`
  - `G2R02_Q8B`

Current conclusion:

- the real fix was not just “use more VRAM”
- the working path is: real `A100-80GB` request plus single-device placement
- at the first matched `8B` GPTQ point, targeted bits beat targeted rank

## Cost-Aware Execution Order

1. Stage `Qwen/Qwen3-8B-Base` to the Modal model volume.
2. Run `R3S_Q8B` smoke baseline on `A100-80GB` with `device_map: "single"`.
3. Run `R3_Q8B` on the same setup.
4. Build `qwen3_8b_gptq_top12_candidate_pool.json` from `R3_Q8B/layer_errors.json`.
5. Run the first matched pair on `A100-80GB`:
   - `G2B02_Q8B`
   - `G2R02_Q8B`
6. Only add higher-budget points if the first pair is too close to call.

## GPU Choice

- `A100-80GB` is the current smallest proven GPU for `8B` GPTQ in this repo.
- Do not start on `A10G`; the `3B` GPTQ transfer path already showed that GPTQ transfer overhead grows enough to make smaller cards a bad bet.
- Avoid `device_map: "auto"` for this branch; the working path is `device_map: "single"`.

## Resume Rule

Next `8B` GPTQ work should be one of:

1. add the `+2.0%` matched pair if more frontier resolution is needed, or
2. move to richer GPTQ action spaces if the current matrix-level rank path remains too weak
