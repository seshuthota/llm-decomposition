# Phase 3 RTN Regime Map

This document closes the active `RTN` branch of Phase 3.

## Scope

Phase 3 asked how the bits-vs-rank decision changes with model scale under the current `RTN 4-bit` plus matrix-level action setup.

Trusted scale points:

- local `Qwen/Qwen3-0.6B-Base`
- Modal `Qwen/Qwen3-1.7B-Base`
- Modal `HuggingFaceTB/SmolLM3-3B-Base`
- Modal `Qwen/Qwen3-8B-Base`

## Final Results

### `0.6B`

- baseline `R2`: perplexity `30.5169`
- targeted bits `R11`-style matched comparator became meaningful only at the largest tested uniform-repair budget
- targeted rank won the Phase 2 local frontier under the current setup

Interpretation:

- the smallest tested model still supported the targeted-rank story

### `1.7B`

- baseline `R2_Q17B`: perplexity `21.3102`
- targeted bits `P2B02_Q17B`: perplexity `21.2415`
- targeted rank `P2R02_Q17B`: perplexity `21.3001`
- targeted bits `P2B03_Q17B`: perplexity `21.1505`
- targeted rank `P2R03_Q17B`: perplexity `21.2971`

Interpretation:

- targeted bits beat targeted rank at both tested budget points

### `3B`

- baseline `R2_S3B`: perplexity `47.9169`
- targeted bits `P3B02_S3B`: perplexity `47.4955`
- targeted rank `P3R02_S3B`: perplexity `47.9833`

Interpretation:

- targeted bits won clearly at the first matched budget point
- the `+2.0%` pair was skipped because the ordering was already stable

### `8B`

- baseline `R2_Q8B`: perplexity `16.1939`
- targeted bits `P3B02_Q8B`: perplexity `16.1429`
- targeted rank `P3R02_Q8B`: perplexity `16.2035`

Interpretation:

- targeted bits again won clearly at the first matched budget point
- the `+2.0%` pair was skipped because the first pair was decisive

## Main Conclusion

Under the current `RTN 4-bit` plus matrix-level action setup, the project now has a clear scale-dependent picture:

- `0.6B`: targeted rank wins
- `1.7B`: targeted bits win
- `3B`: targeted bits win
- `8B`: targeted bits win

So the apparent crossover happens between the smallest tested model and the larger regime.

The strongest current statement is not that one method wins universally. It is:

- targeted rank can win in the very small-model regime
- targeted bits dominate once the model is large enough under the current action space and allocator

## Operational Notes

Phase 3 also produced two infrastructure outcomes:

- `SmolLM3-3B` was practical on `A10G` after RTN vectorization
- `Qwen3-8B` required:
  - CPU-side RTN working tensors during quantization
  - sequential model offload during activation profiling

Those changes made the `8B` baseline and matched pair feasible on `A100` without moving to a larger GPU tier.

## What This Leaves Open

- `GPTQ` is still blocked as a numerically unstable backend path
- hybrid second-stage testing is still untested at larger scale
- richer bit action spaces are still untested

So the natural next question is no longer “does Phase 3 RTN finish?” It already has. The next question is whether:

- `GPTQ` changes the regime map,
- or hybrid second-stage allocation adds anything after the winning bit upgrades have already been taken.
