# Phase 2 Remaining Checklist

This checklist was the finish-line tracker for local RTN Phase 2. It is now complete.

## Current State

Completed:

- `P2B01`: bits, `+0.25%`, too small for matrix-level actions
- `P2B02`: bits, `+1.0%`, first meaningful bits-only point
- `P2R02`: rank, `+1.0%`, corrected incremental-rank allocator, beats `P2B02`
- `P2B03`: bits, `+2.0%`, stronger bits frontier point
- `P2R03`: rank, `+2.0%`, confirms the rank advantage and shows early saturation

Closed:

- `P2R01` was not needed to close the phase
- wrap-up analysis is captured in [../experiments/phase2_conclusion.md](../experiments/phase2_conclusion.md)

## Completion Check

The required Phase 2 closure conditions are now satisfied:

- `P2B02`, `P2R02`, `P2B03`, and `P2R03` are complete and reviewed
- the bits-vs-rank RTN conclusion is written in [../experiments/phase2_conclusion.md](../experiments/phase2_conclusion.md)
- layer-type payoff has been summarized there as well
- there is an explicit decision not to block on a hybrid second-stage local RTN test

## What Comes After Phase 2

The next phase should be one of:

- scale the stabilized allocator logic to `Qwen/Qwen3-1.7B-Base`
- or transfer the same framework to `GPTQ` on another machine

Do not keep adding local Phase 2 variants unless a new quantizer or larger-model result reopens the question.
