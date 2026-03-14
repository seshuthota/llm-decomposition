# Phase 3 Plan

This is the canonical Phase 3 roadmap, derived from [archive/plans/phase_3_implementation_action_plan.md](../archive/plans/phase_3_implementation_action_plan.md).

## Purpose

Phase 3 is a regime-mapping phase.

The goal is to determine how the preferred use of marginal memory varies across:

- model scale
- quantizer type
- budget size
- allocation granularity

The central question remains:

> Under a fixed memory budget, should the next byte go to extra precision, low-rank residual repair, or a hybrid of both?

## What Phase 1 and Phase 2 Already Established

- local `0.6B` RTN favored targeted rank
- `1.7B` RTN transfer favored targeted bits
- the `1.7B` RTN result was reproduced in Kaggle
- GPTQ is still blocked at baseline bring-up, including Kaggle smoke attempts that completed with `NaN` perplexity

So Phase 3 should not assume a universal winner. It should map regimes.

## Primary Goals

1. Build a cross-scale frontier map for bits-only and rank-only allocation under RTN.
2. Test whether frontier ordering changes under a stronger quantizer such as GPTQ.
3. Determine whether hybrid second-stage repair adds value only after the best bit allocations are already taken.
4. Identify the variables that explain frontier changes:
   - layer family
   - residual structure
   - activation-space sensitivity
   - action-space granularity

## Non-Goals

Do not prioritize these unless Phase 3 finishes early:

- gradient-based residual fitting
- very large downstream benchmark suites
- many exotic quantizers
- 14B+ models as the first expansion step
- complex global optimization beyond greedy or simple budgeted search

## Phase 3 Structure

Phase 3 should run in three blocks:

1. Phase 3A: RTN regime map
2. Phase 3B: quantizer transfer
3. Phase 3C: hybrid second-stage testing

## Phase 3A: RTN Regime Map

### Objective

Extend the already working RTN setup across a larger model ladder and a richer action space.

### Recommended Model Ladder

- `Qwen/Qwen3-0.6B-Base`
- `Qwen/Qwen3-1.7B-Base`
- one `3B-4B` class model
- one `7B-8B` class model
- optional `14B` only if earlier results remain ambiguous

### Required Work

1. Freeze the evaluation harness and run metadata.
2. Add a standardized Phase 3 run schema and action schema.
3. Validate RTN baselines for the selected model ladder.
4. Expand bits action space beyond coarse matrix-only upgrades where needed.
5. Keep rank actions incremental and comparable.

### Deliverables

- one RTN baseline per selected model size
- one candidate/action inventory per selected model size
- one cross-scale comparison table for bits vs rank

## Phase 3B: Quantizer Transfer

### Objective

Repeat the key comparison under GPTQ once a valid GPTQ baseline exists.

### Current Reality

This block is currently blocked by GPTQ bring-up.

Before any real Phase 3B comparison:

1. fix GPTQ backend numerical stability
2. validate a GPTQ smoke baseline with finite perplexity
3. validate full `R3_Q17B`
4. build a GPTQ-specific candidate pool
5. only then run:
   - `G2B02_Q17B`
   - `G2R02_Q17B`

### Deliverables

- one trustworthy GPTQ baseline
- one matched GPTQ bits-vs-rank comparison
- one judgment on whether the RTN result is quantizer-dependent

## Phase 3C: Hybrid Second-Stage Testing

### Objective

Test whether rank becomes most useful after the best bit allocations are already taken.

### When To Start

Only after:

- RTN scale mapping is stable enough to show a clear regime trend
- or GPTQ results suggest a mixed / close frontier

### Minimal Hybrid Question

At a fixed regime point:

- start from the winning bits-only point
- spend the next budget slice either on more bits or targeted rank
- compare those choices directly

This is a better hybrid test than revisiting uniform repair.

## Immediate Next Steps

The practical start of Phase 3 should be:

1. Treat RTN Phase 2 as closed.
2. Use the Kaggle and Modal `1.7B` agreement as the trusted RTN large-model anchor.
3. Pause GPTQ retries until the backend path is changed or instrumented more deeply.
4. Prepare the RTN model-ladder expansion for one `3B-4B` model.

## Recommended First Execution Order

1. prepare RTN baseline configs for the next larger model
2. run the next `3B-4B` RTN baseline
3. run the first matched bits-vs-rank pair on that model
4. keep GPTQ as a separate recovery/debug track, not the main execution path

## Practical Summary

Phase 3 should be run as:

- a disciplined mapping phase
- not a grab-bag of extra experiments
- and not a continuation of Phase 2 local tuning

The main output of Phase 3 should be a regime map explaining when bits win, when rank wins, and whether hybrid only matters after the best bit upgrades are exhausted.
