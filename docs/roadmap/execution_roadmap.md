# Execution Roadmap

This document consolidates the old execution plan, evaluation plan, first experiment spec, and model shortlist into one active roadmap.

## Objective

Answer one clean question:

> Under a fixed memory budget, when should extra capacity go to bits, and when should it go to low-rank residual repair?

The current evidence says the project should treat **non-uniform bit allocation** as the mainline path and test low-rank repair as a secondary or second-stage method.

## Active Scope

The project mainline is:

- post-training only
- frozen model
- calibration-driven evaluation
- equal-memory comparisons
- activation-space damage as the primary allocation signal
- mixed-precision bit allocation as the first allocator to beat

Secondary paths:

- truncated-SVD residual repair
- later targeted low-rank repair
- eventual hybrid allocator that adds rank only after useful bit upgrades are chosen

## Model Ladder

Current ladder:

- debug model: `Qwen/Qwen3-0.6B-Base`
- mainline model: `Qwen/Qwen3-1.7B-Base`
- cross-family validation model: `HuggingFaceTB/SmolLM3-3B-Base`

Avoid for the first pass:

- very large or multimodal models
- gated-access models when unnecessary

## Quantization Strategy

Current plan by machine:

- on this machine: `RTN` for local bring-up and allocator development
- on a stronger machine: `GPTQ` for the stronger PTQ baseline

Why:

- `RTN` already runs locally and is enough to build the allocator/evaluation stack
- current GPTQ dependency path is blocked by GPU architecture on the local machine

## Evaluation Rules

Every major comparison should include:

1. full-precision baseline
2. pure quantized baseline
3. mixed-precision bits-only baseline
4. uniform low-rank repair baseline
5. targeted or hybrid allocation result

Primary metrics:

- perplexity on held-out text
- total model memory in bytes
- quality recovered per added megabyte

Secondary metrics:

- compact downstream task accuracy
- latency
- layerwise activation-space error
- layerwise weight-space error

Non-negotiable rule:

- final method claims must be made at equal total memory

## Required Outputs Per Run

Each run should log:

- model name
- quantization method
- bit-width
- repair rank or allocation actions
- total memory footprint
- detailed memory breakdown
- perplexity
- latency
- layerwise error summaries
- residual profile summaries for selected layers

Machine-readable outputs should stay in `results/`.

## Completed Milestone: Phase 1

Completed on `Qwen/Qwen3-0.6B-Base`:

- `R1`: full precision
- `R2`: RTN 4-bit baseline
- `R4-R7`: uniform SVD repair rank sweep
- `R8`, `R10`, `R11`: bits-only matched baselines

Main takeaways:

- quantization damage is non-uniform
- later `mlp.down_proj` layers are especially sensitive
- uniform low-rank repair helps only once rank becomes moderately large
- meaningful bits-only allocation beats uniform low-rank repair

See:

- [../experiments/phase1_results.md](../experiments/phase1_results.md)

## Next Execution Phases

### Phase 2: Better Mixed-Precision Allocator

Build a finer-grained bits-only allocator that is not limited to whole-layer `4-bit -> 8-bit` jumps.

Goals:

- allow meaningful budget use at smaller byte increments
- measure gain per byte more cleanly
- establish a stronger bits-only baseline

### Phase 3: Scale the Mainline

Once the allocator is stable on `Qwen3-0.6B`, move to:

- `Qwen/Qwen3-1.7B-Base`

Only then decide whether to add:

- `HuggingFaceTB/SmolLM3-3B-Base`

### Phase 4: GPTQ on Another Machine

Run the same baseline and allocator logic with a stronger PTQ backend on hardware compatible with the current GPTQ stack.

Questions:

- does GPTQ change the damage ranking?
- are GPTQ residuals more compressible than RTN residuals?
- does the bit-vs-rank decision boundary shift?

### Phase 5: Hybrid Second Stage

Only after the bits-only allocator is strong:

- test whether low-rank repair adds value on the remaining hardest layers
- compare targeted repair against the stronger mixed-precision baseline, not against weak baselines

## Immediate Next Tasks

1. Improve mixed-precision upgrade granularity.
2. Re-run equal-budget comparisons with a meaningful bits-only allocator at smaller budgets.
3. Keep updating the experiment journal and phase summary after each run batch.
