# Phase 1 To Current Report

This document is the consolidated experiment report from the start of Phase 1 through the current `8B` GPTQ validation work.

It is intended to answer four questions clearly:

1. what we tested,
2. what the result was,
3. what changed in project direction after each phase,
4. what the most justified next step is now.

## Executive Summary

The project began as a hybrid-compression idea:

- quantize first,
- then recover quality with targeted low-rank repair.

The main scientific question gradually became:

> under a fixed memory budget, should the next byte go to more bits or more rank?

The strongest current result is that there is **no universal winner**.

Under the current matrix-level action space:

- `RTN 0.6B`: targeted rank wins
- `RTN 1.7B`: targeted bits win
- `RTN 3B`: targeted bits win
- `RTN 8B`: targeted bits win
- `GPTQ 1.7B`: targeted rank wins
- `GPTQ 3B`: neither helps; bits regress less than rank
- `GPTQ 8B`: targeted bits win

So the correct framing now is:

- the answer depends on **quantizer**, **model scale**, and **action-space granularity**
- this project is now studying a **decision frontier**, not trying to prove one method always wins
- the project is **not** currently expanding into shared-family rank as a new algorithm branch; that path is intentionally deferred to keep the current study bounded
- the only remaining bounded GPTQ experimental branch under consideration is a modest multi-bit bits-policy extension

## Initial Setup And Research Direction

The original proposal focused on low-rank repair after quantization.

Initial assumptions:

- quantization damage would be concentrated
- damaged residuals might be compressible with low-rank structure
- a small amount of repair could restore quality efficiently

The first implementation branch built:

- a config-driven experiment harness
- memory accounting
- perplexity evaluation on `WikiText-2`
- layerwise error profiling
- residual profiling

That gave us a clean base for equal-budget comparisons.

## Phase 1: Local RTN On `Qwen/Qwen3-0.6B-Base`

Phase 1 asked whether low-rank repair looked promising enough to lead the project.

### Baseline And Uniform Repair Runs

| Run | Method | Memory (bytes) | Perplexity | Interpretation |
|-----|--------|----------------|------------|----------------|
| `R1` | full precision | `1192099840` | `16.8447` | reference baseline |
| `R2` | `RTN 4-bit` | `307304448` | `30.5169` | large compression, major quality loss |
| `R4` | uniform SVD repair rank 4 | `307550208` | `30.5577` | slightly worse than `R2` |
| `R5` | uniform SVD repair rank 8 | `307795968` | `30.4654` | small recovery |
| `R6` | uniform SVD repair rank 16 | `308287488` | `30.3155` | clearer recovery |
| `R7` | uniform SVD repair rank 32 | `309270528` | `29.9848` | best uniform repair result |

Phase 1 findings:

- quantization damage was real and strongly non-uniform
- later `mlp.down_proj` layers were especially sensitive
- some `self_attn.o_proj` layers also mattered
- uniform low-rank repair helped, but only once rank became non-trivial

### Equal-Budget Bits-Only Comparisons

| Run | Method | Memory (bytes) | Perplexity | Interpretation |
|-----|--------|----------------|------------|----------------|
| `R8` | bits-only match to `R4` | `307304448` | `30.5169` | too little budget to upgrade any matrix |
| `R10` | bits-only match to `R6` | `307304448` | `30.5169` | still too little budget |
| `R11` | bits-only match to `R7` | `308877312` | `28.8618` | first meaningful bits-only comparator |

Key Phase 1 decision:

- once the bits-only comparator became real, it beat uniform low-rank repair
- this killed the original “uniform repair is the mainline” idea

Phase 1 conclusion:

- pivot from uniform low-rank repair
- move to targeted equal-budget bits-vs-rank comparisons

## Phase 2: Targeted RTN Frontier On `Qwen/Qwen3-0.6B-Base`

Phase 2 replaced uniform repair with a fair targeted comparison:

- targeted bits
- targeted rank
- same candidate pool
- same budget scale

### Final Local `0.6B` RTN Frontier

| Run | Method | Budget | Memory (bytes) | Perplexity |
|-----|--------|--------|----------------|------------|
| `R2` | RTN 4-bit baseline | baseline | `307304448` | `30.5169` |
| `P2B02` | targeted bits | `+1.0%` | `309401600` | `30.4238` |
| `P2R02` | targeted rank | `+1.0%` | `310319104` | `29.7223` |
| `P2B03` | targeted bits | `+2.0%` | `312547328` | `30.2506` |
| `P2R03` | targeted rank | `+2.0%` | `310515712` | `29.7252` |

Important implementation detail:

- the first rank allocator under-spent budget
- after switching to incremental rank chunks per layer, targeted rank improved sharply

Phase 2 conclusion on local `0.6B` RTN:

- targeted rank beat targeted bits at both useful budgets

This was the first point where the project stopped being “repair is weak” and became “repair can win in some regimes.”

## RTN Scale-Up On `Qwen/Qwen3-1.7B-Base`

The next question was whether the `0.6B` result transferred.

### Modal `1.7B` RTN Frontier

| Run | Method | Budget | Memory (bytes) | Perplexity |
|-----|--------|--------|----------------|------------|
| `R2_Q17B` | RTN 4-bit baseline | baseline | `887107584` | `21.3102` |
| `P2B02_Q17B` | targeted bits | `+1.0%` | `895496192` | `21.2415` |
| `P2R02_Q17B` | targeted rank | `+1.0%` | `891564032` | `21.3001` |
| `P2B03_Q17B` | targeted bits | `+2.0%` | `901787648` | `21.1505` |
| `P2R03_Q17B` | targeted rank | `+2.0%` | `891564032` | `21.2971` |

Kaggle reproduction later confirmed the same ordering.

Phase conclusion:

- `1.7B` flipped the `0.6B` result
- targeted bits won at both budgets

That changed the project framing again:

- the answer is scale-dependent even under the same quantizer

## Phase 3 RTN Regime Map

Phase 3 extended the RTN comparison across scale.

### `RTN` Final Cross-Scale Results

| Model | Baseline | Targeted Bits | Targeted Rank | Result |
|-------|----------|---------------|---------------|--------|
| `Qwen3-0.6B` | `30.5169` | `30.4238` / `30.2506` | `29.7223` / `29.7252` | rank wins |
| `Qwen3-1.7B` | `21.3102` | `21.2415` / `21.1505` | `21.3001` / `21.2971` | bits win |
| `SmolLM3-3B` | `47.9169` | `47.4955` | `47.9833` | bits win |
| `Qwen3-8B` | `16.1939` | `16.1429` | `16.2035` | bits win |

Phase 3 RTN conclusion:

- targeted rank only won in the smallest tested RTN regime
- targeted bits dominated once model scale increased

That completed the active RTN branch.

## GPTQ Bring-Up

GPTQ was the major unresolved branch for a long time.

### Early GPTQ Problems

We hit several issues before GPTQ became usable:

- local GPU incompatibility on the original machine
- Modal image/dependency problems
- invalid early baselines
- `NaN` perplexity during Kaggle smoke tests
- artifact persistence issues in detached Modal runs
- `hf_device_map` / packing path issues
- tensor replacement issues in targeted GPTQ updates

Those problems were gradually fixed by:

- forcing `float16`
- separating the full-precision reference model from the quantized model
- validating logits/loss for finiteness before accepting perplexity
- fixing Modal result persistence
- injecting `hf_device_map` before GPTQ packing
- replacing selected GPTQ modules with floating `nn.Linear` modules for targeted updates

Once that was done, the GPTQ branch became scientifically usable.

## GPTQ On `Qwen/Qwen3-1.7B-Base`

### Final `1.7B` GPTQ Frontier

| Run | Method | Memory (bytes) | Perplexity |
|-----|--------|----------------|------------|
| `R3_Q17B` | GPTQ 4-bit baseline | `1356968233` | `15.9137` |
| `G2B02_Q17B` | targeted bits `+1.0%` | `1364308265` | `15.8993` |
| `G2R02_Q17B` | targeted rank `+1.0%` | `1360965929` | `15.8823` |
| `G2B03_Q17B` | targeted bits `+2.0%` | `1383182633` | `15.8914` |
| `G2R03_Q17B` | targeted rank `+2.0%` | `1360965929` | `15.8823` |

Interpretation:

- GPTQ at `1.7B` favored targeted rank at the first matched point
- the rank frontier saturated by `+1.0%`, so `G2R03_Q17B` did not move further

This was important because it did **not** match the `RTN 1.7B` result.

So now the project had both:

- scale dependence under `RTN`
- quantizer dependence at `1.7B`

## GPTQ On `SmolLM3-3B-Base`

### Final `3B` GPTQ Minimal Pair

| Run | Method | Memory (bytes) | Perplexity |
|-----|--------|----------------|------------|
| `R3_S3B` | GPTQ 4-bit baseline | `1990237781` | `11.5366` |
| `G3B02_S3B` | targeted bits `+1.0%` | `1997577813` | `11.5483` |
| `G3R02_S3B` | targeted rank `+1.0%` | `2000477781` | `11.6482` |

Interpretation:

- neither targeted method improved over the GPTQ baseline
- targeted bits regressed less than targeted rank

This was enough that the `+2.0%` pair was not worth spending on immediately.

## GPTQ On `Qwen/Qwen3-8B-Base`

The `8B` GPTQ path took two extra infrastructure discoveries:

1. `A100 40GB` was not enough for a stable single-device run
2. `A100-80GB` plus `device_map: "single"` was the working path

Using `device_map: "auto"` was the wrong direction here because it reintroduced the `accelerate` offload path that had already caused `rotary_emb` failures.

### `8B` GPTQ Smoke And Baseline

| Run | Method | Memory (bytes) | Perplexity |
|-----|--------|----------------|------------|
| `R3S_Q8B` | GPTQ smoke | `6103927291` | `13.7503` |
| `R3_Q8B` | GPTQ baseline | `6103975811` | `11.7970` |

Both runs validated with finite logits/loss.

## GPTQ Policy Comparison Status

The bounded GPTQ policy-comparison branch is now complete.

Trusted policy ordering:

- `1.7B`: rank-only > bits-only > hybrid
- `8B`: bits-only > hybrid > rank-only

Meaning:

- the `1.7B` and `8B` GPTQ policy results disagree
- hybrid is helpful but not dominant
- the GPTQ story remains regime-dependent even after filling the missing fair comparisons

## GPTQ Endgame Decision

At this point the GPTQ branch has two valid exits:

1. stop experimentation and close GPTQ
2. run one final bounded multi-bit bits-policy branch

What we are explicitly not doing:

- shared-family rank
- a new rank-method family
- another open-ended rank redesign branch

Reason:

- the current GPTQ evidence is already strong enough to support reporting
- the remaining reasonable objection is on the bits side, not on opening a new method family

The active frozen GPTQ state is recorded in:

- [gptq_pre_final_branch_snapshot.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/gptq_pre_final_branch_snapshot.md)

The active bounded decision plan is:

- [gptq_closure_plan.md](/home/seshu/Documents/Python/llm-decomposition/docs/roadmap/gptq_closure_plan.md)

### `8B` GPTQ First Matched Pair

| Run | Method | Memory (bytes) | Perplexity |
|-----|--------|----------------|------------|
| `G2B02_Q8B` | targeted bits `+1.0%` | `6150113155` | `11.7823` |
| `G2R02_Q8B` | targeted rank `+1.0%` | `6109349763` | `11.7962` |

Interpretation:

- targeted bits improved slightly over the baseline
- targeted rank was effectively flat
- at `8B` under GPTQ, bits beat rank

## Overall Scientific Picture

### RTN Summary

- `0.6B`: rank wins
- `1.7B`: bits win
- `3B`: bits win
- `8B`: bits win

### GPTQ Summary

- `1.7B`: rank wins
- `3B`: neither helps; bits regress less than rank
- `8B`: bits win

### What This Means

The current project conclusion is:

1. There is no universal rule like “bits always win” or “rank always wins.”
2. The answer changes with:
   - quantizer
   - model scale
   - candidate pool
   - action-space granularity
3. The project has successfully moved from a method-comparison story to a regime-map story.

## Operational Lessons

Some engineering findings were also important:

- Modal was reliable for RTN and, after backend fixes, for GPTQ as well.
- Kaggle was useful for RTN confirmation and GPTQ environment testing, but not the final mainline execution platform.
- `A10G` was often enough for medium RTN or some GPTQ baseline work, but transfer and larger GPTQ runs needed more headroom.
- `A100-80GB` with single-device placement was the first proven working setup for `Qwen3-8B` GPTQ.
- accidental file-vs-directory path collisions in `results/modal/...` caused repeated fetch issues and had to be worked around manually.

## Most Important Conclusions By Stage

### After Phase 1

- uniform low-rank repair should not be the mainline
- targeted equal-budget comparisons are necessary

### After Phase 2 / Early Scale-Up

- `0.6B` RTN supported targeted rank
- `1.7B` RTN flipped to targeted bits

### After Phase 3 RTN

- under RTN, targeted bits dominate outside the smallest tested regime

### After GPTQ Validation

- GPTQ does not mirror RTN mechanically
- `1.7B` GPTQ favored rank
- `8B` GPTQ favored bits
- `3B` GPTQ sat in between, where neither targeted method helped

## Recommended Next Step

The next best step is **not** more budget points with the current matrix-level GPTQ action space.

The strongest next branch is:

1. implement richer GPTQ action spaces,
2. rerun the strongest scale points (`1.7B` and `8B`),
3. then evaluate whether the frontier stabilizes.

Why this is the best next step:

- current rank often saturates too early or underperforms once scale increases
- bits are still limited to coarse matrix-level upgrades
- the current frontier likely reflects action-space limits as much as model behavior

Second-best next step:

- hybrid second-stage GPTQ testing
- start from the best bits-only point and ask whether the next budget slice should go to more bits or targeted rank

Lower-value next step right now:

- simply adding more `+2.0%` or higher budget points with the same current action space

## Canonical References

Main source documents:

- [phase1_results.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/phase1_results.md)
- [phase2_conclusion.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/phase2_conclusion.md)
- [qwen3_1p7b_transfer_conclusion.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/qwen3_1p7b_transfer_conclusion.md)
- [phase3_rtn_regime_map.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/phase3_rtn_regime_map.md)
- [qwen3_1p7b_gptq_bringup_status.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/qwen3_1p7b_gptq_bringup_status.md)
- [experiment_journal.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/experiment_journal.md)
- [next_steps.md](/home/seshu/Documents/Python/llm-decomposition/docs/roadmap/next_steps.md)
