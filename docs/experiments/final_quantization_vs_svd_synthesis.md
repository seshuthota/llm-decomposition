# Final Quantization-vs-SVD Synthesis

This document is the canonical project-level synthesis for the bounded study completed so far.

It merges:

- Phase 1 local RTN results
- Phase 2 targeted local RTN results
- Phase 3 RTN scale-map results
- GPTQ bring-up, frontier, policy-comparison, and bounded closure work

The project question is:

> Under a fixed extra-memory budget, should the next byte go to more bits or more SVD-style rank repair?

The final answer is:

> There is no universal winner. The preferred use of extra memory is regime-dependent. It changes with quantizer, model scale, and action-space design.

## Final Project Conclusion

Across the bounded study:

- `RTN 0.6B`: rank wins
- `RTN 1.7B`: bits win
- `RTN 3B`: bits win
- `RTN 8B`: bits win
- `GPTQ 1.7B`: rank wins
- `GPTQ 3B`: neither helps; bits regress less than rank
- `GPTQ 8B`: bits win

So the strongest defensible framing is:

- this is a decision-frontier project, not a universal winner project
- the best policy depends on regime
- action-space design matters, but bounded action-space refinements did not collapse the story into one winner

## RTN Frontier Summary

### Cross-Scale RTN Table

| Model | Baseline | Best Bits | Best Rank | Winner | Interpretation |
|-------|----------|-----------|-----------|--------|----------------|
| `Qwen3-0.6B` | `30.5169` | `30.2506` | `29.7223` | rank | smallest RTN regime still favors targeted rank |
| `Qwen3-1.7B` | `21.3102` | `21.1505` | `21.2971` | bits | scale-up flips the local `0.6B` result |
| `SmolLM3-3B` | `47.9169` | `47.4955` | `47.9833` | bits | first bridge-scale point clearly favors bits |
| `Qwen3-8B` | `16.1939` | `16.1429` | `16.2035` | bits | validation-scale RTN point also favors bits |

RTN conclusion:

- targeted rank can win in the very small-model regime
- targeted bits dominate once the model is large enough under the current RTN matrix-level setup

Canonical RTN references:

- [phase2_conclusion.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/phase2_conclusion.md)
- [phase3_rtn_regime_map.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/phase3_rtn_regime_map.md)

## GPTQ Frontier Summary

### Cross-Scale GPTQ Table

| Model | Baseline | Best Bits | Best Rank | Best Hybrid | Winner | Interpretation |
|-------|----------|-----------|-----------|-------------|--------|----------------|
| `Qwen3-1.7B` | `15.9137` | `15.8914` | `15.8823` | `15.8962` | rank | first matched GPTQ scale favors rank |
| `SmolLM3-3B` | `11.5366` | `11.5483` | `11.6482` | n/a | mixed / neutral | neither method improved over baseline |
| `Qwen3-8B` | `11.7970` | `11.7823` | `11.7962` | `11.7895` | bits | larger GPTQ validation point favors bits |

GPTQ conclusion:

- `1.7B` favors rank
- `3B` is mixed / neutral
- `8B` favors bits

So GPTQ also shows regime dependence rather than a universal winner.

Canonical GPTQ references:

- [qwen3_1p7b_gptq_bringup_status.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/qwen3_1p7b_gptq_bringup_status.md)
- [gptq_pre_final_branch_snapshot.md](/home/seshu/Documents/Python/llm-decomposition/docs/experiments/gptq_pre_final_branch_snapshot.md)

## GPTQ Policy Comparison

The bounded GPTQ policy-comparison branch is complete for the scales where the comparison was worth filling in.

### `1.7B` GPTQ Policy Ordering

| Policy | Run | Perplexity | Interpretation |
|--------|-----|------------|----------------|
| rank-only | `G2R02_Q17B` | `15.8823` | best `1.7B` GPTQ point |
| bits-only | `G2B03_Q17B` | `15.8914` | second-best current matrix policy |
| hybrid | `H2R02M_Q17B` | `15.8962` | useful, but not dominant |

Conclusion:

- `1.7B`: rank-only > bits-only > hybrid

### `8B` GPTQ Policy Ordering

| Policy | Run | Perplexity | Interpretation |
|--------|-----|------------|----------------|
| bits-only | `G2B02_Q8B` | `11.7823` | best `8B` GPTQ point |
| hybrid | `H2R02_Q8B` | `11.7895` | helpful, but still below bits-only |
| rank-only | `G2R02_Q8B` | `11.7962` | essentially flat relative to baseline |

Conclusion:

- `8B`: bits-only > hybrid > rank-only

This is important because:

- the two scales disagree
- hybrid is useful but not universally best
- the policy-level story is itself regime-dependent

## What The Bounded Follow-Ups Established

The project intentionally tested several bounded objections before stopping.

### GPTQ richer / structural follow-ups

What was tried:

- richer bits with row-block and column-block layouts
- finer rank ladders
- row-block rank
- column-block rank
- grouped family-aware matrix rank
- hybrid second-stage from stronger bits bases
- a final bounded multi-bit bits-policy gate

What these established:

- richer bits can matter, but not uniformly across scale
- simple local block variants were not a universal improvement
- grouped family balancing was clearly negative
- hybrid could help as a second-stage correction, but it was not dominant
- the final multi-bit bits gate did not overturn the `1.7B` rank result

### Multi-bit bits closure result

Final bounded run:

- `MB1_Q17B`: `15.9097`

Interpretation:

- the allocator mostly chose cheap `4->5` upgrades
- it improved over baseline
- but it did not beat the current best bits point
- and it remained clearly below the rank winner

So the predefined stop rule fired:

- no `MB2_Q17B`
- no `8B` multi-bit continuation
- GPTQ moves to synthesis rather than more experiments

Canonical closure plan:

- [gptq_closure_plan.md](/home/seshu/Documents/Python/llm-decomposition/docs/roadmap/gptq_closure_plan.md)

## Why The Project Stops Here

The bounded study is now strong enough to support reporting.

Reasons to stop:

- RTN scale map is complete
- GPTQ bring-up is complete
- GPTQ frontier is complete for the bounded action space
- GPTQ policy comparison is complete
- the last bounded bits-side objection has been tested and did not change the conclusion

Reasons not to continue inside this project scope:

- more GPTQ runs would likely reopen branch creep rather than clarify the current conclusion
- shared-family rank would be a new algorithm family, not a bounded extension
- the next larger-model or method-family work is better framed as a new project phase, not part of this bounded study

## Final Interpretation

The original intuition was:

- quantize first
- then spend a small repair budget where it matters most

The completed project shows that this intuition is only partly right.

The stronger result is:

- the best use of extra memory is a regime decision
- that decision depends on:
  - quantizer
  - model scale
  - action-space design

So the right final statement is not:

- “bits always win”
- or “rank always wins”

It is:

> The preferred post-quantization correction policy is regime-dependent. Under bounded action spaces, RTN and GPTQ each exhibit scale-sensitive frontiers in which the better use of extra memory changes across model size and policy class.

## Recommended Next Project Boundary

If work continues later, treat it as a new phase rather than continuing this bounded one.

Reasonable future phase candidates:

- larger-model extension (`14B+`)
- new quantizer family
- genuinely new algorithm family
- broader policy search beyond the current bounded action space

That work should be framed as an extension project, not as unfinished business from the current bounded study.
