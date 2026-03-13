# Phase 2 Conclusion

## Decision

Phase 2 is complete for the current local RTN setup.

The main conclusion is:

> Under the current `RTN 4-bit` + matrix-level action space on `Qwen/Qwen3-0.6B-Base`, targeted low-rank repair outperformed the current targeted bits-only allocator at both matched budget points that mattered.

## Final Frontier Points

| Run | Method | Budget | Memory (bytes) | Perplexity |
|-----|--------|--------|----------------|------------|
| R2 | RTN 4-bit baseline | baseline | 307304448 | 30.5169 |
| P2B02 | targeted bits | `+1.0%` | 309401600 | 30.4238 |
| P2R02 | targeted rank | `+1.0%` | 310319104 | 29.7223 |
| P2B03 | targeted bits | `+2.0%` | 312547328 | 30.2506 |
| P2R03 | targeted rank | `+2.0%` | 310515712 | 29.7252 |

## What This Means

### 1. The corrected rank allocator changed the Phase 2 answer

The first rank comparison under-spent the budget and was not fair.

Once the rank action space was changed from:

- one final rank choice per layer

to:

- incremental rank chunks per layer

the allocator was able to spend budget properly and the rank frontier improved sharply.

### 2. Targeted rank beat the current bits-only frontier at both useful budgets

At `+1.0%`:

- bits: `30.4238`
- rank: `29.7223`

At `+2.0%`:

- bits: `30.2506`
- rank: `29.7252`

So the current local evidence does not support a simple "bits first" claim under this RTN setup.

### 3. The bits-only frontier stayed concentrated on `self_attn.o_proj`

Both bits runs selected only `self_attn.o_proj` upgrades.

That means the current matrix-level gain-per-byte bit allocator is finding attention output projections cheaper and more attractive than the expensive `mlp.down_proj` upgrades.

### 4. The rank frontier used a mixed layer family

The rank allocator spread early rank over:

- `mlp.down_proj`
- `self_attn.o_proj`

and then kept increasing rank on the best matrices through incremental rank steps.

That is a materially richer allocation pattern than the bits-only frontier.

### 5. The `+2.0%` rank point mostly confirmed saturation, not new gains

`P2R03` was very close to `P2R02`.

That suggests:

- the strongest gains were already captured by the `+1.0%` rank allocation
- or the current candidate pool is beginning to saturate

So adding more local RTN variants right now is unlikely to be the highest-value next move.

## Layer-Type Payoff Summary

Bits-only:

- selected `self_attn.o_proj` only

Targeted rank:

- selected both `mlp.down_proj` and `self_attn.o_proj`
- then increased rank progressively on the strongest matrices

Interpretation:

- bits-only and rank are not behaving identically
- but since rank already wins clearly at both budgets, a hybrid second-stage test is not necessary to justify continuing the project

## Decision On Hybrid Second-Stage Testing

Do not block on a hybrid second-stage check in the local RTN path.

Reason:

- the same method already wins clearly at both matched budgets
- the extra local work is more likely to refine an already clear answer than to change it

Hybrid testing may still be worth revisiting later:

- on a larger model
- under `GPTQ`
- or after adding finer-grained bit actions

But it is not required to close Phase 2.

## Recommended Next Phase

The next phase should be one of:

1. scale the stabilized allocator logic to `Qwen/Qwen3-1.7B-Base`
2. port the same comparison framework to `GPTQ` on another machine

My recommendation:

- keep the current local result as the completed RTN Phase 2 conclusion
- next try the same framework on the larger Qwen model if local resources allow
- otherwise move to GPTQ on the stronger machine because quantizer dependence is now the most important unresolved question
