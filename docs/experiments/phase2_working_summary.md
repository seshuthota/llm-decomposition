# Phase 2 Working Summary

This is the live Phase 2 comparison sheet. It is meant to stay short and decision-oriented while raw outputs remain in `results/phase2/`.

## Goal

Phase 2 is building the first fair frontier comparison between:

- targeted bits
- targeted rank
- later hybrid second-stage repair

All runs are currently anchored to:

- base model: `Qwen/Qwen3-0.6B-Base`
- base quantizer: `RTN 4-bit`
- base anchor run: `R2`

## Current Comparison Table

| Run | Status | Method | Budget | Memory (bytes) | Perplexity | Read |
|-----|--------|--------|--------|----------------|------------|------|
| P2B01 | completed | targeted bits, uniform | `+0.25%` | 307304448 | 30.5169 | too small for matrix-level actions |
| P2B02 | completed | targeted bits, greedy activation | `+1.0%` | 309401600 | 30.4238 | first meaningful bits-only point |
| P2R01 | pending | targeted rank, uniform | `+0.25%` | n/a | n/a | not yet reviewed |
| P2R02 | completed | targeted rank, greedy activation | `+1.0%` | 310319104 | 29.7223 | corrected rank allocator beats `P2B02` |
| P2B03 | completed | targeted bits, greedy activation | `+2.0%` | 312547328 | 30.2506 | bits improve, still `o_proj`-only |
| P2R03 | completed | targeted rank, greedy activation | `+2.0%` | 310515712 | 29.7252 | rank still beats bits |

## What We Already Know

- `P2B01` confirmed that matrix-level upgrades are still too coarse for very small budgets.
- `P2B02` and `P2B03` showed a consistent bits-only frontier that chooses `self_attn.o_proj` upgrades on gain-per-byte grounds.
- `P2R02` and `P2R03` showed that, once the rank allocator is allowed to add rank incrementally on strong layers, targeted rank beats the current matrix-level bits-only frontier at both matched budgets.

## Immediate Questions

1. Is the next best move to scale the RTN conclusion to a larger model or to transfer the framework to `GPTQ` on another machine?
2. Does the rank advantage remain under a different quantizer?
3. Does a larger model preserve the same layer-family pattern?

## Decision Rules

- Since `P2R03` also beat `P2B03`, the local RTN Phase 2 result is strong enough to close without another small-budget local detour.
- Hybrid second-stage testing is optional now, not required.
- The next meaningful uncertainty is quantizer dependence or scale dependence, not another local RTN micro-variant.

## Canonical Raw Files

- [../../results/phase2/run_index.md](../../results/phase2/run_index.md)
- [../../results/phase2/P2B02/metrics.json](../../results/phase2/P2B02/metrics.json)
- [../../results/phase2/P2B02/actions.json](../../results/phase2/P2B02/actions.json)
- [../../results/phase2/P2R02/metrics.json](../../results/phase2/P2R02/metrics.json)
- [../../results/phase2/P2R02/actions.json](../../results/phase2/P2R02/actions.json)
- [../../results/phase2/P2B03/metrics.json](../../results/phase2/P2B03/metrics.json)
- [../../results/phase2/P2R03/metrics.json](../../results/phase2/P2R03/metrics.json)
