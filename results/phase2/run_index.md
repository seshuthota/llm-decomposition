# Phase 2 Run Index

Phase 2 starts the first fair frontier comparison on top of `R2`.

## Run Table

| Run | Status | Method | Budget | Memory (bytes) | Perplexity | Quick read |
|-----|--------|--------|--------|----------------|------------|------------|
| P2B01 | completed | targeted mixed precision | `+0.25%` | 307304448 | 30.5169 | budget too small for matrix-level actions |
| P2B02 | completed | targeted mixed precision | `+1.0%` | 309401600 | 30.4238 | first meaningful bits-only frontier point |
| P2R01 | pending | targeted SVD rank | `+0.25%` | n/a | n/a | not yet run |
| P2R02 | completed | targeted SVD rank | `+1.0%` | 310319104 | 29.7223 | corrected rank allocator beats `P2B02` |
| P2B03 | completed | targeted mixed precision | `+2.0%` | 312547328 | 30.2506 | bits frontier improves but stays on `o_proj` only |
| P2R03 | completed | targeted SVD rank | `+2.0%` | 310515712 | 29.7252 | rank still beats bits at larger budget |

## Current Takeaway

- `P2B01` confirmed that matrix-level actions are still too coarse for very small budgets.
- `P2B02` and `P2B03` built a bits-only frontier that consistently selected `self_attn.o_proj` upgrades.
- `P2R02` and `P2R03` showed that corrected targeted rank outperformed the current bits-only frontier at both matched budgets.
- The local RTN Phase 2 answer is now clear enough to move on without more small local ablations.
