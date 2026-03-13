# Qwen3 1.7B Transfer Conclusion

## Decision

The `Qwen/Qwen3-1.7B-Base` transfer study is complete for the current `RTN 4-bit` + matrix-level action setup on Modal.

The main result is:

> On `Qwen/Qwen3-1.7B-Base`, targeted bits beat targeted rank at both tested budget points.

## Final Frontier Points

| Run | Method | Budget | Memory (bytes) | Perplexity |
|-----|--------|--------|----------------|------------|
| R2_Q17B | RTN 4-bit baseline | baseline | 887107584 | 21.3102 |
| P2B02_Q17B | targeted bits | `+1.0%` | 895496192 | 21.2415 |
| P2R02_Q17B | targeted rank | `+1.0%` | 891564032 | 21.3001 |
| P2B03_Q17B | targeted bits | `+2.0%` | 901787648 | 21.1505 |
| P2R03_Q17B | targeted rank | `+2.0%` | 891564032 | 21.2971 |

## What This Means

### 1. The `1.7B` result does not match the `0.6B` local result

Earlier, on `Qwen/Qwen3-0.6B-Base`, the local RTN Phase 2 result favored targeted rank.

At `1.7B`, the transfer result favors targeted bits instead.

That means the project now has a scale-sensitive result:

- `0.6B`: targeted rank won
- `1.7B`: targeted bits won

### 2. The bits advantage is stable across the tested budget range

At `+1.0%`:

- bits: `21.2415`
- rank: `21.3001`

At `+2.0%`:

- bits: `21.1505`
- rank: `21.2971`

So the `1.7B` bits advantage is not a one-budget artifact.

### 3. Rank appears to saturate early under the current `1.7B` setup

`P2R03_Q17B` is very close to `P2R02_Q17B`, even though the nominal budget increased.

That suggests the current targeted-rank setup is not extracting much more value at the larger budget on this model.

### 4. The right framing is now “when does the next byte go to bits vs rank?”

The project should no longer be framed as if one method universally wins.

The stronger research story is:

- the answer depends on scale, quantizer regime, and action granularity
- the key scientific question is how the compression frontier shifts across those regimes

## Recommended Next Step

The highest-value next step is not more local RTN-style ablations.

It is one of:

1. run the same comparison under `GPTQ` on a stronger machine, or
2. run one hybrid second-stage experiment on `1.7B`, starting from the best bits-only point

My recommendation:

- prioritize the `GPTQ` transfer next
- keep the `1.7B` RTN transfer result as a completed scale-up finding
