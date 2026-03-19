# Item 3 Multi-Seed Stability Analysis

This report aggregates the completed calibration-resampling runs for Item 3 of the paper-readiness plan.

Scope:

- GPTQ `Qwen3-1.7B`
- GPTQ `SmolLM3-3B`
- GPTQ `Qwen3-8B`
- three calibration seeds per policy: `42`, `123`, `456`

Primary generated table:

- `results/analysis/multiseed_stability_all_summary.csv`

The seed sweep measures calibration-sample sensitivity, not training stochasticity. The right interpretation is: if a policy ordering flips under these seed changes, the reported frontier is not robust enough for a strong paper claim.

## Consolidated Run Table

| scale | policy | seed | perplexity | memory_mb | latency_ms/token |
| --- | --- | --- | --- | --- | --- |
| 1.7B | rank | 42 | 15.8325 | 1297.9183 | 0.1135 |
| 1.7B | rank | 123 | 15.6916 | 1297.9197 | 0.1141 |
| 1.7B | rank | 456 | 15.9360 | 1297.9222 | 0.1312 |
| 1.7B | bits | 42 | 15.8527 | 1301.1058 | 0.1145 |
| 1.7B | bits | 123 | 15.6947 | 1301.1072 | 0.1407 |
| 1.7B | bits | 456 | 15.9381 | 1301.1097 | 0.1119 |
| 3B | rank | 42 | 11.6482 | 1907.8043 | 0.1232 |
| 3B | rank | 123 | 11.7111 | 1907.8057 | 0.1229 |
| 3B | rank | 456 | 11.8419 | 1907.8082 | 0.1186 |
| 3B | bits | 42 | 11.5483 | 1905.0387 | 0.1794 |
| 3B | bits | 123 | 11.6186 | 1905.0401 | 0.1206 |
| 3B | bits | 456 | 11.6963 | 1905.0426 | 0.1214 |
| 8B | rank | 42 | 11.7962 | 3726.6484 | 0.1772 |
| 8B | rank | 123 | 11.4828 | 3726.6484 | 0.2585 |
| 8B | rank | 456 | 11.5671 | 3726.6484 | 0.1415 |
| 8B | bits | 42 | 11.7823 | 3765.5234 | 0.1779 |
| 8B | bits | 123 | 11.4609 | 3765.5234 | 0.2383 |
| 8B | bits | 456 | 11.5494 | 3765.5234 | 0.2363 |

## Aggregate Statistics

| scale | policy | mean_ppl | std_ppl | min_ppl | max_ppl |
| --- | --- | --- | --- | --- | --- |
| 1.7B | rank | 15.8200 | 0.1227 | 15.6916 | 15.9360 |
| 1.7B | bits | 15.8285 | 0.1235 | 15.6947 | 15.9381 |
| 3B | rank | 11.7337 | 0.0988 | 11.6482 | 11.8419 |
| 3B | bits | 11.6211 | 0.0740 | 11.5483 | 11.6963 |
| 8B | rank | 11.6154 | 0.1622 | 11.4828 | 11.7962 |
| 8B | bits | 11.5975 | 0.1660 | 11.4609 | 11.7823 |

Mean deltas (`bits - rank`):

- `1.7B`: `+0.0085` PPL
- `3B`: `-0.1127` PPL
- `8B`: `-0.0178` PPL

Lower perplexity is better, so negative values favor bits and positive values favor rank.

## Per-Scale Interpretation

### GPTQ / Qwen3-1.7B

The original `1.7B` GPTQ gap was too small to survive seed variation. The mean difference is only `0.0085` PPL, much smaller than the observed standard deviation for either policy (`~0.123`). The ordering is not stable:

- seed `42`: rank wins
- seed `123`: effectively tied
- seed `456`: effectively tied

Paper implication:

- do not claim a robust `1.7B` rank-over-bits advantage
- report this regime as within experimental noise under calibration resampling

### GPTQ / SmolLM3-3B

The `3B` regime is much cleaner. Bits beats rank on all three seeds, and the mean gap (`0.1127` PPL) is larger than the bits-side standard deviation (`0.0740`) and comparable to the rank-side spread (`0.0988`).

Paper implication:

- `3B` is a stable bits-favoring or at least clearly non-rank-favoring point
- this scale remains a useful midpoint between the ambiguous `1.7B` result and the cleaner `8B` result

### GPTQ / Qwen3-8B

The `8B` result is now complete across both policies and all three seeds. Bits beats rank on every seed:

- seed `42`: `11.7823` vs `11.7962`
- seed `123`: `11.4609` vs `11.4828`
- seed `456`: `11.5494` vs `11.5671`

The mean gap is modest (`0.0178` PPL), but unlike the `1.7B` case the direction is stable across all three seeds.

Operational note:

- the `8B` sweep required fixing the Modal GPTQ runtime and config path
- the saved runs live under `results/modal_importfix_probe_v2/...`
- all completed `8B` runs passed GPTQ finite-value validation

Paper implication:

- `8B` is a directionally stable bits-favoring GPTQ regime
- the claim should still be framed as modest but consistent, not as a large absolute win

## Paper-Ready Claims From Item 3

1. Multi-seed evidence does not support a strong rank-over-bits claim at `1.7B` GPTQ.
2. Multi-seed evidence supports a bits-favoring interpretation at `8B` GPTQ.
3. `3B` remains a useful intermediate scale where bits also stays ahead of rank.
4. The paper should present error bars or mean ± std at `1.7B` and `8B`, not only single-seed point estimates.

## Key Artifacts

- report: `docs/experiments/item3_multiseed_analysis.md`
- roadmap tracker: `docs/roadmap/paper_readiness_plan.md`
- generated table: `results/analysis/multiseed_stability_all_summary.csv`
- `8B` blocker history: `docs/roadmap/qwen3_8b_multiseed_blocker.md`
