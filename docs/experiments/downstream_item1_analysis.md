# Item 1 Downstream Analysis

This report aggregates the completed downstream evaluation matrix for the paper-readiness plan:

- GPTQ `Qwen3-1.7B`
- GPTQ `SmolLM3-3B`
- GPTQ `Qwen3-8B`
- RTN `Qwen3-1.7B` anchor

## Metric Definitions

For each run we compute:

- `avg_downstream`: the arithmetic mean of six task metrics
- task metric choice:
  - `hellaswag`, `arc_easy`, `arc_challenge`, `piqa`: `acc_norm`
  - `winogrande`, `boolq`: `acc`

For each quantizer/scale family, deltas are measured relative to the family baseline:

```text
ΔPPL = PPL_baseline - PPL_run
Δavg = avg_downstream_run - avg_downstream_baseline
quality_recovered_fraction = Δavg / (avg_downstream_full_precision - avg_downstream_baseline)
quality_recovered_per_mb = quality_recovered_fraction / added_mb
```

## Consolidated Run Table

### GPTQ / Qwen3-1.7B

| policy | run_id | perplexity | memory_mb | avg_downstream | latency_ms/token |
| --- | --- | --- | --- | --- | --- |
| full_precision | DS_FP_Q17B | 15.0347 | 3281.7 | 0.6856 | 0.1155 |
| baseline_4bit | DS_R3_Q17B | 15.9010 | 1294.1 | 0.6738 | 0.1266 |
| bits | DS_G2B03_Q17B | 15.8914 | 1319.1 | 0.6769 | 0.1249 |
| rank | DS_G2R02_Q17B | 15.8823 | 1297.9 | 0.6726 | 0.1185 |
| hybrid | DS_H2R02M_Q17B | 15.8962 | 1304.9 | 0.6727 | 0.1219 |
### GPTQ / Qwen3-8B

| policy | run_id | perplexity | memory_mb | avg_downstream | latency_ms/token |
| --- | --- | --- | --- | --- | --- |
| full_precision | DS_FP_Q8B | 11.2148 | 15622.6 | 0.7765 | 0.1025 |
| baseline_4bit | DS_R3_Q8B | 11.7970 | 5821.2 | 0.7756 | 0.1527 |
| bits | DS_G2B02_Q8B | 11.7823 | 5865.2 | 0.7749 | 0.2519 |
| rank | DS_G2R02_Q8B | 11.7962 | 5826.3 | 0.7758 | 0.1739 |
| hybrid | DS_H2R02_Q8B | 11.7895 | 5870.3 | 0.7748 | 0.2351 |
### GPTQ / SmolLM3-3B

| policy | run_id | perplexity | memory_mb | avg_downstream | latency_ms/token |
| --- | --- | --- | --- | --- | --- |
| full_precision | DS_FP_S3B | 10.8249 | 5865.3 | 0.7404 | 0.1342 |
| baseline_4bit | DS_R3_S3B | 11.5455 | 1898.0 | 0.7202 | 0.1964 |
| bits | DS_G3B02_S3B | 11.5483 | 1905.0 | 0.7222 | 0.1531 |
| rank | DS_G3R02_S3B | 11.6482 | 1907.8 | 0.7220 | 0.1035 |
### RTN / Qwen3-1.7B

| policy | run_id | perplexity | memory_mb | avg_downstream | latency_ms/token |
| --- | --- | --- | --- | --- | --- |
| baseline_4bit | DS_R2_Q17B | 21.2540 | 846.0 | 0.6472 | 0.0904 |
| bits | DS_P2B03_Q17B | 21.0963 | 860.0 | 0.6477 | 0.0907 |
| rank | DS_P2R02_Q17B | 21.2482 | 850.3 | 0.6488 | 0.0908 |

## Cross-Run Correlation

- Pearson correlation between `ΔPPL` and `Δavg` when full-precision references are included: `0.7747`
- Pearson correlation between `ΔPPL` and `Δavg` for compressed policies only: `-0.2099`

Interpretation:

- Full-precision anchors preserve a strong positive global trend: better perplexity generally corresponds to better downstream score when the comparison spans the entire quality range.
- Inside the compressed-policy regime, the relationship is weak and unstable. That is the paper-relevant result: small perplexity wins do not reliably transfer into uniformly better downstream task performance.

## Quality Recovered Per Added MB

| family | policy | run_id | ΔPPL | Δavg | added_mb | recovered_frac | recovered_frac_per_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPTQ / Qwen3-1.7B | bits | DS_G2B03_Q17B | +0.0097 | +0.0031 | 25.00 | +0.2626 | +0.01050 |
| GPTQ / Qwen3-1.7B | rank | DS_G2R02_Q17B | +0.0187 | -0.0012 | 3.81 | -0.0980 | -0.02570 |
| GPTQ / Qwen3-1.7B | hybrid | DS_H2R02M_Q17B | +0.0048 | -0.0011 | 10.81 | -0.0902 | -0.00834 |
| GPTQ / Qwen3-8B | bits | DS_G2B02_Q8B | +0.0147 | -0.0007 | 44.00 | -0.6648 | -0.01511 |
| GPTQ / Qwen3-8B | rank | DS_G2R02_Q8B | +0.0007 | +0.0003 | 5.12 | +0.2626 | +0.05125 |
| GPTQ / Qwen3-8B | hybrid | DS_H2R02_Q8B | +0.0075 | -0.0007 | 49.12 | -0.7414 | -0.01509 |
| GPTQ / SmolLM3-3B | bits | DS_G3B02_S3B | -0.0029 | +0.0020 | 7.00 | +0.1014 | +0.01449 |
| GPTQ / SmolLM3-3B | rank | DS_G3R02_S3B | -0.1027 | +0.0018 | 9.77 | +0.0888 | +0.00909 |
| RTN / Qwen3-1.7B | bits | DS_P2B03_Q17B | +0.1577 | +0.0005 | 14.00 | +0.0135 | +0.00096 |
| RTN / Qwen3-1.7B | rank | DS_P2R02_Q17B | +0.0059 | +0.0016 | 4.25 | +0.0415 | +0.00977 |

## Task-Win Counts Within Each Family

| family | run_id | task_wins |
| --- | --- | --- |
| GPTQ / Qwen3-1.7B | DS_R3_Q17B | 1 |
| GPTQ / Qwen3-1.7B | DS_G2B03_Q17B | 3 |
| GPTQ / Qwen3-1.7B | DS_G2R02_Q17B | 1 |
| GPTQ / Qwen3-1.7B | DS_H2R02M_Q17B | 1 |
| GPTQ / Qwen3-8B | DS_R3_Q8B | 2 |
| GPTQ / Qwen3-8B | DS_G2B02_Q8B | 0 |
| GPTQ / Qwen3-8B | DS_G2R02_Q8B | 2 |
| GPTQ / Qwen3-8B | DS_H2R02_Q8B | 2 |
| GPTQ / SmolLM3-3B | DS_R3_S3B | 2 |
| GPTQ / SmolLM3-3B | DS_G3B02_S3B | 3 |
| GPTQ / SmolLM3-3B | DS_G3R02_S3B | 1 |
| RTN / Qwen3-1.7B | DS_R2_Q17B | 2 |
| RTN / Qwen3-1.7B | DS_P2B03_Q17B | 3 |
| RTN / Qwen3-1.7B | DS_P2R02_Q17B | 1 |

## Best Downstream Policy By Family

| family | best_run | policy | avg_downstream | perplexity |
| --- | --- | --- | --- | --- |
| GPTQ / Qwen3-1.7B | DS_G2B03_Q17B | bits | 0.6769 | 15.8914 |
| GPTQ / Qwen3-8B | DS_G2R02_Q8B | rank | 0.7758 | 11.7962 |
| GPTQ / SmolLM3-3B | DS_G3B02_S3B | bits | 0.7222 | 11.5483 |
| RTN / Qwen3-1.7B | DS_P2R02_Q17B | rank | 0.6488 | 21.2482 |

## Conclusions

1. The downstream branch does not collapse the project into a universal winner.
2. Perplexity remains useful as a coarse global quality measure, but it is not sufficient to rank nearby compressed policies.
3. `Qwen3-1.7B` preserves the cross-quantizer contrast:
   - GPTQ: rank is best by perplexity, but bits wins the most task-level comparisons and the highest mean downstream score.
   - RTN: bits is best by perplexity and also wins the most task-level comparisons.
4. `SmolLM3-3B` remains the neutral midpoint:
   - baseline still has the best perplexity
   - bits slightly improves the mean downstream score over baseline
   - rank remains the weakest policy
5. `Qwen3-8B` stays bits-favoring by perplexity, but downstream is essentially tied between baseline, rank, and hybrid at the current budget slice.

These results are strong enough to mark Item 1 complete: the paper can now claim that the regime map holds beyond perplexity, but with a more nuanced downstream interpretation than the perplexity frontier alone.
