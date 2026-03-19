# Item 4 Latency Analysis

This report summarizes the completed latency benchmark matrix for the paper-critical GPTQ comparison points:

- `Qwen3-1.7B` on `A10G`
- `Qwen3-8B` on `A100`
- baseline 4-bit, best bits-only, best rank-only
- batch sizes `1` and `8`

Canonical generated table:

- `results/analysis/latency_item4_summary.csv`

Benchmark contract:

- greedy `generate(...)`
- KV cache enabled
- prompt length `512`
- decode length `128`
- `3` warmups
- `10` timed repetitions

## Consolidated Run Table

### Qwen3-1.7B GPTQ on A10G

| Policy | Batch | Decode tok/s | Decode ms/token | First-token ms | Peak VRAM MB |
|--------|-------|--------------|-----------------|----------------|--------------|
| Baseline (`R3_Q17B`) | 1 | 15.6895 | 63.8874 | 68.3374 | 1488.33 |
| Bits (`G2B03_Q17B`) | 1 | 18.4842 | 54.1166 | 55.1588 | 4825.36 |
| Rank (`G2R02_Q17B`) | 1 | 19.9779 | 50.0631 | 52.8372 | 1634.43 |
| Baseline (`R3_Q17B`) | 8 | 138.0313 | 7.2448 | 303.5549 | 2030.62 |
| Bits (`G2B03_Q17B`) | 8 | 140.3932 | 7.1271 | 283.0136 | 5366.27 |
| Rank (`G2R02_Q17B`) | 8 | 118.7763 | 8.4514 | 283.6735 | 2178.47 |

### Qwen3-8B GPTQ on A100

| Policy | Batch | Decode tok/s | Decode ms/token | First-token ms | Peak VRAM MB |
|--------|-------|--------------|-----------------|----------------|--------------|
| Baseline (`R3_Q8B`) | 1 | 13.9960 | 71.4508 | 76.2555 | 6178.06 |
| Bits (`G2B02_Q8B`) | 1 | 13.7762 | 72.5901 | 77.7610 | 6255.19 |
| Rank (`G2R02_Q8B`) | 1 | 8.5127 | 117.4797 | 123.2996 | 6444.69 |
| Baseline (`R3_Q8B`) | 8 | 69.4134 | 14.4075 | 361.2871 | 6958.13 |
| Bits (`G2B02_Q8B`) | 8 | 112.0511 | 8.9306 | 346.4224 | 22663.13 |
| Rank (`G2R02_Q8B`) | 8 | 115.5812 | 8.6520 | 346.0992 | 7230.34 |

## Main Findings

### 1. Latency conclusions are workload-dependent

There is no single latency ordering that holds across both model scales and both batching regimes.

- `8B`, batch `1`:
  - bits is effectively baseline-speed (`+1.6%` decode ms/token vs baseline)
  - rank is much slower (`+64.4%` decode ms/token vs baseline; `+61.8%` vs bits)
- `8B`, batch `8`:
  - both targeted policies outperform the baseline on decode throughput
  - rank is slightly faster than bits on decode throughput (`+3.1%` tok/s vs bits)
- `1.7B`, batch `1`:
  - both targeted policies are faster than the baseline
  - rank is the fastest of the three
- `1.7B`, batch `8`:
  - bits is slightly faster than the baseline
  - rank is slower than both bits and baseline

The paper should therefore avoid a global claim such as "rank is slower than bits" without qualifying the workload.

### 2. Peak VRAM differences are large and policy-specific

Peak VRAM is not a minor secondary effect in this benchmark matrix.

- `1.7B`, batch `1`:
  - bits uses `+224%` peak VRAM vs baseline
  - rank uses only `+9.8%` peak VRAM vs baseline
- `1.7B`, batch `8`:
  - bits uses `+164%` peak VRAM vs baseline
  - rank uses `+7.3%` peak VRAM vs baseline
- `8B`, batch `1`:
  - peak VRAM differences are modest
  - bits `+1.3%` vs baseline, rank `+4.3%`
- `8B`, batch `8`:
  - bits uses `+226%` peak VRAM vs baseline
  - rank uses only `+3.9%` peak VRAM vs baseline

This means throughput alone is not sufficient for the practical recommendation. The paper must report peak VRAM with the latency result.

### 3. The 8B paper story remains bits-favoring, but with an important deployment caveat

From Item 3, `8B` was already bits-favoring on perplexity.

Item 4 adds:

- for low-batch interactive decode, bits is clearly the practical winner over rank
- for higher batching, rank can be throughput-competitive while using much less peak VRAM than bits

Paper-safe interpretation:

- `8B`, interactive/single-request decode: prefer bits
- `8B`, higher-batch throughput serving with tight VRAM limits: rank remains a viable deployment option even though it loses on perplexity

### 4. The 1.7B latency story is mixed, not rank-dominated

At `1.7B`, Item 3 already showed that the quality difference between bits and rank is within noise.

Item 4 adds:

- batch `1`: rank is fastest, bits is second, baseline is slowest
- batch `8`: bits is fastest, baseline is second, rank is slowest
- rank has only a small peak-VRAM premium over baseline, while bits has a large peak-VRAM premium

Paper-safe interpretation:

- `1.7B` does not support a simple single-winner policy recommendation
- the deployment choice is workload-sensitive and memory-sensitive

## Paper-Ready Claims From Item 4

- latency is workload-dependent; the practical ordering changes with batch size and model scale
- peak VRAM must be reported with decode throughput, because some targeted-bits runs materially increase serving memory
- `8B` interactive decode supports a clean practical recommendation in favor of bits over rank
- `1.7B` remains mixed: both quality and latency conclusions are conditional, not absolute

## Key Artifacts

- canonical generated table:
  - `results/analysis/latency_item4_summary.csv`
- implementation tracker:
  - `docs/roadmap/item4_latency_measurement_plan.md`
- saved run artifacts:
  - `results/modal_latency/qwen3_8b_gptq_baselines/`
  - `results/modal_latency/qwen3_8b_gptq_transfer/`
  - `results/modal_latency/qwen3_1p7b_gptq_baselines/`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/`
