# G2B03_Q8B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-8b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_qwen3_8b_gptq/g2b03_qwen3_8b_gptq_bits_activation_2p0pct.json`
- Status: completed

Result summary:

- perplexity: `11.8024`
- memory total bytes: `6,175,278,979`

Observations:

- This was the missing `8B` bits-only comparator at the same total budget scale as `H2R02_Q8B`.
- The run was numerically stable on `A100-80GB`.

Interpretation:

- Adding more bits beyond the first `+1.0%` slice hurt rather than helped on `8B`.
- This run is worse than both `G2B02_Q8B` and `H2R02_Q8B`.
- So at `8B`, the best bits-only policy is still the first `+1.0%` point, not a larger bits budget.
