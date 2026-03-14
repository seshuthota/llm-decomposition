# G2R03_Q8B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-8b-base`
- Method: `targeted_svd_rank`
- Config source: `/root/project/configs/scaleup_qwen3_8b_gptq/g2r03_qwen3_8b_gptq_rank_activation_2p0pct.json`
- Status: completed

Result summary:

- perplexity: `11.7962`
- memory total bytes: `6,109,349,763`
- repair bytes: `5,373,952`

Observations:

- This was the missing `8B` rank-only comparator at the same total budget scale as `H2R02_Q8B`.
- The run was numerically stable on `A100-80GB`.
- It matched the earlier `G2R02_Q8B` result exactly at the metric level.

Interpretation:

- The current `8B` rank action space is saturated by `+1.0%`.
- Spending the extra slice on more rank does not improve the frontier.
- At equal total budget on `8B`, hybrid is better than rank-only, but bits-only `+1.0%` still remains best.
