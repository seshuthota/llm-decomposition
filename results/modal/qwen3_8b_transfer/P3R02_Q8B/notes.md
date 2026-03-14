# P3R02_Q8B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-8b-base`
- Method: `targeted_svd_rank`
- Config source: `/root/project/configs/scaleup_qwen3_8b/p3r02_qwen3_8b_rank_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `16.20352045227616`
- memory total bytes: `3911213056`
- extra budget bytes: `39023002`
- repair factor bytes: `8912896`

Interpretation:

- targeted rank was slightly worse than the `R2_Q8B` baseline and clearly worse than `P3B02_Q8B`
- the first matched pair was decisive enough to skip the `+2.0%` pair for `Qwen3-8B`

Use this file for run-specific observations or anomalies.
