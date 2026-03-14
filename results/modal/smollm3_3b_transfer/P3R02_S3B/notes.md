# P3R02_S3B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/smollm3-3b-base`
- Method: `targeted_svd_rank`
- Config source: `/root/project/configs/scaleup_smollm3_3b/p3r02_smollm3_3b_rank_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `47.98327419715927`
- memory total bytes: `1595498496`
- extra budget bytes: `15855206`
- repair factor bytes: `9977856`

Interpretation:

- At `+1.0%`, targeted rank was slightly worse than the `R2_S3B` baseline and clearly worse than `P3B02_S3B`.
- This made the first matched pair decisive enough to skip the `+2.0%` pair for `SmolLM3-3B`.

Use this file for run-specific observations or anomalies.
