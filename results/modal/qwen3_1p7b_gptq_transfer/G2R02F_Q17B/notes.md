# G2R02F_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_svd_rank`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2r02_fineranks_qwen3_1p7b_gptq_rank_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `15.9073`
- memory: `1,364,963,625` bytes
- repair bytes: `7,995,392`
- validation: finite logits and finite loss

Interpretation:

- the finer-rank ladder successfully spent more budget than the original matrix-level rank point
- but the result was worse than the original `G2R02_Q17B` rank point (`15.8823`)
- so simply making the rank ladder finer is not enough; the next GPTQ branch should likely move to hybrid second-stage or a more structural rank action space

Use this file for run-specific observations or anomalies.
