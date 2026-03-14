# G2R02CB128_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_svd_rank`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2r02_colblock128_qwen3_1p7b_gptq_rank_activation_1p0pct.json`
- Status: completed

Result summary:

- perplexity: `15.9004`
- memory total bytes: `1,370,521,897`
- repair bytes: `13,553,664`

Observations:

- This pilot used `128`-column structural rank repairs, matching the column grouping used by the quantizer.
- The selected actions concentrated on `self_attn.k_proj` column slices in layers `8-9`.
- The run was numerically stable on `A10G`.

Interpretation:

- Column-block rank was slightly better than the earlier row-block rank pilot.
- It still underperformed the original matrix-level rank result `G2R02_Q17B` (`15.8823`).
- It also remained worse than the stronger richer-bits point `G2B02RB128_Q17B` (`15.8970`).
- So small structural block-rank variants are still not beating the simpler matrix-level rank policy on `1.7B`.
