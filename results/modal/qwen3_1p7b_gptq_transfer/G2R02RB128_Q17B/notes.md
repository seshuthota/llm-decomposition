# G2R02RB128_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_svd_rank`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2r02_rowblock128_qwen3_1p7b_gptq_rank_activation_1p0pct.json`
- Status: completed

Result summary:

- perplexity: `15.9034`
- memory total bytes: `1,370,523,945`
- repair bytes: `13,555,712`

Observations:

- This was the first structural GPTQ rank pilot using row-block repairs (`128` rows per block).
- The run was numerically valid and stayed within the `A10G` budget envelope.
- It underperformed the earlier matrix-level rank point `G2R02_Q17B` (`15.8823`).
- It also underperformed the stronger richer-bits point `G2B02RB128_Q17B` (`15.8970`).

Interpretation:

- Merely making GPTQ rank more local via row-block repair was not enough to improve the `1.7B` frontier.
- This result argues against spending more on small structural-rank variants before testing other bits layouts or more fundamental rank designs.
