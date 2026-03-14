# H2R02_Q8B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-8b-base`
- Method: `hybrid_second_stage`
- Config source: `/root/project/configs/scaleup_qwen3_8b_gptq/h2r02_qwen3_8b_gptq_hybrid_activation_1p0pct.json`
- Status: completed

Result summary:

- perplexity: `11.7895`
- memory total bytes: `6,155,487,107`
- prior bit bytes: `46,137,344`
- second-stage rank bytes: `5,373,952`

Observations:

- This run started from the stronger `8B` GPTQ bits point `G2B02_Q8B` and spent the next slice on rank.
- The result improved over baseline `R3_Q8B` (`11.7970`) and over the rank-only point `G2R02_Q8B` (`11.7962`).
- It did not beat the stronger bits-only point `G2B02_Q8B` (`11.7823`).

Interpretation:

- Hybrid second-stage repair is useful on `8B` relative to rank-only follow-up.
- Under the current action spaces, `8B` still prefers bits as the stronger standalone policy.
