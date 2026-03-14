# H2R02M_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `hybrid_second_stage`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/h2r02_matrix_qwen3_1p7b_gptq_hybrid_activation_1p0pct.json`
- Status: completed

Result summary:

- perplexity: `15.8962`
- memory total bytes: `1,368,305,961`
- prior bit bytes: `7,340,032`
- second-stage rank bytes: `3,997,696`

Observations:

- This run removed the earlier granularity confound by starting from the matrix-level bits policy `G2B02_Q17B`.
- The result was numerically stable on `A10G`.

Interpretation:

- Matrix-hybrid improved over the `+1.0%` bits-only point `G2B02_Q17B`.
- It did not beat the `+2.0%` bits-only point `G2B03_Q17B`.
- It also did not beat the pure rank point `G2R02_Q17B`.
- So the `1.7B` matrix-policy ordering is now clear:
  - rank-only best
  - bits-only second
  - hybrid third
