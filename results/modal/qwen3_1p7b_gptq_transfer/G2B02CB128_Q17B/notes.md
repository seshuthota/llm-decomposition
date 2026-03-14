# G2B02CB128_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2b02_colblock128_qwen3_1p7b_gptq_bits_activation_1p0pct.json`
- Status: completed

Result summary:

- perplexity: `15.9171`
- memory total bytes: `1,370,468,649`

Observations:

- This pilot used `128`-column bit upgrades, which align naturally with the current quantizer grouping along columns.
- The selected actions concentrated on `self_attn.v_proj` column slices in late layers.
- The run was numerically stable and cheap enough to validate on `A10G`.

Interpretation:

- The design is valid, but it underperformed the `R3_Q17B` baseline (`15.9137`).
- It also underperformed the stronger matrix-level and row-block bits points.
- So column-local GPTQ bits are not a good follow-on branch under the current scoring rule.
