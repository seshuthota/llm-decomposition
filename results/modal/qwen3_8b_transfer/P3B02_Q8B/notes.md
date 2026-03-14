# P3B02_Q8B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-8b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_qwen3_8b/p3b02_qwen3_8b_bits_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `16.142918191325858`
- memory total bytes: `3935854592`
- extra budget bytes: `39023002`

Selected actions:

- `model.layers.21.self_attn.o_proj` `4 -> 8`
- `model.layers.20.self_attn.o_proj` `4 -> 8`
- `model.layers.18.self_attn.o_proj` `4 -> 8`
- `model.layers.22.self_attn.o_proj` `4 -> 8`

Interpretation:

- targeted bits improved over the `R2_Q8B` baseline at the first matched budget point
- the gain-per-byte policy again concentrated on late `self_attn.o_proj` matrices

Use this file for run-specific observations or anomalies.
