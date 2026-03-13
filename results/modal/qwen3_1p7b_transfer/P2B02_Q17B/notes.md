# P2B02_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_1p7b/p2b02_qwen3_1p7b_bits_activation_1p0pct.json`
- Status: completed

Summary:

- This is the first valid `1.7B` bits-only transfer point.
- Final perplexity: `21.24151620487837`
- Final memory total bytes: `895496192`
- Extra budget bytes: `8871076`
- Selected matrix upgrades:
  - `model.layers.16.self_attn.o_proj`
  - `model.layers.14.self_attn.o_proj`
  - `model.layers.13.self_attn.o_proj`
  - `model.layers.15.self_attn.o_proj`

Use this file for run-specific observations or anomalies.
