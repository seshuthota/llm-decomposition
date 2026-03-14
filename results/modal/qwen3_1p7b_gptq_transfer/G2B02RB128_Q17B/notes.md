# G2B02RB128_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2b02_rowblock128_qwen3_1p7b_gptq_bits_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `15.8970`
- memory: `1,370,468,649` bytes
- validation: finite logits and finite loss

Interpretation:

- the finer `128`-row richer-bits design is valid on the cheap `A10G` path
- it improves over the baseline `R3_Q17B`
- it also beats the existing matrix-level bits point `G2B02_Q17B` (`15.8993`)
- it is the first richer GPTQ bits result strong enough to justify one `8B` validation run on `A100-80GB`

Use this file for run-specific observations or anomalies.
