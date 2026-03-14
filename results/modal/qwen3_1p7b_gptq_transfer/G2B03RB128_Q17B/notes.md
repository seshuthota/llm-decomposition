# G2B03RB128_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2b03_rowblock128_qwen3_1p7b_gptq_bits_activation_2p0pct.json`
- Status: completed

Result:

- perplexity: `15.9272`
- memory: `1,383,969,065` bytes
- validation: finite logits and finite loss

Interpretation:

- the `+2.0%` richer-bits follow-up was valid on the cheap `A10G` path
- but adding more row-block bits after the strong `G2B02RB128_Q17B` point made perplexity worse, not better
- so the next slice after the best richer-bits point should not be assumed to go to more of the same bit actions

Use this file for run-specific observations or anomalies.
