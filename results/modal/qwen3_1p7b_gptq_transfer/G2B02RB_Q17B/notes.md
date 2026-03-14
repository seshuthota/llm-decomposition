# G2B02RB_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2b02_rowblock256_qwen3_1p7b_gptq_bits_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `15.9060`
- memory: `1,370,337,577` bytes
- validation: finite logits and finite loss

Interpretation:

- the row-block richer-bits pilot is valid on Modal on `A10G`
- it improves over the `R3_Q17B` GPTQ baseline
- but it underperforms the existing matrix-level bits point `G2B02_Q17B` (`15.8993`)
- so the first richer-bits design is not yet a clear improvement and does not justify an immediate `8B` rerun at higher cost

Use this file for run-specific observations or anomalies.
