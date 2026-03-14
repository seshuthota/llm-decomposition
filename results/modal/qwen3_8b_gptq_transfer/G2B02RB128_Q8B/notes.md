# G2B02RB128_Q8B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-8b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_qwen3_8b_gptq/g2b02_rowblock128_qwen3_8b_gptq_bits_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `11.7954`
- memory: `6,164,268,931` bytes
- validation: finite logits and finite loss

Interpretation:

- the richer-bits `128`-row design remains valid on `8B`
- it improves slightly over the `R3_Q8B` baseline
- but it does not beat the existing matrix-level bits point `G2B02_Q8B` (`11.7823`)
- so the current richer-bits branch is mixed:
  - `1.7B`: improved over matrix-level bits
  - `8B`: worse than matrix-level bits

Use this file for run-specific observations or anomalies.
