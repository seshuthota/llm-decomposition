# H2R02RB128_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `hybrid_second_stage`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/h2r02_rowblock128_qwen3_1p7b_gptq_hybrid_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `15.8989`
- memory: `1,374,466,345` bytes
- prior bits base: `G2B02RB128_Q17B`
- prior bit bytes: `13,500,416`
- second-stage rank bytes: `3,997,696`
- validation: finite logits and finite loss

Interpretation:

- the hybrid second-stage path is now implemented and valid
- spending the next budget slice on rank after the best richer-bits point was much better than spending that same slice on more richer bits (`G2B03RB128_Q17B`: `15.9272`)
- but it still did not beat the earlier pure-rank `+1.0%` matrix-level GPTQ point `G2R02_Q17B` (`15.8823`)
- so the result is useful evidence for second-stage behavior, but not yet evidence that hybrid is the best overall GPTQ policy on `1.7B`

Use this file for run-specific observations or anomalies.
