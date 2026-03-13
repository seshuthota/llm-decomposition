# R3_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `gptq`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/r3_qwen3_1p7b_gptq_4bit.json`
- Status: paused / invalid baseline

Observed state so far:

- the currently written `metrics.json` is not trustworthy
- perplexity is catastrophically high (`16103625.67`)
- `layer_errors.json` only contains `lm_head`
- this run should not be used as the GPTQ anchor baseline

Next action on resume:

- re-run GPTQ via a smaller smoke configuration first
- only then re-attempt the full `R3_Q17B` baseline
