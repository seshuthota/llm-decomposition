# R2_Q8B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-8b-base`
- Method: `rtn`
- Config source: `/root/project/configs/scaleup_qwen3_8b/r2_qwen3_8b_rtn_4bit.json`
- Status: completed

Result:

- perplexity: `16.193944972311687`
- memory total bytes: `3902300160`
- memory metadata bytes: `118251520`
- latency ms/token: `0.12391592523730807`

Notes:

- this baseline required two large-model fixes:
  - CPU-side RTN working tensors during quantization
  - sequential offload during activation profiling
- after those changes, `Qwen3-8B` completed on `A100` without escalating to a larger GPU tier

Use this file for run-specific observations or anomalies.
