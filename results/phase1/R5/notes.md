# R5 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `uniform_svd_repair`
- Config source: `configs/phase1/r5_qwen3_0p6b_uniform_svd_rank8.json`
- Status: `completed`
- Perplexity: `30.4654`
- Memory: `307795968` bytes
- Added bytes vs `R2`: `491520`

Notes:

- Rank 8 produced the first small quality recovery over `R2`.
- This kept the low-rank path alive, but the gain was still modest.
