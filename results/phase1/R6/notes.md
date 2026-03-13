# R6 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `uniform_svd_repair`
- Config source: `configs/phase1/r6_qwen3_0p6b_uniform_svd_rank16.json`
- Status: `completed`
- Perplexity: `30.3155`
- Memory: `308287488` bytes
- Added bytes vs `R2`: `983040`

Notes:

- Rank 16 gave a clearer recovery than `R5`.
- This showed that repair quality improved with rank, but the cost-benefit question remained open.
