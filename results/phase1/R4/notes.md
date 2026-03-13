# R4 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `uniform_svd_repair`
- Config source: `configs/phase1/r4_qwen3_0p6b_uniform_svd_rank4.json`
- Status: `completed`
- Perplexity: `30.5577`
- Memory: `307550208` bytes
- Added bytes vs `R2`: `245760`

Notes:

- First low-rank repair point.
- Slightly worse than plain `R2`, so rank 4 was too weak to help in this setup.
