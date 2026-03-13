# R10 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `mixed_precision_budget_match`
- Config source: `configs/phase1/r10_qwen3_0p6b_bits_match_r6.json`
- Status: `completed`
- Perplexity: `30.5169`
- Memory: `307304448` bytes
- Upgraded layers: `[]`

Notes:

- Even this larger budget was still too small for a whole-layer upgrade.
- The result again collapsed to the plain `R2` baseline.
