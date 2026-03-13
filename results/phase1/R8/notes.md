# R8 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `mixed_precision_budget_match`
- Config source: `configs/phase1/r8_qwen3_0p6b_bits_match_r4.json`
- Status: `completed`
- Perplexity: `30.5169`
- Memory: `307304448` bytes
- Upgraded layers: `[]`

Notes:

- This budget was too small to upgrade any layer under the whole-layer `4-bit -> 8-bit` comparator.
- The result effectively reproduced `R2`.
