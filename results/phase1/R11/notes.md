# R11 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `mixed_precision_budget_match`
- Config source: `configs/phase1/r11_qwen3_0p6b_bits_match_r7.json`
- Status: `completed`
- Perplexity: `28.8618`
- Memory: `308877312` bytes
- Upgraded layers: `model.layers.27.mlp.down_proj`

Notes:

- First meaningful bits-only comparison in Phase 1.
- This run beat `R7` clearly on perplexity while staying in the same overall budget range.
- It is the main empirical reason the project pivoted toward mixed-precision allocation.
