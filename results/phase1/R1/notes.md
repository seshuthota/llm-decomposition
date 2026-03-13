# R1 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `full_precision`
- Config source: `configs/phase1/r1_qwen3_0p6b_full_precision.json`
- Status: `completed`
- Perplexity: `16.8447`
- Memory: `1192099840` bytes
- Device: `cuda`

Notes:

- This is the baseline reference point for every later comparison.
- The first execution exposed a sequence-building issue in the evaluation pipeline, which was fixed afterward in `llm_decomposition/hf_utils.py`.
