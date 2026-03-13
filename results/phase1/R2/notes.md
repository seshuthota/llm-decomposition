# R2 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `rtn`
- Config source: `configs/phase1/r2_qwen3_0p6b_rtn_4bit.json`
- Status: `completed`
- Perplexity: `30.5169`
- Memory: `307304448` bytes
- Metadata bytes: `9312256`

Notes:

- This is the main local quantized baseline.
- Quantization damage was clearly concentrated in later `mlp.down_proj` layers, with some `self_attn.o_proj` layers also standing out.
- Residual profiles suggested that RTN errors were not strongly low-rank at tiny ranks.
