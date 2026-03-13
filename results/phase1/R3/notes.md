# R3 Notes

- Model: `Qwen/Qwen3-0.6B-Base`
- Method: `gptq`
- Config source: `configs/phase1/r3_qwen3_0p6b_gptq_4bit.json`
- Status: `blocked`

Notes:

- The local GPTQ path was not completed.
- The first blocker was missing `optimum`.
- The follow-up install attempt showed the current GPTQ dependency path expected newer GPU architecture than the local `GTX 1660 SUPER` (`sm_75`).
- GPTQ is deferred to another machine.
