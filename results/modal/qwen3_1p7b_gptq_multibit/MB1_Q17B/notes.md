# MB1_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/mb1_qwen3_1p7b_gptq_multibit_bits_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `15.9097`
- memory: `1,366,667,561` bytes
- candidate bit widths: `5/6/8`
- validation: finite logits and finite loss

Interpretation:

- this was the first bounded multi-bit bits-policy run for the GPTQ endgame branch
- the allocator selected only `4->5` upgrades at the `+1.0%` budget
- the result was worse than the current best bits point `G2B02_Q17B` (`15.8993`)
- it was also well below the current best rank point `G2R02_Q17B` (`15.8823`)
- this hits the stop rule from `docs/roadmap/gptq_closure_plan.md`
- `MB2_Q17B` is therefore not justified and should not be run unless the bounded endgame plan is explicitly reopened

Use this file for run-specific observations or anomalies.
