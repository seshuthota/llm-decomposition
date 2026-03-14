# P3B02_S3B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/smollm3-3b-base`
- Method: `targeted_mixed_precision`
- Config source: `/root/project/configs/scaleup_smollm3_3b/p3b02_smollm3_3b_bits_activation_1p0pct.json`
- Status: completed

Result:

- perplexity: `47.4955338117833`
- memory total bytes: `1591812096`
- extra budget bytes: `15855206`

Selected actions:

- `model.layers.7.self_attn.o_proj` `4 -> 8`
- `model.layers.3.self_attn.o_proj` `4 -> 8`
- `model.layers.0.self_attn.o_proj` `4 -> 8`

Interpretation:

- At `+1.0%`, targeted bits improved over the `R2_S3B` baseline.
- The allocator favored `self_attn.o_proj` matrices on gain-per-byte, even though several `mlp.down_proj` layers had larger raw damage.

Use this file for run-specific observations or anomalies.
