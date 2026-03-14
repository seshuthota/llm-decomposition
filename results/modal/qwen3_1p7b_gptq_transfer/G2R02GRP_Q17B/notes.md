# G2R02GRP_Q17B Notes

- Model: `/__modal/volumes/vo-34bHdyikTfjM8k8hJ7wQXB/qwen3-1.7b-base`
- Method: `targeted_svd_rank`
- Config source: `/root/project/configs/scaleup_1p7b_gptq/g2r02_grouped_qwen3_1p7b_gptq_rank_activation_1p0pct.json`
- Status: completed

Result summary:

- perplexity: `15.9224`
- memory total bytes: `1,360,965,929`
- repair bytes: `3,997,696`

Observations:

- This run used a family-aware grouped allocator that forced two balanced rounds across major layer families before falling back to greedy selection.
- The selected actions did spread early budget across `self_attn.k_proj`, `self_attn.v_proj`, `mlp.down_proj`, and `mlp.gate_proj`.
- The run was numerically stable on `A10G`.

Interpretation:

- The grouped allocator achieved the intended behavioral change, but quality regressed sharply relative to the original matrix-level rank point.
- This is strong evidence that simple family balancing is not the missing ingredient in the current GPTQ rank policy.
- The next rank branch should be a more fundamental redesign, not another sequencing or balancing heuristic on top of the same action set.
