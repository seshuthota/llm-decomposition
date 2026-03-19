## Activation-vs-Weight Allocator Ablation

This experiment closes Research Objective 4 for the current paper scope:

> does activation-space profiling justify its extra cost relative to a cheaper
> weight-space proxy when allocating the same extra-memory budget?

Scope:
- quantizer: GPTQ
- model: `Qwen/Qwen3-1.7B-Base`
- budget: `+1.0%`
- policies:
  - targeted bits
  - targeted rank
- proxy families:
  - `activation`
  - `weight`

Artifacts:
- summary table: [proxy_ablation_q17b_summary.csv](/home/seshu/Documents/Python/llm-decomposition/results/analysis/proxy_ablation_q17b_summary.csv)
- selection diff: [proxy_ablation_q17b_selection_diff.json](/home/seshu/Documents/Python/llm-decomposition/results/analysis/proxy_ablation_q17b_selection_diff.json)

### Runs

| Run | Policy | Proxy | Perplexity | Profiling Wall Time (s) | Selection Profiling (s) |
| --- | --- | --- | ---: | ---: | ---: |
| `G2B02A_Q17B` | bits | activation | `15.8993` | `12.49` | `15.12` |
| `G2B02W_Q17B` | bits | weight | `15.8993` | `0.0004` | `0.0004` |
| `G2R02A_Q17B` | rank | activation | `15.9224` | `10.08` | `12.52` |
| `G2R02W_Q17B` | rank | weight | `15.8823` | `0.0004` | `0.0004` |

### Observations

1. Bits allocation is effectively invariant at this setting.
The activation and weight proxies choose the same 7 upgraded matrices and land on
the same final perplexity (`15.8993`). Activation scoring only changes the order
and relative scores of the chosen `v_proj` and `k_proj` upgrades.

2. Rank allocation is not invariant.
Activation and weight proxies converge to the same final 12 repaired layers and
the same final per-layer rank caps, but they choose a materially different
increment ordering. That difference is enough to move final perplexity from
`15.8823` (weight) to `15.9224` (activation).

3. Activation profiling is expensive at `1.7B`.
For both bits and rank, fresh activation-based selection adds about `10-15`
seconds of profiling/selection work, while the weight proxy path is effectively
free at this scale (`~0.0004 s`).

### Selection Differences

Bits:
- shared selected targets:
  - `model.layers.19.self_attn.v_proj`
  - `model.layers.20.self_attn.v_proj`
  - `model.layers.21.self_attn.v_proj`
  - `model.layers.22.self_attn.v_proj`
  - `model.layers.23.self_attn.v_proj`
  - `model.layers.8.self_attn.k_proj`
  - `model.layers.9.self_attn.k_proj`
- conclusion: no target-set difference, only score/order differences

Rank:
- final selected layer set is identical across activation and weight proxies
- final selected per-layer ranks are identical across activation and weight proxies
- conclusion: the quality difference comes from the ordering of incremental rank
  actions, not from a different final layer family

### Conclusion

For the current `1.7B` GPTQ paper scope, activation-space profiling is **not**
justified by allocation quality:

- bits: no gain at all over the cheaper weight proxy
- rank: materially worse than the cheaper weight proxy
- cost: activation profiling adds measurable wall time while the weight proxy is
  effectively free

Paper-facing statement:

> At `Qwen3-1.7B` under GPTQ, weight-space error is the better default allocator
> proxy. Activation-space profiling adds substantial selection cost, does not
> improve bits allocation, and degrades targeted-rank allocation at the matched
> `+1.0%` budget.

Bounded decision:
- the `8B` weight-proxy follow-up remains optional and is not required for the
  current paper MVP because the `1.7B` result already answers the allocator
  ablation question in the intended scope.
