# SmolLM3-3B GPTQ Validation Plan

## Goal

Validate whether the `1.7B` GPTQ result transfers to a larger model before spending more budget on an `8B` run.

The concrete question is:

> Under a `GPTQ 4-bit` baseline on `HuggingFaceTB/SmolLM3-3B-Base`, does the next small memory budget go to targeted bits or targeted rank?

## Execution Order

1. Stage `HuggingFaceTB/SmolLM3-3B-Base` to the Modal model volume.
2. Run `R3_S3B` on `A10G`.
3. Build `smollm3_3b_gptq_top12_candidate_pool.json` from `R3_S3B/layer_errors.json`.
4. Run the first matched pair on `A10G`:
   - `G3B02_S3B`
   - `G3R02_S3B`
5. Compare the first matched pair.
6. Only run the `+2.0%` pair if the first point is too close or ambiguous:
   - `G3B03_S3B`
   - `G3R03_S3B`

## GPU Choice

- Use `A10G` as the default smallest viable GPU for `3B` GPTQ.
- Only escalate to `A100` if the baseline or transfer runs fail for capacity/runtime reasons rather than code issues.

## Storage

- Model subpath on Modal: `/smollm3-3b-base`
- Results phase directories:
  - `results/modal/smollm3_3b_gptq_baselines`
  - `results/modal/smollm3_3b_gptq_transfer`

## Decision Rule

- If `G3B02_S3B` and `G3R02_S3B` clearly separate, stop there and record the transfer result.
- If the pair is close or the rank action space saturates too early, run the `+2.0%` pair before deciding whether `8B` is worth the cost.
