# Phase 3 Modal Execution

This file is the operational Phase 3 run order for the Modal-backed RTN scale map.

## Scope

Phase 3 is currently defined as:

1. `0.6B` RTN anchor: already complete
2. `1.7B` RTN anchor: already complete
3. `3B` bridge-scale RTN run set
4. `8B` validation-scale RTN run set

`GPTQ` is explicitly excluded from the active execution path until the backend is numerically stable.

## Model Order

### 1. Bridge scale

- model: `HuggingFaceTB/SmolLM3-3B-Base`
- baseline GPU: `A10G`
- first matched pair GPU: `A10G`
- second matched pair GPU: `A10G` unless runtime or memory shows a clear need to escalate

### 2. Validation scale

- model: `Qwen/Qwen3-8B-Base`
- baseline GPU: `A100`
- first matched pair GPU: `A100`
- second matched pair GPU: `A100` only if needed

## Execution Sequence

### SmolLM3 3B

1. staged directly to the Modal model volume
2. baseline completed on `A10G`:
   - `R2_S3B` perplexity: `47.9169`
3. candidate pool built from `results/modal/smollm3_3b_baselines/R2_S3B/layer_errors.json`
4. matched `+1.0%` pair completed on `A10G`:
   - `P3B02_S3B` perplexity: `47.4955`
   - `P3R02_S3B` perplexity: `47.9833`
5. decision:
   - targeted bits won clearly at the first matched budget point
   - skip the `+2.0%` pair for this model to conserve GPU time
6. proceed to `Qwen3-8B`

### Qwen3 8B

1. staged directly to the Modal model volume
2. baseline completed on `A100`:
   - `R2_Q8B` perplexity: `16.1939`
3. candidate pool built from `results/modal/qwen3_8b_baselines/R2_Q8B/layer_errors.json`
4. matched `+1.0%` pair completed on `A100`:
   - `P3B02_Q8B` perplexity: `16.1429`
   - `P3R02_Q8B` perplexity: `16.2035`
5. decision:
   - targeted bits won clearly at the first matched budget point
   - skip the `+2.0%` pair for this model as well
6. Phase 3 RTN scale map is complete

## Decision Rule

At each new model size:

- if the first matched pair is clearly ordered, record that result and move on
- if the first matched pair is close enough to be unstable, run the `+2.0%` pair
- if a method saturates early, note that and do not spend extra GPU time on redundant follow-up

## Cost Discipline

- do not use `A100` unless the smaller GPU is likely to fail or has already proven too slow/unreliable
- do not keep multiple model snapshots in the Modal volume longer than necessary
- do not launch paired runs in parallel; Phase 3 is sequenced one run at a time
