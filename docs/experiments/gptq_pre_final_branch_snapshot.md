# GPTQ Pre-Final-Branch Snapshot

This file freezes the trusted GPTQ state before any optional final bounded branch.

It is intended to make later interpretation cleaner by separating:

- the trusted current GPTQ story
- any optional multi-bit bits extension

## Trusted Frontier Snapshot

### `Qwen/Qwen3-1.7B-Base`

| Run | Policy | Memory (bytes) | Perplexity | Notes |
|-----|--------|----------------|------------|-------|
| `R3_Q17B` | GPTQ baseline | `1356968233` | `15.9137` | trusted baseline |
| `G2B02_Q17B` | bits `+1.0%` | `1364308265` | `15.8993` | first bits point |
| `G2B03_Q17B` | bits `+2.0%` | `1383182633` | `15.8914` | best matrix bits point |
| `G2R02_Q17B` | rank `+1.0%` | `1360965929` | `15.8823` | best rank point |
| `G2R03_Q17B` | rank `+2.0%` | `1360965929` | `15.8823` | saturated; unchanged |
| `H2R02M_Q17B` | matrix hybrid | `1368305961` | `15.8962` | best clean matrix hybrid |

Trusted policy ordering:

- rank-only best
- bits-only second
- hybrid third

### `HuggingFaceTB/SmolLM3-3B-Base`

| Run | Policy | Memory (bytes) | Perplexity | Notes |
|-----|--------|----------------|------------|-------|
| `R3_S3B` | GPTQ baseline | `1990237781` | `11.5366` | trusted baseline |
| `G3B02_S3B` | bits `+1.0%` | `1997577813` | `11.5483` | regressed less than rank |
| `G3R02_S3B` | rank `+1.0%` | `2000477781` | `11.6482` | worst of the pair |

Trusted interpretation:

- neither bits nor rank improved over baseline
- bits regressed less than rank

### `Qwen/Qwen3-8B-Base`

| Run | Policy | Memory (bytes) | Perplexity | Notes |
|-----|--------|----------------|------------|-------|
| `R3_Q8B` | GPTQ baseline | `6103975811` | `11.7970` | trusted baseline |
| `G2B02_Q8B` | bits `+1.0%` | `6175311591` | `11.7823` | best current `8B` point |
| `G2B03_Q8B` | bits `+2.0%` | `6175278979` | `11.8024` | worse than `+1.0%` |
| `G2R02_Q8B` | rank `+1.0%` | `6109349763` | `11.7962` | effectively flat |
| `G2R03_Q8B` | rank `+2.0%` | `6109349763` | `11.7962` | saturated; unchanged |
| `H2R02_Q8B` | hybrid | `6180685543` | `11.7895` | between bits and rank |

Trusted policy ordering:

- bits-only best
- hybrid second
- rank-only third

## Trusted Supporting Results

These runs are useful context but are not the main frozen frontier:

### `1.7B` richer / structural follow-ups

- `G2B02RB128_Q17B`: `15.8970`
- `G2B02CB128_Q17B`: `15.9171`
- `G2R02F_Q17B`: `15.9073`
- `G2R02RB128_Q17B`: `15.9034`
- `G2R02CB128_Q17B`: `15.9004`
- `G2R02GRP_Q17B`: `15.9224`
- `H2R02RB128_Q17B`: `15.8989`

Interpretation:

- richer bits can help at `1.7B`, but the story is sensitive to design
- fine-grained / grouped / structural rank variants did not beat the original matrix-rank point
- hybrid was useful, but not dominant

### `8B` richer bits follow-up

- `G2B02RB128_Q8B`: `11.7954`

Interpretation:

- finer row-block bits were valid
- but matrix-level bits remained better at `8B`

## Hardware Snapshot

Trusted working GPTQ paths:

- `1.7B`: `A10G` is sufficient for the validated frontier runs
- `3B`: `A100`-class Modal path used for the validated pair
- `8B`: `A100-80GB` with `device_map: "single"` is the trusted path

Known non-preferred paths:

- `8B` GPTQ with `device_map: "auto"` caused offload-path failures
- `8B` GPTQ on smaller single-device setups was not stable enough for the final baseline work

## Code / Config State

This snapshot assumes the current repo state after:

- GPTQ bring-up fixes
- finite-logit validation
- matrix policy comparison completion
- shared-family rank explicitly deferred

The active bounded plan from this point is:

- [gptq_closure_plan.md](/home/seshu/Documents/Python/llm-decomposition/docs/roadmap/gptq_closure_plan.md)

## Decision Use

Use this snapshot as the baseline for the final bounded choice:

1. close GPTQ now, or
2. run the multi-bit bits-policy branch on `1.7B` first

Any future result should be interpreted as a refinement of this frozen state, not a replacement for it unless it materially changes the frontier ordering.

## Update After Optional Branch Execution

The optional `1.7B` multi-bit bits-policy branch has now been executed:

- `MB1_Q17B`: `15.9097`

Interpretation:

- the allocator mostly chose cheap `4->5` upgrades
- the result did not beat the current best bits point `G2B02_Q17B` (`15.8993`)
- it also remained clearly below the current best rank point `G2R02_Q17B` (`15.8823`)

So the stop rule fired:

- `MB2_Q17B` is not justified
- `8B` multi-bit validation is not justified
- the GPTQ experimental branch is now ready for synthesis
