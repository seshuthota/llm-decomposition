# Detailed Quantization-vs-SVD Report

This is the full project report for the bounded study from Phase 1 through the final GPTQ endgame stop.

It is written against the initial project framing in:

- [initial_proposal.md](/home/seshu/Documents/Python/llm-decomposition/docs/proposals/initial_proposal.md)
- [current_proposal.md](/home/seshu/Documents/Python/llm-decomposition/docs/proposals/current_proposal.md)

and synthesizes what was actually learned from the completed experiments.

## 1. Executive Summary

The project started with a hybrid-compression question:

> If a quantized language model is given a small extra memory budget, should that budget be spent on more quantization bits or on SVD-style low-rank repair?

The final result is not a universal winner. It is a regime map.

Final bounded conclusions:

| Quantizer | Model | Baseline PPL | Best Bits PPL | Best Rank PPL | Best Hybrid PPL | Winner |
|-----------|-------|--------------|---------------|---------------|-----------------|--------|
| `RTN` | `Qwen3-0.6B` | `30.5169` | `30.2506` | `29.7223` | n/a | rank |
| `RTN` | `Qwen3-1.7B` | `21.3102` | `21.1505` | `21.2971` | n/a | bits |
| `RTN` | `SmolLM3-3B` | `47.9169` | `47.4955` | `47.9833` | n/a | bits |
| `RTN` | `Qwen3-8B` | `16.1939` | `16.1429` | `16.2035` | n/a | bits |
| `GPTQ` | `Qwen3-1.7B` | `15.9137` | `15.8914` | `15.8823` | `15.8962` | rank |
| `GPTQ` | `SmolLM3-3B` | `11.5366` | `11.5483` | `11.6482` | n/a | mixed / neutral |
| `GPTQ` | `Qwen3-8B` | `11.7970` | `11.7823` | `11.7962` | `11.7895` | bits |

So the best global summary is:

- there is no universal “bits always win” or “rank always wins” rule
- the decision changes with quantizer, scale, and action-space design
- this project is best understood as a **decision-frontier study under fixed memory budgets**

## 2. Original Plan vs. Final Outcome

### 2.1 Initial research formulation

The original project plan framed the model as:

\[
W_i \approx Q_i + A_i B_i
\]

where:

- \(Q_i\) is the quantized weight matrix for layer \(i\)
- \(A_i B_i\) is a low-rank repair term

The allocation problem was:

\[
\max_{a \in \mathcal{A}} \sum_{a} \widehat{\Delta q}(a)
\quad
\text{s.t.}
\quad
\sum_{a} c(a) \le B
\]

where:

- \(a\) is a candidate action
- \(c(a)\) is its byte cost
- \(\widehat{\Delta q}(a)\) is its estimated quality gain
- \(B\) is the extra-memory budget

The proposal expected two things:

1. non-uniform bit allocation would often be the strongest first use of memory
2. low-rank repair might still matter after the most important bit upgrades

### 2.2 What the experiments changed

What actually happened:

- uniform low-rank repair was weaker than a fair bits-only comparator
- targeted rank could still win in some regimes
- targeted bits dominated in others
- GPTQ did not simply mirror RTN
- bounded action-space refinements changed details, but not enough to create a universal winner

So the final contribution is not “hybrid compression wins.” It is:

- a bounded empirical map of **when the next byte should go to bits or rank**

## 3. Experimental Protocol

### 3.1 Dataset and metric

Main evaluation setup:

- calibration dataset: `WikiText-2` train split
- evaluation dataset: `WikiText-2` test split
- main metric: perplexity
- secondary metrics:
  - latency per token
  - total memory in bytes
  - repair bytes / upgrade bytes

### 3.2 Core comparison rule

All important comparisons were made under a fixed extra-memory budget:

- baseline compressed model first
- then matched-budget bits-only or rank-only follow-ups
- and, later, bounded hybrid follow-ups

### 3.3 Platforms used

- local workstation for early `0.6B` RTN
- Modal as the main execution platform for scale-up and GPTQ
- Kaggle for RTN reproduction and GPTQ environment debugging

## 4. Phase 1: Local RTN Bring-Up and Uniform Repair

Phase 1 answered a narrow question:

> is uniform low-rank repair strong enough to be the project mainline?

### 4.1 Phase 1 results

| Run | Method | Memory (bytes) | Perplexity | Takeaway |
|-----|--------|----------------|------------|----------|
| `R1` | full precision | `1192099840` | `16.8447` | reference |
| `R2` | `RTN 4-bit` | `307304448` | `30.5169` | quantization cliff is large |
| `R4` | uniform rank 4 | `307550208` | `30.5577` | worse than RTN baseline |
| `R5` | uniform rank 8 | `307795968` | `30.4654` | slight recovery |
| `R6` | uniform rank 16 | `308287488` | `30.3155` | clearer recovery |
| `R7` | uniform rank 32 | `309270528` | `29.9848` | best uniform repair point |
| `R11` | equal-budget bits comparator | `308877312` | `28.8618` | beats uniform rank |

### 4.2 Phase 1 interpretation

- quantization damage was clearly non-uniform
- later `mlp.down_proj` and some `self_attn.o_proj` layers mattered most
- uniform low-rank repair helped only once rank became non-trivial
- but the fair bits-only comparator was already better

Phase 1 therefore killed the original “uniform low-rank repair is the mainline” idea.

### 4.3 Phase 1 plot

![Phase 1 Memory-Perplexity Trajectory](assets/phase1_memory_perplexity.png)

Interpretation:

- the uniform-rank branch improved gradually
- but the fair bits comparator moved the frontier more decisively

## 5. Phase 2: Local Targeted RTN on `Qwen3-0.6B`

Phase 2 replaced uniform repair with the correct comparison:

- same candidate pool
- same budget
- targeted bits vs targeted rank

### 5.1 Phase 2 results

| Run | Policy | Budget | Memory (bytes) | Perplexity |
|-----|--------|--------|----------------|------------|
| `R2` | RTN baseline | baseline | `307304448` | `30.5169` |
| `P2B02` | targeted bits | `+1.0%` | `309401600` | `30.4238` |
| `P2R02` | targeted rank | `+1.0%` | `310319104` | `29.7223` |
| `P2B03` | targeted bits | `+2.0%` | `312547328` | `30.2506` |
| `P2R03` | targeted rank | `+2.0%` | `310515712` | `29.7252` |

### 5.2 Phase 2 interpretation

- after fixing the rank allocator to spend budget incrementally, targeted rank won cleanly
- this was the first strong evidence that rank can win in some regimes

This result mattered because it re-opened the rank side after Phase 1 had seemingly favored bits.

## 6. Phase 3 RTN: Cross-Scale Regime Map

The natural next question was whether the `0.6B` RTN result transferred.

### 6.1 RTN cross-scale table

| Model | Baseline | Best Bits | Best Rank | Winner |
|-------|----------|-----------|-----------|--------|
| `Qwen3-0.6B` | `30.5169` | `30.2506` | `29.7223` | rank |
| `Qwen3-1.7B` | `21.3102` | `21.1505` | `21.2971` | bits |
| `SmolLM3-3B` | `47.9169` | `47.4955` | `47.9833` | bits |
| `Qwen3-8B` | `16.1939` | `16.1429` | `16.2035` | bits |

### 6.2 RTN scale plot

![RTN Cross-Scale Improvement vs Baseline](assets/rtn_cross_scale_deltas.png)

### 6.3 RTN interpretation

RTN produced the clearest scale transition in the whole project:

- smallest tested model: rank wins
- larger models: bits win

So under the current RTN matrix-level action space:

- targeted rank is not the general rule
- targeted bits dominate once model scale increases

## 7. GPTQ Bring-Up and Infrastructure Lessons

GPTQ was not just another experiment branch. It required real infrastructure recovery.

### 7.1 Main failures encountered

- local GPU compatibility limits
- Modal dependency and image-build failures
- invalid early baselines
- `NaN` perplexity on Kaggle smoke tests
- detached-run artifact persistence issues
- `hf_device_map` failures during packing
- packed-module replacement issues during targeted updates
- `8B` offload-path failures with `device_map: "auto"`

### 7.2 What fixed GPTQ

Key fixes:

- forced `float16`
- explicit finite-logit / finite-loss validation
- separate full-precision reference from the quantized model
- proper Modal result persistence
- inject `hf_device_map` before GPTQ packing
- replace selected packed modules with floating `nn.Linear` modules
- for `8B`, use `A100-80GB` with `device_map: "single"`

This matters scientifically because GPTQ only became interpretable after these fixes.

## 8. GPTQ Frontier Across Scale

### 8.1 GPTQ cross-scale table

| Model | Baseline | Best Bits | Best Rank | Best Hybrid | Winner |
|-------|----------|-----------|-----------|-------------|--------|
| `Qwen3-1.7B` | `15.9137` | `15.8914` | `15.8823` | `15.8962` | rank |
| `SmolLM3-3B` | `11.5366` | `11.5483` | `11.6482` | n/a | mixed / neutral |
| `Qwen3-8B` | `11.7970` | `11.7823` | `11.7962` | `11.7895` | bits |

### 8.2 GPTQ scale plot

![GPTQ Cross-Scale Improvement vs Baseline](assets/gptq_cross_scale_deltas.png)

### 8.3 GPTQ interpretation

GPTQ did not mirror RTN:

- `1.7B`: rank wins
- `3B`: neither helps; bits regress less than rank
- `8B`: bits win

So quantizer choice is a real scientific variable, not just an implementation detail.

## 9. GPTQ Policy Comparison

The bounded GPTQ policy-comparison branch filled in the missing “bits vs rank vs hybrid” picture.

### 9.1 `1.7B` GPTQ policy ordering

| Policy | Run | Perplexity | Order |
|--------|-----|------------|-------|
| rank-only | `G2R02_Q17B` | `15.8823` | 1 |
| bits-only | `G2B03_Q17B` | `15.8914` | 2 |
| hybrid | `H2R02M_Q17B` | `15.8962` | 3 |

### 9.2 `8B` GPTQ policy ordering

| Policy | Run | Perplexity | Order |
|--------|-----|------------|-------|
| bits-only | `G2B02_Q8B` | `11.7823` | 1 |
| hybrid | `H2R02_Q8B` | `11.7895` | 2 |
| rank-only | `G2R02_Q8B` | `11.7962` | 3 |

### 9.3 Policy plot

![GPTQ Policy Ordering by Scale](assets/gptq_policy_ordering.png)

### 9.4 Policy interpretation

This is one of the strongest results in the repo:

- the policy ordering itself changes with scale
- hybrid is useful, but not dominant
- there is still no universal winner even after fair policy completion

## 10. Bounded Follow-Ups and Why They Matter

The project did not stop at the first matrix-level frontier. It explicitly tested bounded objections.

### 10.1 What was tried

Bits-side follow-ups:

- row-block bits
- column-block bits
- multi-bit bits (`5/6/8`)

Rank-side follow-ups:

- finer rank ladders
- row-block rank
- column-block rank
- grouped family-aware rank

Policy follow-up:

- hybrid second-stage

### 10.2 What survived

Surviving conclusions:

- `1.7B` GPTQ still favors rank
- `8B` GPTQ still favors bits
- hybrid can help, but does not dominate

### 10.3 Final bounded gate: multi-bit bits

The final bounded branch was:

- `MB1_Q17B`: `15.9097`

Compared with:

- baseline `R3_Q17B`: `15.9137`
- best bits `G2B03_Q17B`: `15.8914`
- best rank `G2R02_Q17B`: `15.8823`

Interpretation:

- multi-bit bits improved slightly over baseline
- but did not beat the current best bits frontier
- and remained clearly behind rank

So the final bounded objection on the bits side failed to overturn the result.

## 11. Hypotheses: What Was Supported vs Not Supported

### 11.1 Supported

Supported strongly:

1. quantization damage is strongly non-uniform
2. equal-memory comparisons are essential; naive method comparisons are misleading
3. the best policy depends on regime
4. quantizer choice changes the frontier

### 11.2 Partly supported

Partly supported:

1. “bits are often the strongest first use of extra memory”
   - true in most RTN scale-up points
   - true at `8B` GPTQ
   - false at `1.7B` GPTQ

2. “low-rank repair may add value after key bit decisions”
   - sometimes true
   - hybrid was useful
   - but not dominant

### 11.3 Not established as a final claim

Not established cleanly:

1. activation-space allocation is definitively better than all weight-space proxies
   - activation signals were central to the working path
   - but the project did not isolate this as its own final clean causal claim

2. a new rank method family is necessary
   - bounded follow-ups did not justify opening shared-family rank inside this project scope

## 12. Main Lessons by Phase

### Phase 1

- uniform repair is not the mainline
- fair bits-only comparison matters immediately

### Phase 2

- targeted rank can win if the allocator is allowed to spend budget correctly

### Phase 3 RTN

- bits dominate outside the smallest RTN regime

### GPTQ

- GPTQ is a different frontier, not a re-run of RTN
- `1.7B` and `8B` disagree
- bounded refinements strengthened the regime-dependent interpretation

## 13. Final Project Interpretation

The completed project is best summarized by this statement:

> Under bounded post-training action spaces, the preferred use of extra memory is regime-dependent. RTN and GPTQ exhibit different scale-sensitive frontiers, and the better use of the next byte changes across model size, quantizer, and policy class.

That is a stronger and more defensible contribution than a simple “bits vs rank” winner claim.

## 14. Recommended Boundary for Future Work

If the work continues, it should be framed as a new phase rather than as unfinished cleanup.

Reasonable future extensions:

- larger models (`14B+`)
- a new quantizer family
- genuinely new rank-method families
- broader search over action-space design

Those are extension-project questions, not unfinished work inside the bounded study documented here.
