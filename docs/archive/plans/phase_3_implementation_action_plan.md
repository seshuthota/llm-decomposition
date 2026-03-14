# Phase 3 Implementation Action Plan

## Purpose

Phase 3 is not a conclusion phase. It is a **regime-mapping phase**.

The goal is to determine how the preferred use of marginal memory varies across:

- **model scale**,
- **quantizer type**,
- **budget size**, and
- **allocation granularity**.

The central question remains:

> Under a fixed memory budget, should the next byte go to extra precision, low-rank residual repair, or a hybrid of both?

Phase 1 and Phase 2 showed that the answer can change depending on setup. Therefore, Phase 3 must be structured to map those setup-dependent regimes before any broad conclusion is made.

---

# 1. Phase 3 Goals

## Primary Goals

1. Build a **cross-scale frontier map** for bits-only and rank-only allocation under RTN.
2. Test whether the frontier ordering changes under a stronger quantizer such as GPTQ.
3. Determine whether hybrid second-stage repair adds value only after the best bit allocations are already taken.
4. Identify the main explanatory variables behind frontier changes:
   - layer family,
   - residual structure,
   - activation-space sensitivity,
   - and allocator action-space granularity.

## Secondary Goals

1. Measure how stable allocator choices are across calibration subsets.
2. Test whether activation-space signals outperform cheaper weight-space proxies.
3. Log basic latency and implementation cost so quality improvements can be interpreted in deployment context.

## Non-Goals for Phase 3

The following are explicitly out of scope unless Phase 3 finishes early:

- gradient-based residual fitting,
- extensive downstream benchmark suites,
- many exotic quantizers,
- very large 14B+ models as a first priority,
- and complex global optimization methods beyond greedy or simple knapsack-style search.

---

# 2. Core Hypotheses to Test

Phase 3 is designed to test the following hypotheses rather than assume them:

## H1 — Scale Dependence
The relative value of extra bits versus extra rank changes with model size.

## H2 — Quantizer Dependence
The bits-versus-rank frontier depends on how the base quantizer shapes the residual error.

## H3 — Action-Space Dependence
The apparent winner between bits and rank depends strongly on the granularity of available allocation actions.

## H4 — Hybrid as Second-Stage Repair
Low-rank residual repair may be most useful after the highest-value bit upgrades are already exhausted.

## H5 — Better Signals Improve Allocation
Activation-space signals should select better actions than cheap weight-space metrics.

---

# 3. Phase 3 Structure

Phase 3 will run in three major blocks:

1. **Phase 3A — RTN Regime Map**
2. **Phase 3B — Quantizer Transfer**
3. **Phase 3C — Hybrid Second-Stage Testing**

Each block has its own deliverables and decision gates.

---

# 4. Workstream Breakdown

## Workstream A — Infrastructure and Experiment Discipline

### Objective
Create a stable execution and logging environment so that all Phase 3 comparisons are reproducible and directly comparable.

### Tasks

1. Freeze the evaluation harness.
   - same perplexity pipeline
   - same calibration pipeline
   - same memory accounting
   - same run metadata format
   - same naming scheme

2. Standardize run schemas.
   Every run should store:
   - model
   - quantizer
   - budget target
   - actual memory bytes
   - method type
   - allocation policy
   - candidate signal type
   - selected actions
   - per-action byte cost
   - realized perplexity
   - latency notes if available

3. Create a canonical action log format.
   For every chosen action, record:
   - action type (`bit_upgrade`, `rank_add`, `hybrid_split`)
   - target matrix
   - target group if applicable
   - delta budget used
   - predicted gain
   - realized gain
   - cumulative budget

4. Standardize model and dataset manifests.
   Keep one manifest that defines:
   - calibration subset names
   - eval subset names
   - allowed model ladder
   - supported quantizers
   - allowed budget points

### Deliverables

- `phase3_run_schema.json`
- `phase3_action_schema.json`
- `phase3_dataset_manifest.yaml`
- `phase3_model_manifest.yaml`

### Exit Criteria

- all future runs can be reproduced from config only
- all results can be aggregated by a common parser

---

## Workstream B — Model Ladder Expansion

### Objective
Move beyond 0.6B and 1.7B to establish a meaningful scale curve.

### Recommended Model Ladder

Use the following sequence:

- `Qwen/Qwen3-0.6B-Base` — fast debugging anchor
- `Qwen/Qwen3-1.7B-Base` — first scale transition
- one **3B–4B** class model — bridge scale
- one **7B–8B** class model — main validation scale
- **14B** only if resources allow and earlier results remain ambiguous

### Priority Order

1. 3B/4B
2. 7B/8B
3. optional 14B

### Tasks

1. Validate that the harness runs end-to-end on each target model.
2. Confirm RTN baseline memory and perplexity for each size.
3. Confirm that allocation actions are compatible with the model’s layer layout.
4. Update layer-family mapping so comparisons remain aligned across model sizes.

### Deliverables

- one validated RTN baseline per model size
- one layer-family map per model
- one model-readiness note per target size

### Exit Criteria

- RTN baseline exists and is reproducible for each selected model size
- action inventory is generated for each size

---

## Workstream C — Bits Action-Space Expansion

### Objective
Make the bits-only frontier more expressive and less vulnerable to coarse-action artifacts.

### Required Action Types

Implement bits allocation at multiple granularities:

1. **Matrix-level upgrades**
   - `mlp.down_proj`
   - `mlp.up_proj`
   - `mlp.gate_proj`
   - `self_attn.o_proj`
   - optional `q/k/v`

2. **Groupwise or blockwise upgrades**
   - output-channel groups
   - input-channel groups
   - contiguous row/column blocks
   - quantizer-native groups where possible

3. **Intermediate precision levels** if implementation supports them
   - 4 -> 5 bit
   - 4 -> 6 bit
   - 4 -> 8 bit

### Tasks

1. Build matrix-level action generation.
2. Add groupwise action generation.
3. Add memory-cost calculator for each action.
4. Add support for intermediate precision levels where feasible.
5. Validate that action costs are correct and budget matching is accurate.

### Deliverables

- `candidate_bit_actions.jsonl` per model and quantizer
- action-cost validation notebook or script
- bits action-space README

### Exit Criteria

- bits-only frontier is smooth across multiple budgets
- small-budget points are no longer mostly uninformative

---

## Workstream D — Rank Action-Space Standardization

### Objective
Ensure rank allocation remains expressive but disciplined and comparable across models.

### Required Rank Actions

For selected sensitive matrices, support incremental rank additions such as:

- rank +4
- rank +8
- rank +16
- rank +32

Where feasible, use the same matrix families tested on the bits side.

### Tasks

1. Restrict initial rank action generation to a candidate pool of promising matrices.
2. Support incremental rank chunks rather than one-shot rank assignments.
3. Verify that memory accounting includes all residual parameters.
4. Log per-rank marginal gain curves.

### Deliverables

- `candidate_rank_actions.jsonl`
- rank gain-curve logs per model
- rank-action validation notes

### Exit Criteria

- rank actions spend budget predictably
- rank frontier can be generated at all budget points without under-spending artifacts

---

## Workstream E — Allocation Signal Comparison

### Objective
Determine which signals best predict useful actions.

### Signals to Compare

1. **Weight-space residual norms**
2. **Activation-space output deviation**
3. **Direct marginal probing** on a small calibration evaluation set

### Tasks

1. Implement signal extraction for all candidate actions.
2. Rank actions by each signal.
3. Compare signal ranking versus realized perplexity gain.
4. Measure top-k hit rate and rank correlation between signal ranking and realized gain.

### Deliverables

- signal correlation tables
- top-k hit rate plots
- one short note on proxy quality versus compute cost

### Exit Criteria

- a preferred signal is selected for the main allocator
- signal choice is justified empirically, not by intuition alone

---

## Workstream F — RTN Regime Map

### Objective
Build the main cross-scale map under RTN before adding quantizer variation.

### Methods to Compare

At minimum:

- bits-only frontier
- rank-only frontier
- optional hybrid at selected points

### Budget Points

Recommended budget ladder:

- +0.25%
- +0.5%
- +1.0%
- +2.0%
- +4.0%

If some small points remain too coarse on larger models, keep them for completeness but mark them accordingly.

### Tasks

1. Generate RTN baselines for all selected model sizes.
2. Run bits-only frontier for all budget points.
3. Run rank-only frontier for all budget points.
4. Aggregate results into quality-versus-memory plots.
5. Record layer-family selections at each frontier point.

### Core Questions

- Does the frontier ordering change with scale?
- At what size, if any, does the winner flip?
- Do bits and rank concentrate on the same layer families?
- Does rank saturate earlier or later at larger scale?

### Deliverables

- RTN frontier plots by model size
- RTN layer-family selection plots
- RTN regime summary memo

### Exit Criteria

- at least 3 meaningful model sizes have comparable RTN frontiers
- frontier behavior across scale is clear enough to motivate transfer testing

---

## Workstream G — Residual Structure Profiling

### Objective
Explain frontier changes rather than only report them.

### Measurements

For selected matrices and models, profile:

- residual singular value decay
- cumulative energy captured by top-rank approximations
- activation-space amplification of residual error
- difference in residual structure across scale

### Tasks

1. Profile top sensitive matrices chosen by bits and rank.
2. Compare residual spectra across 0.6B, 1.7B, and larger models.
3. Check whether winning rank actions correspond to matrices with more compressible residuals.
4. Relate residual structure back to frontier results.

### Deliverables

- residual spectrum plots
- low-rankness summary tables
- explanatory analysis memo

### Exit Criteria

- at least one plausible mechanistic explanation exists for any observed scale flip or frontier change

---

## Workstream H — Calibration Stability Tests

### Objective
Check whether allocation conclusions are robust to calibration data choice.

### Tasks

1. Run selected key frontier points with at least two calibration subsets.
2. Run selected points at two calibration sizes, e.g. small and medium.
3. Measure whether selected actions remain stable.
4. Measure whether frontier ordering changes materially.

### Deliverables

- calibration sensitivity table
- action-overlap statistics across calibration variants

### Exit Criteria

- key conclusions do not depend on one fragile calibration subset
- or fragility is explicitly documented if it exists

---

## Workstream I — Quantizer Transfer (GPTQ First)

### Objective
Test whether the RTN conclusions survive a stronger PTQ method.

### Priority Quantizers

1. **GPTQ**
2. optional **AWQ** if infrastructure and hardware permit

### Transfer Scope

Do not repeat every RTN run. Repeat the most informative ones:

- selected model sizes: 1.7B and one larger model
- selected budget points: +1.0%, +2.0%, +4.0%
- selected methods: bits-only and rank-only first
- hybrid later only if needed

### Tasks

1. Stabilize GPTQ bring-up on the target machine.
2. Reuse the same action schemas and allocator logic.
3. Run baseline GPTQ quantized models.
4. Repeat key frontier points.
5. Compare frontier ordering and layer-family patterns against RTN.

### Deliverables

- GPTQ baseline notes
- GPTQ matched frontier table
- RTN versus GPTQ comparison memo

### Exit Criteria

- at least one trustworthy GPTQ frontier exists for one medium or larger model
- quantizer dependence can be stated with evidence rather than speculation

---

## Workstream J — Hybrid Second-Stage Testing

### Objective
Test the original “next byte” question in the most meaningful way.

### Design

For selected stable regimes:

1. start from quantized baseline
2. allocate best bits-only plan under budget `B`
3. give an extra mini-budget `Δ`
4. compare three choices for spending `Δ`:
   - more bits
   - more rank
   - mixed split

### When to Run This

Only after:

- bits-only frontier is stable
- rank-only frontier is stable
- at least one cross-scale or cross-quantizer regime is understood

### Tasks

1. Define `B` and `Δ` budgets for selected models.
2. Freeze the best bits-only state at `B`.
3. Run second-stage alternatives.
4. Compare marginal gains per byte.
5. Record whether rank becomes useful only after bits saturate.

### Deliverables

- hybrid second-stage comparison table
- marginal next-byte plots
- hybrid interpretation memo

### Exit Criteria

- hybrid is either shown to matter in at least one regime, or shown to be unnecessary under tested conditions

---

## Workstream K — Evaluation Beyond Perplexity

### Objective
Ensure that conclusions are not an artifact of one metric.

### Tasks

1. Keep perplexity as the primary metric.
2. Add a compact secondary evaluation suite such as a small zero-shot task set.
3. Run this only on the most important frontier points.
4. Check whether perplexity improvements track task behavior.

### Deliverables

- compact downstream summary table
- note on alignment or mismatch between perplexity and task accuracy

### Exit Criteria

- no major contradiction between perplexity and secondary task behavior remains unexplored

---

## Workstream L — Latency and Practical Cost Logging

### Objective
Keep deployment interpretation honest.

### Tasks

1. Log total added parameter bytes.
2. Log number of upgraded or repaired matrices.
3. Where feasible, measure rough inference latency change.
4. Record implementation complexity notes.

### Deliverables

- practical-cost summary table
- latency notes for key frontier points

### Exit Criteria

- every major result includes enough practical context to interpret deployment tradeoffs

---

# 5. Recommended Execution Order

## Stage 0 — Freeze Infrastructure

1. finalize run schema
2. finalize action schema
3. freeze evaluation and accounting
4. validate aggregation scripts

## Stage 1 — RTN Cross-Scale Expansion

1. bring up 3B/4B RTN baseline
2. bring up 7B/8B RTN baseline
3. run bits-only frontiers
4. run rank-only frontiers
5. aggregate scale map

## Stage 2 — Explanatory Analysis

1. residual structure profiling
2. signal-quality comparison
3. calibration sensitivity tests

## Stage 3 — Quantizer Transfer

1. stabilize GPTQ backend
2. rerun selected key points
3. compare RTN versus GPTQ

## Stage 4 — Hybrid Second Stage

1. select stable regimes
2. run next-byte hybrid tests
3. interpret when hybrid is useful

## Stage 5 — Final Analysis and Writing

1. build regime summary tables
2. build frontier plots
3. draft conclusions only after all above stages are complete

---

# 6. Phase 3 Experiment Matrix

## Core RTN Matrix

For each selected model size:

- Quantizer: RTN 4-bit
- Methods:
  - bits-only
  - rank-only
- Budgets:
  - +0.25%
  - +0.5%
  - +1.0%
  - +2.0%
  - +4.0%
- Signals:
  - weight-space
  - activation-space
  - direct marginal probing on selected subsets

## GPTQ Transfer Matrix

For selected model sizes only:

- Quantizer: GPTQ 4-bit
- Methods:
  - bits-only
  - rank-only
- Budgets:
  - +1.0%
  - +2.0%
  - +4.0%
- Signal:
  - best-performing Phase 3 signal

## Hybrid Matrix

For selected stable regimes only:

- Start state: best bits-only allocation at budget `B`
- Extra mini-budget: `Δ`
- Compare:
  - more bits
  - more rank
  - mixed split

---

# 7. Decision Gates

## Gate 1 — RTN Scale Readiness
Proceed only if:

- at least three model sizes have clean RTN baselines
- action inventories are valid
- frontiers can be generated reproducibly

## Gate 2 — Explanatory Readiness
Proceed only if:

- preferred allocation signal is identified
- at least one residual-structure analysis is complete

## Gate 3 — GPTQ Readiness
Proceed only if:

- GPTQ bring-up is stable on target hardware
- memory accounting matches RTN conventions

## Gate 4 — Hybrid Readiness
Proceed only if:

- bits-only and rank-only frontiers are both mature
- at least one regime difference is already clear

## Gate 5 — Conclusion Readiness
Only write broad conclusions if:

- more than two model sizes are tested,
- more than one quantizer is tested,
- frontier results are stable across calibration variants,
- and at least one explanatory analysis supports the observed trends.

---

# 8. Success Criteria for Phase 3

Phase 3 will be considered successful if it achieves the following:

1. A credible **cross-scale RTN frontier map** exists.
2. At least one trustworthy **GPTQ transfer result** exists.
3. The project can identify at least one strong driver of frontier changes, such as:
   - scale,
   - quantizer,
   - layer family,
   - or action-space granularity.
4. The project can state at least one evidence-backed regime claim such as:
   - bits dominate at larger scale under RTN,
   - rank dominates only under certain action-space conditions,
   - or hybrid matters only after bits saturate.

---

# 9. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Scaling experiments become too expensive too early. | Prioritize 3B/4B and 7B/8B before any 14B attempt. |
| GPTQ remains unstable on available hardware. | Finish RTN regime mapping first and keep GPTQ as the main transfer target rather than blocking all progress on it. |
| Bits and rank remain hard to compare fairly. | Make action-space definitions explicit and report all conclusions as conditional on the tested allocation granularity. |
| Calibration choice changes allocator decisions too much. | Add dedicated calibration stability tests before making strong claims. |
| Perplexity improvements do not reflect downstream quality. | Add a compact secondary task check for key frontier points. |
| Too many experiments create analysis sprawl. | Use decision gates and freeze the next step only after each block produces interpretable results. |

---

# 10. Concrete Near-Term Checklist

## Immediate Next 10 Tasks

1. Freeze the Phase 3 run schema and action schema.
2. Finalize the model ladder for 3B/4B and 7B/8B targets.
3. Generate RTN baselines for the next model size.
4. Expand bits action space beyond current coarse matrix-level upgrades where feasible.
5. Revalidate rank action accounting on larger models.
6. Implement action ranking comparison across weight-space and activation-space signals.
7. Run RTN bits-only frontier on the next scale.
8. Run RTN rank-only frontier on the next scale.
9. Produce first cross-scale frontier plot.
10. Prepare GPTQ bring-up checklist for the remote machine.

---

# 11. Expected Outputs

By the end of Phase 3, the project should produce:

- a cross-scale bits-versus-rank frontier map,
- at least one cross-quantizer comparison,
- gain-per-byte plots,
- layer-family selection analyses,
- residual structure analyses,
- calibration stability summaries,
- and a much more defensible final interpretation of where the next byte should go.

---

# 12. One-Sentence Phase 3 Summary

Phase 3 will determine how the best use of extra memory in a quantized language model changes with scale, quantizer, budget, and allocation granularity, so that later conclusions about bits, rank, or hybrid repair are evidence-based rather than premature.

