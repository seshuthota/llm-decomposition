This is a strong consolidation. The core framing is now much sharper: this is a **regime-mapping project**, not a “prove rank wins” or “prove bits win” project. Your own report supports that shift clearly, especially because the winner changes across `RTN` vs `GPTQ` and across scale. 

Here is the **detailed Phase 3 implementation action plan** I would recommend from this point.

# Phase 3 Implementation Action Plan

## 1. Phase 3 objective

The objective of Phase 3 is:

> determine whether the current GPTQ bits-vs-rank frontier is a true model/quantizer effect or mostly a limitation of the current matrix-level action space.

So Phase 3 should **not** focus on adding many more budget points to the current setup.
It should focus on:

* improving the GPTQ action space,
* rerunning the most informative scales,
* testing whether the conclusion is stable after that richer action space is introduced.

That matches the most justified next step from your report. 

---

## 2. Main hypothesis for Phase 3

### Working hypothesis

The current GPTQ results may be constrained by coarse decision granularity:

* bits allocation is still matrix-level,
* rank allocation saturates early,
* some matrices may need finer selective upgrades than “entire matrix bit bump” or “entire matrix low-rank patch.”

### What Phase 3 is trying to answer

1. Does richer GPTQ action granularity improve the frontier?
2. Does the `1.7B` GPTQ rank advantage still hold after richer actions?
3. Does the `8B` GPTQ bits advantage still hold after richer actions?
4. Is the `3B` null result a real dead zone, or just an artifact of coarse actions?

---

## 3. Phase 3 deliverables

By the end of Phase 3, you should have these concrete outputs:

### Deliverable A — richer GPTQ action-space implementation

At least one, ideally two, richer action mechanisms:

* finer bits allocation than whole-matrix upgrades
* finer rank allocation than current matrix-level repair chunks

### Deliverable B — validated reruns on strongest scales

Rerun:

* `Qwen3-1.7B-Base`
* `Qwen3-8B-Base`

These are the most informative scale points because they currently disagree.

### Deliverable C — decision-frontier report

A clean report answering:

* did richer action space change the winner?
* where does the frontier stabilize?
* what is robust vs unstable?

### Deliverable D — optional hybrid second-stage result

Take best bits-only GPTQ point, then ask:

* should next budget slice go to more bits or rank?

That is your strongest follow-up once the richer action space exists.

---

## 4. Scope for Phase 3

## In scope

* GPTQ only
* richer action space
* reruns on `1.7B` and `8B`
* optional revisit of `3B` only if new action space shows promise
* strict equal-budget comparisons
* finite-logit / finite-loss validation on every accepted result

## Out of scope

* expanding RTN further right now
* large budget sweeps with old matrix-level GPTQ action space
* too many new model families before stabilizing GPTQ methodology
* chasing many small deltas without repeated validation

---

## 5. Workstreams

## Workstream 1 — action-space design

This is the most important workstream.

### 1A. Richer bits action space

Current limitation:

* entire matrix must be upgraded

Possible richer options:

* blockwise bit upgrades within a matrix
* row-group or column-group targeted upgrades
* top-error block promotion
* outlier-aware selective higher-bit storage

Goal:

* allow budget to flow into only the most damaging regions of a matrix

### 1B. Richer rank action space

Current limitation:

* rank repair saturates quickly
* matrix-level repair may be too coarse

Possible richer options:

* variable chunk sizes per matrix
* blockwise or submatrix low-rank patches
* rank tied to measured residual structure, not fixed chunk schedule
* rank allocation with diminishing-return penalty

Goal:

* make rank competitive where damage is structured but not globally low-rank

### 1C. Unified action interface

Build a common abstraction:

* candidate action
* byte cost
* predicted benefit score
* realized benefit after execution

Every action should be represented the same way so the allocator can compare:

* bit upgrade action
* rank patch action
* later hybrid action

This will make your frontier logic much cleaner.

---

## Workstream 2 — scoring and allocation

Your allocator now becomes central.

### 2A. Candidate scoring

For every candidate action, compute:

* byte cost
* local error reduction proxy
* normalized value-per-byte
* action type (`bits` or `rank`)
* target matrix / block identity

### 2B. Allocation rule

Use one allocator that can:

* sort actions by score-per-byte
* enforce budget cap
* avoid pathological overspending
* optionally enforce diversity so one huge matrix does not consume all budget

### 2C. Saturation tracking

For rank actions especially, log:

* marginal gain of each extra chunk
* where returns flatten
* whether saturation is matrix-specific or global

This is important because your current report already flags early saturation as a key issue. 

---

## Workstream 3 — experiment infrastructure hardening

Before major reruns, lock the execution path.

### 3A. Result validity checks

Every run should fail fast unless:

* logits are finite
* loss is finite
* memory accounting is valid
* target modifications actually applied
* baseline and modified model configs are recorded

### 3B. Reproducibility metadata

Store with each run:

* model name
* quantizer config
* calibration config
* action-space version
* allocator version
* target list
* realized bytes
* perplexity
* machine / GPU / device map
* seed if relevant

### 3C. Artifact structure cleanup

You already hit file-vs-directory collisions in results paths.
Fix this now permanently with a deterministic run layout, for example:

`results/{quantizer}/{model}/{action_space_version}/{run_id}/`

Inside:

* `config.json`
* `targets.json`
* `metrics.json`
* `allocator_trace.json`
* `stdout.log`
* `environment.json`

---

## Workstream 4 — core rerun matrix

Do not explode the matrix. Keep it targeted.

## Priority tier 1

### Model A — `Qwen/Qwen3-1.7B-Base`

Why:

* GPTQ currently favors rank here
* RTN favored bits here
* this is the strongest disagreement point

Run:

* GPTQ 4-bit baseline
* richer bits `+1.0%`
* richer rank `+1.0%`
* richer bits `+2.0%`
* richer rank `+2.0%`

### Model B — `Qwen/Qwen3-8B-Base`

Why:

* GPTQ currently favors bits here
* large-scale behavior may expose action-space limits most clearly

Run:

* GPTQ 4-bit baseline
* richer bits `+1.0%`
* richer rank `+1.0%`
* richer bits `+2.0%`
* richer rank `+2.0%`

## Priority tier 2

### Model C — `SmolLM3-3B`

Only run if:

* richer action space materially improves one of the tier-1 models
* or you specifically need an intermediate-scale check

This avoids wasting compute on a scale point that was previously inconclusive.

---

## 6. Execution order

This is the sequence I would use.

## Step 1 — freeze the current baseline branch

Before changing anything:

* tag the current working GPTQ branch
* preserve all currently trusted results
* freeze baseline scripts

Reason:
you do not want Phase 3 implementation work to blur which results came from old vs new action spaces.

## Step 2 — implement unified action abstraction

Add one internal representation for all allocation choices.

Output should support:

* cost
* score
* target
* action type
* metadata

## Step 3 — implement richer bits action space

Do this first because:

* bits already wins in several regimes
* finer bits may reveal whether current bits wins are even stronger than they look

## Step 4 — run `1.7B` smoke tests

Only quick tests:

* verify model loads
* verify target actions apply
* verify no NaNs
* verify memory accounting

## Step 5 — full `1.7B` matched frontier

This is your first real decision gate.

## Step 6 — implement richer rank refinements if needed

If richer bits works but richer rank still saturates too early, improve rank before moving to `8B`.

## Step 7 — run `8B` smoke tests on proven hardware path

Use the already validated single-device A100-80GB path.

## Step 8 — full `8B` matched frontier

This is your second major decision gate.

## Step 9 — compare frontier stability

Now answer:

* did winners change?
* did margins widen or shrink?
* did null regions disappear?

## Step 10 — optional hybrid stage

Start from best bits point, then allocate next slice between:

* more bits
* rank repair

This is only worth doing after action-space improvements are validated.

---

## 7. Decision gates

You should have explicit stop/go criteria.

## Gate 1 — after richer bits implementation

Proceed only if:

* memory accounting is trustworthy
* targeted actions apply correctly
* smoke perplexity is finite and consistent

If not, stop and fix infra before spending compute.

## Gate 2 — after `1.7B` rerun

Questions:

* does rank still beat bits under GPTQ?
* does richer bits close the gap?
* does richer rank expand its lead?

Interpretation:

* if rank still wins strongly, `1.7B` becomes a robust GPTQ-rank regime
* if bits catches up or wins, the earlier result was likely action-space-limited

## Gate 3 — after `8B` rerun

Questions:

* does bits remain better?
* does richer rank become competitive?
* does `8B` remain a strong large-scale bits regime?

## Gate 4 — before hybrid stage

Only proceed if:

* at least one richer action space clearly beats old matrix-level GPTQ frontier
* and results are stable enough to justify second-stage budget allocation

---

## 8. Success criteria

Phase 3 is successful if you can say at least one of the following with confidence:

### Success outcome A

Richer GPTQ action spaces preserve the same qualitative winners:

* then current results are robust

### Success outcome B

Richer GPTQ action spaces change one or more qualitative winners:

* then current results were action-space-limited
* that becomes a publishable insight in itself

### Success outcome C

The frontier becomes more interpretable:

* fewer flat/saturated rank results
* fewer meaningless bits steps
* better value-per-byte structure

### Success outcome D

You can define real regimes such as:

* small-scale GPTQ rank-favoring regime
* large-scale GPTQ bits-favoring regime
* intermediate dead zone / mixed regime

---

## 9. Risks and mitigations

## Risk 1 — action-space explosion

Too many candidate actions may make search expensive.

Mitigation:

* restrict candidate pool to top-K error regions
* prune low-value actions early
* keep scoring cheap

## Risk 2 — false improvements from infra issues

A model modification may silently fail or partially apply.

Mitigation:

* explicit post-apply checks
* log target module diffs
* verify changed parameter counts

## Risk 3 — `8B` instability / resource waste

Large GPTQ reruns are expensive.

Mitigation:

* only run `8B` after `1.7B` proves the new action space is operational
* keep `8B` to the minimal matched pair first

## Risk 4 — overinterpreting tiny perplexity differences

Some deltas are very small.

Mitigation:

* define a minimum meaningful delta threshold
* rerun borderline cases
* prefer stable ranking over one-off tiny wins

## Risk 5 — rank keeps saturating

You may improve action space and still see flat rank behavior.

Mitigation:

* treat that as a scientific result, not a failure
* log marginal gain curves explicitly

---

## 10. Recommended minimal experiment matrix

Here is the lean version.

### Stage A — implementation validation

* richer bits smoke on `1.7B`
* richer rank smoke on `1.7B`

### Stage B — first informative frontier

* `1.7B` baseline
* `1.7B` richer bits `+1.0%`
* `1.7B` richer rank `+1.0%`
* optionally `+2.0%` pair only if `+1.0%` is informative

### Stage C — large-scale confirmation

* `8B` baseline
* `8B` richer bits `+1.0%`
* `8B` richer rank `+1.0%`
* optionally `+2.0%` pair only if `+1.0%` is informative

### Stage D — optional intermediate revisit

* `3B` only if Stage B/C shows action-space improvement

---

## 11. Immediate 10-task checklist

1. Freeze current trusted GPTQ results and tag the branch.
2. Clean result path layout and artifact naming.
3. Add unified action abstraction for bits/rank candidates.
4. Implement richer bits candidate generation.
5. Add allocator trace logging with byte-cost and score-per-byte.
6. Add explicit post-modification validation checks.
7. Run `1.7B` smoke tests for richer bits.
8. Run `1.7B` matched `+1.0%` frontier.
9. Decide whether rank implementation needs refinement before `8B`.
10. Run `8B` matched `+1.0%` frontier on the validated single-device path.

---

## 12. My recommendation on project wording

I would now describe the project like this:

> We study the equal-budget allocation frontier between precision and low-rank capacity in compressed LLMs. Early results show that the preferred allocation is not universal; it depends on quantizer, scale, and the granularity of the action space. Phase 3 focuses on testing whether current GPTQ frontiers are robust or are still limited by coarse matrix-level decisions.

That is a much stronger research story than:

* “rank is better”
* or “bits are better”

