# GPTQ Endgame Plan

This file is the bounded endgame plan for the GPTQ branch.

It defines two valid exits:

1. stop experimentation and close GPTQ
2. run exactly one final bounded experiment branch: multi-bit bits-policy

The objective now is not to reopen method design. It is to either close GPTQ cleanly or run one last tightly scoped stress test on the bits side.

## Objective

The GPTQ branch should now optimize for:

- interpretability
- bounded scope
- low risk of branch creep
- a defendable final write-up

Current GPTQ evidence is already strong enough to support a regime-dependent conclusion. Any remaining work must refine that conclusion, not muddy it.

## Frozen Scientific Position

Unless the final bounded branch changes the story, the official GPTQ interpretation is:

- `1.7B`: rank-only best
- `3B`: neither helps; bits regress less than rank
- `8B`: bits-only best

Policy-comparison conclusion:

- `1.7B`: rank-only > bits-only > hybrid
- `8B`: bits-only > hybrid > rank-only

Meaning:

- GPTQ is regime-dependent across scale
- hybrid is useful but not universally dominant
- action-space design matters
- current evidence does not justify opening a new rank-method family

## Branch Boundary

Shared-family rank is explicitly out of scope for now.

Reason:

- it would open a materially different algorithm family
- it would convert this bounded closure branch into a new design project
- the current evidence does not justify that expansion

Also out of scope:

- new structural rank branches
- reopening grouped-rank / block-rank pilots
- large budget sweeps
- adding new scales before closing the current GPTQ story

## Path A: Close Now

Choose this path if the goal is:

- efficient completion
- strong enough evidence for reporting
- avoiding more compute for marginal gain

Tasks:

1. freeze the current trusted GPTQ frontier tables
2. freeze the current policy-comparison ordering
3. write the final GPTQ synthesis
4. move to final reporting / next project branch

This is the lowest-cost and highest-confidence exit.

## Path B: One Final Bounded Branch

Choose this path only if we want one more controlled check on the bits side.

Recommended branch:

- multi-bit bits-policy

Core question:

> At `1.7B` GPTQ, does allowing a richer set of bit upgrade levels materially improve the bits frontier enough to challenge the current rank-only winner?

If no, GPTQ closure becomes much stronger.

If yes, the bits-side story needs to be updated carefully.

## Freeze First

Before any further experiment, freeze the current trusted state.

Tasks:

1. freeze trusted frontier tables for:
   - `1.7B`
   - `3B`
   - `8B`
2. freeze trusted policy ordering:
   - `1.7B`: rank > bits > hybrid
   - `8B`: bits > hybrid > rank
3. snapshot:
   - run ids
   - perplexities
   - memory totals
   - hardware path
   - current code/config state

Canonical snapshot file:

- `docs/experiments/gptq_pre_final_branch_snapshot.md`

## Final Bounded Branch: Multi-Bit Bits Policy

### Scope

In scope:

- GPTQ only
- bits-only policy only
- modest expansion of upgrade levels
- `1.7B` first
- `8B` only if `1.7B` changes materially

Out of scope:

- rank redesign
- new hybrid search
- new structural rank actions
- more local bits-layout variants

### Exact Change

Change only one thing:

- the set of allowed bits actions

Keep fixed:

- candidate pool construction
- allocator framework
- equal-budget comparison style
- evaluation protocol

Recommended allowed target levels:

- `5-bit`
- `6-bit`
- `8-bit`

Implementation preference:

- direct target-bit actions with explicit byte costs

### Hardening Requirements

Before full runs:

1. validate byte accounting for each new bit level
2. validate that all candidate levels are actually considered
3. validate that selected modules are upgraded correctly
4. log a full action trace:
   - target matrix
   - chosen bit level
   - byte cost
   - cumulative spend
   - selection order

## Execution Plan

### Phase 1: `1.7B` first

Use existing trusted references:

- baseline: `R3_Q17B`
- best current bits: `G2B03_Q17B`
- best current rank: `G2R02_Q17B`

Recommended new run set:

1. `MB1_Q17B`
   - multi-bit bits policy
   - budget matched to current `+1.0%`
2. `MB2_Q17B`
   - multi-bit bits policy
   - budget matched to current `+2.0%`

Optional only if needed:

3. `MBH_Q17B`
   - hybrid from the best multi-bit point
   - only if the new bits point becomes genuinely competitive

### Phase 2: Interpretation Gate

After `1.7B`, stop and decide.

Outcome A:

- multi-bit bits still loses clearly to rank
- stop immediately
- conclude the `1.7B` rank preference is robust

Outcome B:

- multi-bit bits roughly matches rank
- consider one confirming step only if necessary
- `8B` becomes conditionally justified

Outcome C:

- multi-bit bits beats rank
- this materially changes the story
- then `8B` validation becomes justified

### Phase 3: Conditional `8B`

Only run `8B` if the `1.7B` result changes materially.

Recommended run set:

1. `MB1_Q8B`
   - multi-bit bits policy
   - budget matched to current `+1.0%`
2. optional `MB2_Q8B`
   - only if `MB1_Q8B` changes the frontier enough to justify it

## Stop Rules

Stop immediately if any of these are true:

1. `1.7B` multi-bit bits does not materially improve over current best bits
2. `1.7B` improves slightly but remains clearly below rank
3. `8B` multi-bit bits does not beat existing best bits
4. any follow-up would require:
   - a new method family
   - rank redesign
   - larger branch expansion

If any stop rule is hit, close GPTQ and move to synthesis.

### Current Branch Status

The optional branch has now been tested at the first gate:

- `MB1_Q17B`: completed
- result: did not beat the current best bits point and remained clearly below rank

So the current status is:

- stop rule hit at `1.7B`
- do not run `MB2_Q17B`
- do not continue to `8B`
- GPTQ should now move to synthesis / reporting

## Reporting Package

Whether or not the final branch runs, prepare:

1. final GPTQ frontier table
2. policy-comparison table
3. cross-scale summary table
4. one-page closure note
5. final decision memo

Suggested summary table columns:

- model
- baseline
- best bits
- best rank
- best hybrid
- winner
- interpretation

## Recommended Decision Policy

The recommended execution rule is:

> Run the multi-bit bits-policy branch on `1.7B` only. Continue to `8B` only if the `1.7B` result materially changes the current ordering. Otherwise close GPTQ and move to final synthesis.

That is the best balance of:

- scientific value
- bounded scope
- cost control
- report clarity
