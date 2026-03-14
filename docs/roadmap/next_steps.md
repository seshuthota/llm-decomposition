# Next Steps

This file is the current restart point for the project.

## Stable Results

- Local `RTN` work is complete.
- `Qwen/Qwen3-0.6B-Base` under local `RTN` favored targeted rank over the current matrix-level targeted bits frontier.
- `Qwen/Qwen3-1.7B-Base` under Modal `RTN` favored targeted bits over targeted rank at both tested budgets.
- `HuggingFaceTB/SmolLM3-3B-Base` under Modal `RTN` favored targeted bits at the first matched budget point.
- `Qwen/Qwen3-8B-Base` under Modal `RTN` also favored targeted bits at the first matched budget point.

These results are already documented and should be treated as trustworthy.

## Current Status

The original GPTQ bring-up blocker on `Qwen/Qwen3-1.7B-Base` has been cleared on Modal.

Current status:

- GPTQ baseline `R3_Q17B` is now valid
- GPTQ transfer points `G2B02_Q17B`, `G2R02_Q17B`, `G2B03_Q17B`, and `G2R03_Q17B` are now valid
- the first matched GPTQ point favors targeted rank over targeted bits
- the current GPTQ rank action space saturates by `+1.0%`, so `G2R03_Q17B` does not move beyond `G2R02_Q17B`
- `HuggingFaceTB/SmolLM3-3B-Base` GPTQ validation is now also available at the first matched point:
  - baseline `R3_S3B`: `11.5366`
  - bits `G3B02_S3B`: `11.5483`
  - rank `G3R02_S3B`: `11.6482`
  - conclusion: neither method improved over the `3B` GPTQ baseline, and bits regressed less than rank
- `Qwen/Qwen3-8B-Base` GPTQ validation is now also available at the first matched point:
  - baseline `R3_Q8B`: `11.7970`
  - bits `G2B02_Q8B`: `11.7823`
  - rank `G2R02_Q8B`: `11.7962`
  - conclusion: targeted bits beat targeted rank at `8B`
- the first richer GPTQ bits pilot is now also available on `1.7B`:
  - row-block bits `G2B02RB_Q17B`: `15.9060`
  - conclusion: valid and cheaper to run on `A10G`, but worse than the existing matrix-level bits point `G2B02_Q17B` (`15.8993`)
- the refined richer GPTQ bits pilot is now also available:
  - `1.7B` `128`-row blocks `G2B02RB128_Q17B`: `15.8970`
    - beats matrix-level bits `G2B02_Q17B` (`15.8993`)
  - `8B` `128`-row blocks `G2B02RB128_Q8B`: `11.7954`
    - improves over baseline but does not beat matrix-level bits `G2B02_Q8B` (`11.7823`)
  - conclusion: richer bits are now clearly action-space dependent across scale
- a finer GPTQ rank ladder is now also available on `1.7B`:
  - `G2R02F_Q17B`: `15.9073`
  - repair bytes: `7,995,392`
  - conclusion: finer rank spent more budget but was worse than the original matrix-level rank point `G2R02_Q17B` (`15.8823`)
- a GPTQ hybrid second-stage pilot is now also available on `1.7B`:
  - richer-bits `+2.0%` follow-up `G2B03RB128_Q17B`: `15.9272`
  - hybrid second-stage `H2R02RB128_Q17B`: `15.8989`
  - conclusion:
    - after the best richer-bits point `G2B02RB128_Q17B`, giving the next slice to rank was much better than giving it to more richer bits
    - but hybrid still did not beat the earlier pure-rank `+1.0%` GPTQ point `G2R02_Q17B` (`15.8823`)
- a structural GPTQ rank pilot is now also available on `1.7B`:
  - row-block rank `G2R02RB128_Q17B`: `15.9034`
  - conclusion:
    - the run was valid and cheap enough to test on `A10G`
    - but it underperformed both matrix-level rank and the stronger richer-bits point
- column-aligned GPTQ structural pilots are now also available on `1.7B`:
  - column-block bits `G2B02CB128_Q17B`: `15.9171`
  - column-block rank `G2R02CB128_Q17B`: `15.9004`
  - conclusion:
    - column-block bits were worse than baseline and all stronger bits points
    - column-block rank was slightly better than row-block rank, but still weaker than matrix-level rank
    - together, these results argue against spending more on small blockwise local variants under the current scoring rule
- a grouped family-aware GPTQ rank allocator is now also available on `1.7B`:
  - grouped rank `G2R02GRP_Q17B`: `15.9224`
  - conclusion:
    - it successfully diversified the early budget across the main layer families
    - but it performed much worse than the original matrix-level rank point
    - so simple family balancing is not the missing ingredient in the current rank design
- a GPTQ hybrid second-stage validation is now also available on `8B`:
  - hybrid `H2R02_Q8B`: `11.7895`
  - conclusion:
    - hybrid improved over baseline and rank-only on `8B`
    - but it still did not beat bits-only `G2B02_Q8B` (`11.7823`)
- the `1.7B` matrix-policy comparison is now complete:
  - matrix hybrid `H2R02M_Q17B`: `15.8962`
  - conclusion:
    - hybrid improved over the first bits-only point
    - but it still did not beat matrix bits `G2B03_Q17B` (`15.8914`)
    - and it still did not beat matrix rank `G2R02_Q17B` (`15.8823`)
- the `8B` equal-budget policy comparison is now complete:
  - bits `+2.0%` `G2B03_Q8B`: `11.8024`
  - rank `+2.0%` `G2R03_Q8B`: `11.7962`
  - conclusion:
    - extra bits beyond `G2B02_Q8B` hurt
    - extra rank beyond `G2R02_Q8B` did nothing
    - hybrid remains between bits-only and rank-only at `8B`
- the bounded multi-bit bits-policy branch has now been tested on `1.7B`:
  - `MB1_Q17B`: `15.9097`
  - conclusion:
    - the allocator mostly chose cheap `4->5` upgrades
    - the result did not beat the current best bits point `G2B02_Q17B` (`15.8993`)
    - it also remained clearly below the rank winner `G2R02_Q17B` (`15.8823`)
    - this triggers the stop rule for the final bounded branch
    - `MB2_Q17B` and any `8B` multi-bit follow-up are not justified

Canonical references:

- `docs/experiments/qwen3_1p7b_gptq_bringup_status.md`
- `docs/roadmap/qwen3_1p7b_gptq_plan.md`
- `docs/roadmap/gptq_recovery_plan.md`
- `docs/roadmap/gptq_richer_action_space_plan.md`
- `docs/experiments/experiment_journal.md`

## Immediate Resume Plan

When work resumes, do this in order:

1. Treat the `RTN` scale map as complete and stable.
2. Treat the `GPTQ 1.7B` matrix-level transfer study as complete for the current action space.
3. Treat the first `GPTQ 3B` matrix-level transfer point as complete:
   - the `1.7B` GPTQ rank win did not transfer cleanly
   - the current `3B` GPTQ transfer point favors bits only in the weak sense that bits regressed less than rank
4. Decide the next research branch:
   - treat the first GPTQ hybrid second-stage pilot as complete on `1.7B`
   - treat the first `8B` hybrid validation as complete
   - treat the first structural row-block rank pilot on `1.7B` as negative
   - treat shared-family rank as explicitly **out of scope for now**
   - treat the final policy-comparison branch as complete
   - treat the bounded multi-bit bits-policy branch as complete and stopped at `1.7B`
   - freeze the trusted GPTQ state in:
     - `docs/experiments/gptq_pre_final_branch_snapshot.md`
   - the remaining GPTQ work is now synthesis / write-up, not another experiment branch

Current `8B` GPTQ status:

- smoke baseline `R3S_Q8B` is valid on Modal
- full baseline `R3_Q8B` is valid on Modal
- first matched pair `G2B02_Q8B` and `G2R02_Q8B` is complete
- the working execution path is:
  - `A100-80GB`
  - `device_map: "single"`
- so `8B` GPTQ is no longer blocked on backend bring-up

## Decision Rule For The Next GPTQ Branch

The active GPTQ endgame plan is now:

- `docs/roadmap/gptq_closure_plan.md`

Default rule:

- do not reopen rank redesign
- do not add more local-layout variants
- do not add more budget sweeps first

If the goal is a stronger GPTQ paper result without reopening method design:

- the multi-bit bits-policy branch has already been run on `1.7B`
- it did not materially improve over the existing bits frontier
- so the correct next move is final synthesis, not more GPTQ runs

If the goal is faster project closure:

- document the current GPTQ result as:
  - `1.7B`: rank wins
  - `3B`: neither helps; bits regress less than rank
  - `8B`: bits win
- then move to the next unfinished branch

## Practical Summary

The project is no longer blocked on GPTQ bring-up. The new decision is about which experiment branch is worth implementing next.

The next session should start from:

- complete RTN regime map
- complete GPTQ `1.7B` transfer pair plus the saturated `+2.0%` follow-up
- complete GPTQ `3B` first matched pair
- complete GPTQ `8B` smoke, baseline, and first matched pair
- complete first richer-bits GPTQ pilot on `1.7B`
- complete first richer-bits GPTQ validation on `8B`
- complete first finer-rank GPTQ pilot on `1.7B`
- complete first GPTQ hybrid second-stage pilot on `1.7B`
- complete first structural row-block rank GPTQ pilot on `1.7B`
- complete first GPTQ hybrid second-stage validation on `8B`
- complete first column-block bits GPTQ pilot on `1.7B`
- complete first column-block rank GPTQ pilot on `1.7B`
- complete first grouped family-aware rank allocator validation on `1.7B`
- complete `1.7B` matrix-policy hybrid comparison
- complete `8B` equal-budget policy comparison

and then choose between:

- project closure / synthesis
