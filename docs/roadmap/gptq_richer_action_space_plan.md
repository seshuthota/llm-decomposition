# GPTQ Richer Action-Space Plan

This file is the canonical next-step roadmap derived from [archive/feedback/feedback.md](../archive/feedback/feedback.md).

## Objective

The next branch is not more budget sweeps with the current matrix-level action space.

The objective is:

- determine whether current GPTQ frontier behavior is a real quantizer/model effect
- or mostly a limitation of the current coarse matrix-level action space

## Why This Is The Right Next Branch

The current trusted GPTQ results are:

- `1.7B`: targeted rank beats targeted bits
- `3B`: neither method improves over the baseline; bits regress less than rank
- `8B`: targeted bits beat targeted rank

This is already enough to show regime dependence, but not enough to claim the current frontier is stable. Two current limitations are visible:

- bits actions are still whole-matrix upgrades
- rank actions saturate early, especially outside the `1.7B` case

So the next job is to improve action expressiveness before adding many more points.

## Deliverables

Phase 3 continuation should produce:

1. a unified action representation for bits and rank
2. at least one richer GPTQ bits action space
3. reruns on the strongest GPTQ scales:
   - `Qwen3-1.7B-Base`
   - `Qwen3-8B-Base`
4. a decision report stating whether the current GPTQ conclusions survive the richer action space
5. optionally, one hybrid second-stage check after the richer action space is working

## Workstreams

## Workstream 1: Unified Action Interface

All candidate actions should share one representation with:

- action identity
- action type
- target granularity
- byte cost
- proxy score
- predicted gain per byte
- execution-time selection metadata

This allows the allocator and logging path to compare:

- bit upgrades
- rank repairs
- later hybrid actions

## Workstream 2: Richer Bits Actions

This is the first implementation priority.

The main limitation today is that a bit action upgrades an entire matrix. Richer candidates should allow budget to flow only to the most damaging regions.

Likely options:

- blockwise bit upgrades
- row-group or column-group upgrades
- outlier-aware higher-bit storage

Start with one design, not several.

The first implemented design is:

- row-block bit upgrades
- fixed row block size
- same allocator and candidate pool as the current matrix-level GPTQ runs

This keeps the first richer-action comparison controlled.

## Workstream 3: Richer Rank Actions

Current incremental rank chunks are better than the original one-shot rank path, but still saturate too early on several models.

Likely options:

- variable chunk sizes
- stronger diminishing-return tracking
- submatrix or blockwise repair actions

Do not start here. Start with richer bits first and only deepen rank if the richer bits reruns still leave the frontier unclear.

## Workstream 4: Validation Reruns

The first reruns after action-space work should be:

1. `Qwen3-1.7B-Base` under GPTQ
2. `Qwen3-8B-Base` under GPTQ

These are the most informative scales because the current matrix-level results disagree.

Do not prioritize `3B` immediately. Revisit it only if the richer action space materially changes one of the stronger scale points.

## Workstream 5: Optional Hybrid Second Stage

Only after the richer action space is implemented and rerun:

- start from the best bits-only GPTQ point
- ask whether the next budget slice should go to more bits or targeted rank

This is higher value than adding many more points with the old coarse action space.

## Execution Order

Implement and run in this order:

1. unify action representation
2. add one richer GPTQ bits action space
3. rerun `1.7B` GPTQ
4. rerun `8B` GPTQ
5. compare against the current matrix-level GPTQ results
6. decide whether richer rank or hybrid second-stage is necessary

## Stop Rule

Stop the richer-action branch once one of these becomes clear:

- the current GPTQ winners survive richer action space
- the winner flips under richer action space
- or the result becomes clearly budget/action-space dependent

At that point, write the decision-frontier conclusion instead of continuing to add marginal variants.

## Current Branch Boundary

The richer-action branch is now bounded.

Specifically:

- shared-family rank repair is **not** the next step
- that path is being treated as a separate future algorithm branch, not part of the current bounded GPTQ frontier study

The active handoff from this roadmap is now:

- `docs/roadmap/gptq_closure_plan.md`

That file defines the remaining allowed GPTQ closure options inside the current project scope.
