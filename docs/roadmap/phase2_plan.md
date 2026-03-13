# Phase 2 Plan

## Title

Phase 2: Build the fixed-budget frontier for bits, rank, and second-stage hybrid repair under RTN.

## Purpose

Phase 1 established two things:

- quantization damage is clearly non-uniform
- once the bits-only comparison became meaningful, extra bits beat uniform low-rank repair

That means Phase 2 should not be framed as "try more repairs." It should be framed as:

> under fixed memory, map where extra bytes should go first, and only then ask whether rank adds anything after those best bit allocations are already taken

## Core Questions

Phase 2 should answer three linked questions:

1. What is the best quality reachable at each extra-memory budget using only precision upgrades?
2. What is the best quality reachable at the same budgets using only targeted low-rank repair?
3. After the best bit allocations are already taken, does an extra budget slice ever go further when spent on rank?

These correspond to three frontiers:

- bit-only frontier
- rank-only frontier
- hybrid second-stage frontier

## Scope

Phase 2 stays deliberately narrow.

Fixed choices:

- model: `Qwen/Qwen3-0.6B-Base`
- quantizer: `RTN`
- baseline anchor: `R2`
- evaluation focus: perplexity, memory, gain per byte
- calibration-driven, frozen-model setup only

Deferred:

- GPTQ on the local machine
- large-model scaling before allocator behavior stabilizes
- expensive gradient-based rank fitting
- broad downstream benchmark expansion

## Why RTN Remains the Local Mainline

RTN is already working on the current machine and is sufficient for:

- allocator design
- memory accounting
- action logging
- frontier construction

The code should still be written so that the same action schema can later be applied to `GPTQ` on another machine.

## Phase 2 Structure

Phase 2 should be executed in two stages.

### Phase 2A

Build the first fair frontier comparison:

- matrix-level targeted bits
- matrix-level targeted rank
- equal budgets
- equal candidate layer pool
- equal calibration and eval setup

### Phase 2B

Only if Phase 2A still leaves important ambiguity:

- add groupwise or blockwise bit actions
- add dynamic score refresh in the allocator
- test hybrid second-stage allocation

This keeps Phase 2 from sprawling.

## Experimental Accounting

Before any new runs, standardize the accounting.

Lock the following:

- same base model
- same RTN-4 baseline
- same calibration subset
- same eval subset
- same memory accounting rules
- same seed policy
- same output schema

For every candidate action, log:

- `action_type`: `bit_upgrade` or `rank_repair`
- `target_granularity`: `matrix`, `group`, or later `block`
- `target_name`
- `byte_cost`
- `proxy_score`
- `predicted_gain_per_byte`
- `realized_perplexity`
- `realized_gain`
- `cumulative_budget_bytes`

Deliverable:

- a canonical action schema file, such as `docs/reference/phase2_action_schema.md` or a JSON schema in `configs/`

## Candidate Pool

Do not let the action space explode immediately.

Start with a restricted candidate pool taken from Phase 1 sensitivity:

- top `8-12` matrices by activation-space damage from `R2`
- mostly later `mlp.down_proj`
- include several `self_attn.o_proj`
- include a few low-sensitivity control matrices

This pool must be shared by both the targeted-bit and targeted-rank baselines so the comparison stays fair.

## Phase 2A: Bit-Only Frontier

### Goal

Build the best bits-only model reachable at each extra-memory budget on top of `R2`.

### First Action Space

Implement bit actions in this order:

1. matrix-level upgrades
2. groupwise or blockwise upgrades
3. intermediate-bit upgrades if the implementation supports them cleanly

Start with matrix-level upgrades only.

Recommended first matrices:

- `mlp.down_proj`
- `mlp.up_proj`
- `mlp.gate_proj`
- `self_attn.o_proj`

Add `q_proj`, `k_proj`, and `v_proj` only if needed later.

### Budget Schedule

Use budgets relative to `R2`:

- `+0.05%`
- `+0.1%`
- `+0.25%`
- `+0.5%`
- `+1.0%`
- `+2.0%`

These can be tuned slightly, but the point is to get enough points to show curvature rather than just isolated wins.

### Bit-Only Baselines

At each budget, compare:

- uniform bits baseline
- simple sensitivity-sorted baseline
- greedy allocator
- optional random control for sanity

### Output

Produce:

- perplexity vs added bytes
- quality recovered vs added bytes
- selected action sequence by budget
- payoff by matrix type

Success condition:

- the bits-only frontier is smoother and more informative than the coarse whole-layer story from Phase 1

## Phase 2A: Targeted-Rank Frontier

### Goal

Build the fair rank-only comparison that Phase 1 did not have.

This is the critical missing baseline. Phase 1 compared targeted bits against uniform rank. Phase 2 must compare targeted bits against targeted rank.

### Rank Action Space

Mirror the bit-action fairness as closely as possible.

Start with rank actions like:

- rank `4` on one matrix
- rank `8` on one matrix
- rank `16` on one matrix
- rank `32` on one matrix

Apply these only to the restricted candidate pool.

Do not try rank everywhere at first.

### Rank Baselines

At each budget, compare:

- uniform-rank baseline
- top-k sensitivity-ranked allocation
- greedy gain-per-byte rank allocation

### Output

Produce:

- rank-only frontier at the same budget points as the bit frontier
- matched targeted-bit vs targeted-rank comparisons
- payoff by matrix type

Useful outcomes:

- targeted bits still clearly win
- targeted rank closes the gap but still loses
- targeted rank only wins in certain layer types or budget ranges

All three are useful results if the comparison is fair.

## Allocation Signals

Use three scoring families:

1. weight-space residual norm
2. activation-space deviation
3. direct measured marginal improvement on a small calibration evaluation

Implementation order:

1. `greedy_weight`
2. `greedy_activation`
3. `greedy_measured`

Reason:

- weight-space is cheap and gives a fast baseline
- activation-space is more likely to matter
- directly measured marginal probing is strongest but can become expensive

Do not start with the most expensive signal first.

## Allocator Versions

Phase 2 does not need a complex optimizer immediately.

### Version 1

Static greedy:

- precompute scores
- divide by byte cost
- choose best remaining action
- continue until budget is exhausted

### Version 2

Dynamic greedy:

- refresh scores after each selected action
- repeat until budget is exhausted

Recommendation:

- get static greedy working first
- only add dynamic refresh if the results suggest that interaction effects matter

## Phase 2B: Hybrid Second-Stage Frontier

Only do this after both the bit-only and targeted-rank frontiers are stable.

### Main Question

After the best bit allocations are already taken, where should the next bytes go?

### Experiment Design

For each selected base budget `B`:

1. start from `R2`
2. build the best bits-only model under budget `B`
3. add a small extra budget `delta`
4. compare:
   - spend `delta` on more bits
   - spend `delta` on targeted rank
   - split `delta` between bits and rank

This is the cleanest test of the revised thesis.

## Analysis Tracks

Two analyses should be treated as first-class deliverables.

### 1. Layer-Type Payoff Analysis

Aggregate selected actions and realized gains by:

- `mlp.down_proj`
- `mlp.up_proj`
- `mlp.gate_proj`
- `self_attn.o_proj`
- `q_proj`
- `k_proj`
- `v_proj`

Questions:

- does the later `mlp.down_proj` story survive finer allocation?
- do certain layer families favor bits while others favor rank?

### 2. Proxy-Quality Correlation

For each candidate action, compare:

- weight-space error
- activation-space deviation
- realized perplexity gain

Useful summaries:

- rank correlation
- top-k hit rate
- average gain of top-ranked selected actions

This can become a practical result on its own.

## Run Matrix

The exact run IDs can be decided once the implementation exists, but the matrix should be structured like this.

### Phase 2A Runs

1. regenerate the fixed `R2` anchor if needed
2. build matrix-level candidate bit actions
3. run bit-only frontier across all budget points
4. build targeted-rank candidate actions on the same matrix pool
5. run rank-only frontier across the same budget points
6. compare targeted bits vs targeted rank

### Phase 2B Runs

1. choose `2-3` representative base budgets from the bit frontier
2. add second-stage hybrid mini-budget runs
3. compare bits-only extension vs rank extension vs mixed extension

## Weekly Execution Order

### Week 1

- freeze evaluation/accounting
- define the Phase 2 action schema
- implement matrix-level bit upgrades
- generate the first candidate action inventory

### Week 2

- compute activation-space profiling on the candidate pool
- run the first bit-only greedy allocator
- produce the first bit-only frontier

### Week 3

- if needed, add groupwise or blockwise bit actions
- rerun the bit-only frontier
- identify stable high-value layer types

### Week 4

- implement targeted SVD rank actions
- build the rank-only frontier

### Week 5

- compare targeted bits vs targeted rank
- write an interim conclusion memo

### Week 6

- run hybrid second-stage experiments
- determine whether rank ever becomes the better next-byte use after the best bit actions are exhausted

### Week 7 and Beyond

- port the stabilized allocator to `Qwen/Qwen3-1.7B-Base`
- later run the same logic with `GPTQ` on a stronger machine

## Stop / Go Criteria

### Stop Increasing Bit Granularity When

- the bit-only frontier is smooth across several budgets
- the top allocator decisions stop changing materially with extra granularity

### Stop Increasing Rank Complexity When

- targeted rank clearly fails to close the gap to targeted bits across budgets
- or targeted rank only becomes useful after the strongest bit actions are already exhausted

### Escalate to Larger Models When

- the top selected layer types are stable across repeat runs
- allocator decisions are no longer dominated by harness noise

## Success Criteria

Phase 2 is successful if it produces:

1. a smooth bit-only frontier over several budget points
2. a fair targeted-rank frontier over the same points
3. a direct targeted bits vs targeted rank comparison
4. at least one clear rule about where the next byte should go
5. a clean answer on whether low-rank repair is mainly a first-line tool or a second-stage tool

## Deliverables

- Phase 2 action schema
- candidate bit-action inventory
- candidate rank-action inventory
- bit-only frontier plots
- targeted-rank frontier plots
- hybrid second-stage comparison plots
- layer-type payoff tables
- proxy-quality correlation analysis
- an interim memo after the targeted bits vs targeted rank comparison
- an updated proposal or roadmap after Phase 2 conclusions stabilize

## Initial Repo Artifacts

The repo scaffolding for this phase should start with:

- `docs/reference/phase2_action_schema.md`
- `configs/phase2/phase2_action_schema.json`
- `configs/phase2/qwen3_0p6b_r2_top12_candidate_pool.json`
- `configs/phase2/phase2_first_pass_manifest.json`
- `scripts/run_phase2.py`
- `results/phase2/README.md`

The first pass should use a small exploratory matrix before expanding to the full frontier:

- `P2B01`: bits-only, matrix-level, uniform, `+0.25%`
- `P2B02`: bits-only, matrix-level, greedy activation, `+1.0%`
- `P2R01`: targeted-rank, matrix-level, uniform, `+0.25%`
- `P2R02`: targeted-rank, matrix-level, greedy activation, `+1.0%`

## Working Headline

If Phase 2 needs a one-line framing for notes or a future paper section:

> Build the fixed-budget frontier for bits, rank, and second-stage hybrid repair under RTN.
