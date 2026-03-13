Yes — your Phase 2 outline is directionally right, and I’d make it even sharper:

**Phase 2 should be about building the decision frontier, not just trying more repairs.**

Given your Phase 1 result, the next goal is no longer “make low-rank look better.” It is:

> **under fixed memory, map where extra bytes should go first, and only then ask whether rank adds anything after those best bit allocations are already taken.**

That direction also matches where recent PTQ work has moved: stronger baselines now use fine-grained, budgeted mixed-precision allocation rather than coarse whole-layer jumps, while recent low-rank reconstruction work is increasingly structured and saliency-aware rather than uniform-rank repair. ([arXiv][1])

## My judgment on your proposed Phase 2

Your six points are good. The only thing I would change is this:

* do **not** jump from “bits beat uniform rank” directly to “bits first, then hybrid”
* first do the fairer comparison:

  * **targeted bits vs targeted rank**
  * same candidate layers
  * same memory budget
  * same calibration set
  * same evaluation metric

If bits still win there, your story becomes much stronger.

## What I would make Phase 2 about

I’d define Phase 2 around **three frontiers**:

1. **Bit-only frontier**
   Best quality reachable at each extra-memory budget using only precision upgrades.

2. **Rank-only frontier**
   Best quality reachable at each extra-memory budget using only targeted low-rank residuals.

3. **Hybrid second-stage frontier**
   Starting from the best bit-only point at a given budget, does spending the *next* budget slice on rank ever beat spending it on more bits?

That gives you a very clean paper structure.

---

# Detailed Phase 2 plan

## Phase 2 objective

Build a **budget-aware allocator** on RTN that can compare candidate actions in terms of:

* **memory cost**
* **predicted gain**
* **gain per byte**

Then use it to answer:

* when do bits dominate?
* when does targeted rank catch up?
* does rank become useful only after the highest-value bit upgrades are exhausted?

---

## Phase 2A — Lock down the experimental accounting

Before new experiments, make the accounting airtight.

### What to standardize

Use exactly one fixed setup for all Phase 2 runs:

* same base model
* same RTN baseline
* same calibration subset
* same eval subset
* same memory accounting rules
* same run metadata format
* same seed policy

### What to log for every action

For every candidate action, store:

* action type (`bit_upgrade`, `rank_repair`)
* target (`layer`, `matrix`, `group`, `channel block`)
* byte cost
* perplexity change
* proxy score before allocation
* actual realized gain after applying the action
* cumulative budget used

This sounds boring, but it is what will let you make clean gain-per-byte plots later.

### Deliverable

A single canonical `phase2_action_schema.json` or similar config format.

---

## Phase 2B — Build a finer-grained bit action space

This is the most important part.

Your current whole-layer 4→8 setup is too coarse, and Phase 1 already showed that it can make some budgets meaningless. Recent mixed-precision PTQ work is explicitly moving toward finer-grained allocation under hard budgets, often at block or subspace level rather than blunt per-layer jumps. ([arXiv][1])

### Recommended candidate bit actions

I would define bit actions at **three granularities**:

### 1. Matrix-level upgrades

Instead of upgrading an entire transformer block, upgrade specific matrices:

* `mlp.down_proj`
* `mlp.up_proj`
* `mlp.gate_proj`
* `self_attn.o_proj`
* optionally `q_proj`, `k_proj`, `v_proj`

Since Phase 1 already suggests later `mlp.down_proj` is high-value, start there first.

### 2. Groupwise/blockwise upgrades

Within a matrix, allow upgrades for:

* output-channel groups
* input-channel groups
* contiguous row/column blocks
* quantization groups if your implementation already has that structure

This is probably the best local compromise between fairness and implementation pain.

### 3. Intermediate precision levels

If your stack allows it, support:

* 4 → 5 bit
* 4 → 6 bit
* 4 → 8 bit

Even if 5/6-bit is slower or uglier implementation-wise, it is scientifically valuable because it lets you probe smaller budget increments.

### My recommendation

Do this in order:

1. **matrix-level**
2. **groupwise**
3. **intermediate-bit**
4. only then partial arbitrary channel upgrades if still needed

That keeps the implementation manageable.

### Deliverable

A `candidate_bit_actions.jsonl` file containing all legal upgrade actions and byte costs.

---

## Phase 2C — Add a fair targeted-rank baseline

This is the missing comparison.

Right now the meaningful win is:

* targeted bit upgrade
* versus uniform rank repair

That is enough for a pivot, but not enough for the final conclusion.

### Rank action space should mirror the bit action space

If bits are targeted, rank must also be targeted.

Recommended rank actions:

* rank-4 on one matrix
* rank-8 on one matrix
* rank-16 on one matrix
* rank-32 on one matrix

only on the most promising matrices at first:

* top sensitive `mlp.down_proj`
* top sensitive `self_attn.o_proj`
* maybe a few controls from low-sensitivity layers

### Important constraint

Do **not** try rank everywhere initially.
Use Phase 1 sensitivity results to restrict to a small candidate pool first, maybe top 8–12 matrices.

### Why this matters

This gives you a clean comparison:

* best targeted bit action at budget B
* best targeted rank action at budget B

That is much stronger than comparing targeted bits to uniform rank.

### Deliverable

A `candidate_rank_actions.jsonl` file with byte cost and target matrix metadata.

---

## Phase 2D — Build the allocator

Your allocator does not need to be fancy in Phase 2. It needs to be **defensible**.

### Version 1: greedy gain-per-byte

For each candidate action:

* estimate score
* divide by byte cost
* pick best
* update model
* recompute if needed
* repeat until budget exhausted

This is enough for a solid paper if the instrumentation is clean.

### Allocation signals to compare

Use three signals:

1. **weight-space residual norm**
2. **activation-space deviation**
3. **direct measured marginal improvement** on a small calibration eval

If activation-space or direct marginal testing predicts winning actions much better than weight-space norms, that is a publishable result on its own. Recent work on bit allocation and structured reconstruction is explicitly sensitivity-driven, so making this comparison central is a good move. ([arXiv][1])

### My advice

Treat these as phases:

* first allocator: static greedy using precomputed scores
* second allocator: dynamic greedy with score refresh after each chosen action

You may find static is already enough.

### Deliverable

Three allocators:

* `greedy_weight`
* `greedy_activation`
* `greedy_measured`

---

## Phase 2E — Produce the bit-only frontier first

This should be the first major experiment block.

### Question

At equal extra memory, what is the best bits-only model you can build on RTN?

### Suggested budget points

Use budgets relative to the RTN-4 baseline, for example:

* +0.05%
* +0.1%
* +0.25%
* +0.5%
* +1.0%
* +2.0%

The exact values can be tuned, but you want enough points to show curvature.

### What to compare

At each budget:

* uniform bits baseline
* simple sensitivity sort baseline
* greedy allocator
* maybe random control for sanity

### Output

You want a curve:

* x-axis = added bytes
* y-axis = perplexity recovery or perplexity itself

This becomes your **bit-only Pareto frontier**.

### Success criterion

Phase 2 is already succeeding if this frontier is much stronger and smoother than the coarse whole-layer upgrade story from Phase 1.

---

## Phase 2F — Produce the targeted-rank frontier

Only after bit-only frontier is stable.

### Question

At the same budget points, how far can targeted SVD repair go if it is allocated intelligently rather than uniformly?

### Compare

At each budget:

* uniform rank baseline
* top-k sensitive-layer rank allocation
* greedy gain-per-byte rank allocation

### What you are testing

Not whether rank “works at all,” because Phase 1 already showed it does a bit.
You are now testing whether **targeted rank can compete once it is given the same fairness as targeted bits**.

### Likely outcomes

There are three useful possibilities:

1. targeted bits still clearly win
2. targeted rank closes the gap but still loses
3. targeted rank wins only in certain layer types or budget ranges

All three are publishable if shown cleanly.

---

## Phase 2G — Only then test hybrid second-stage repair

This is where your proposed step 5 is exactly right.

Once you know the best bits-only allocation at each budget, ask:

### Main question

If I already spent the first extra bytes optimally on bits, where should the next bytes go?

### Experiment design

For each budget tier:

1. start from RTN-4 baseline
2. build best bits-only model under budget B
3. add an extra mini-budget Δ
4. compare:

   * spend Δ on more bits
   * spend Δ on targeted rank
   * split Δ between bits and rank

This is the cleanest “next byte” experiment in the whole project.

### Why this is powerful

It directly matches the revised thesis:

* **bits first**
* then test whether rank adds marginal value after the obvious precision wins are already taken

That is a much stronger and more realistic deployment story than repair-first.

---

## Phase 2H — Add two analysis tracks that will make the paper much stronger

## 1. Layer-type payoff analysis

For every selected action, aggregate gains by:

* `mlp.down_proj`
* `mlp.up_proj`
* `mlp.gate_proj`
* `attn.o_proj`
* `q/k/v`

You want to know whether the “later down_proj dominates” story survives finer allocation.

## 2. Proxy-quality correlation

For each candidate action, compare:

* weight-space error
* activation-space deviation
* realized perplexity gain

Then compute ranking correlation or top-k hit rate.

This can turn into a clean result like:

* weight norm is cheap but weak
* activation-space is much better
* direct marginal probing is best but expensive

That kind of practical tradeoff is useful.

---

## Phase 2I — Keep RTN local, but plan the transfer test now

I agree with you: keep RTN as the local mainline.

That is the right move because Phase 2 is mainly about:

* allocator logic
* budget accounting
* experiment discipline
* frontier construction

But design the code so the same action format can later run on GPTQ.

### What to prepare now

Abstract the quantizer backend so that later you can swap:

* RTN
* GPTQ
* maybe AWQ if feasible

That matters because recent work suggests both the residual structure and the value of specific allocation policies can depend strongly on how the base quantizer shapes error. ([arXiv][2])

---

## A practical Phase 2 sequence

If I were running this, I would do it in this order:

### Week 1

* freeze evaluation/accounting
* implement matrix-level bit upgrades
* generate candidate action inventory

### Week 2

* add activation-space profiling
* rank matrices by sensitivity
* run first bit-only greedy allocator

### Week 3

* add groupwise or blockwise bit actions
* rerun budget frontier
* identify stable winning layers/types

### Week 4

* implement targeted SVD rank actions on top sensitive matrices
* build rank-only frontier

### Week 5

* compare targeted bits vs targeted rank at matched budgets
* write interim conclusion

### Week 6

* run hybrid second-stage experiments
* identify whether any remaining regime favors rank

### Week 7+

* port stabilized allocator to Qwen3-1.7B
* later run GPTQ backend on stronger machine

---

## Concrete stop/go criteria

I’d define these explicitly.

### Stop pushing bit granularity when:

* the bit-only frontier becomes smooth across several budgets
* added implementation complexity stops changing top allocator decisions much

### Stop pushing rank complexity when:

* targeted rank clearly fails to close the gap to targeted bits across budgets
* or targeted rank only helps after best bit actions are exhausted

### Escalate to larger models when:

* the top selected layer types are stable across repeated local runs
* your allocator decisions are no longer changing because of harness noise

---

## My recommended Phase 2 headline

If you want a crisp framing for your notes:

> **Phase 2: Build the fixed-budget frontier for bits, rank, and second-stage hybrid repair under RTN.**

That is cleaner than “do more hybrid testing.”

## My strongest recommendation

The single best addition to your current plan is:

> **make targeted-rank a mandatory intermediate baseline before claiming bits-first as the final rule.**

Because right now your best win is against **uniform** rank.
That is enough to pivot the project, but not enough to close the argument.

Once you have:

* fine-grained bits-only frontier
* targeted-rank frontier
* hybrid second-stage frontier

you’ll have a much stronger paper than the original proposal promised.

If you want, I can turn this into a **repo-ready Phase 2 execution document** in canvas with sections like goals, tasks, run matrix, success criteria, and deliverables.

[1]: https://arxiv.org/abs/2602.17698 "[2602.17698] ScaleBITS: Scalable Bitwidth Search for Hardware-Aligned Mixed-Precision LLMs"
[2]: https://arxiv.org/abs/2402.02446 "[2402.02446] LQER: Low-Rank Quantization Error Reconstruction for LLMs"
