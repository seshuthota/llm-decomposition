# Phase 2 Action Schema

This document defines the canonical action format for Phase 2 allocator experiments.

## Purpose

Every candidate action in Phase 2 should be representable with the same schema so that:

- bits-only and rank-only actions can be compared fairly
- action costs are logged consistently
- gain-per-byte analysis can be done without per-method special cases
- the same action format can later be reused for `GPTQ`

## Required Fields

Each action record should contain:

- `action_id`: stable unique identifier
- `action_type`: `bit_upgrade` or `rank_repair`
- `target_granularity`: `matrix`, `group`, or `block`
- `target_name`: fully qualified layer or submatrix path
- `base_run_id`: anchor run, currently `R2`
- `byte_cost`: estimated additional persistent bytes
- `proxy_family`: `weight`, `activation`, or `measured`
- `proxy_score`: scalar score before selection
- `predicted_gain_per_byte`: optional estimated gain efficiency
- `bit_from`: optional source bit-width for bit actions
- `bit_to`: optional target bit-width for bit actions
- `rank`: optional rank for rank actions
- `metadata`: free-form method-specific details

## Execution-Time Fields

These fields are not required in the initial candidate inventory, but should be logged once an action is evaluated or selected:

- `selected`: whether the allocator chose the action
- `selection_order`: action position in the allocation sequence
- `cumulative_budget_bytes`: total extra bytes consumed after selecting the action
- `realized_perplexity`: perplexity after applying the action within the current run
- `realized_gain`: quality improvement relative to the base point or previous step
- `status`: `pending`, `evaluated`, `selected`, `skipped`, or `rejected`

## Candidate Pool Metadata

Each action inventory should also define:

- `candidate_pool_name`
- `source_run_id`
- `source_layer_errors_path`
- `selection_rule`
- `seed`

This keeps the targeted-bit and targeted-rank pools aligned.

## JSON Example

```json
{
  "action_id": "bit_matrix_layer27_downproj_4to8",
  "action_type": "bit_upgrade",
  "target_granularity": "matrix",
  "target_name": "model.layers.27.mlp.down_proj",
  "base_run_id": "R2",
  "byte_cost": 1572864,
  "proxy_family": "activation",
  "proxy_score": 0.6607827854,
  "predicted_gain_per_byte": null,
  "bit_from": 4,
  "bit_to": 8,
  "rank": null,
  "metadata": {
    "group_size": 128,
    "symmetric": true
  }
}
```

```json
{
  "action_id": "rank_matrix_layer27_downproj_r16",
  "action_type": "rank_repair",
  "target_granularity": "matrix",
  "target_name": "model.layers.27.mlp.down_proj",
  "base_run_id": "R2",
  "byte_cost": 131072,
  "proxy_family": "activation",
  "proxy_score": 0.6607827854,
  "predicted_gain_per_byte": null,
  "bit_from": null,
  "bit_to": null,
  "rank": 16,
  "metadata": {
    "factor_dtype_bytes": 2,
    "construction": "truncated_svd"
  }
}
```

## Phase 2A Defaults

The first pass should use:

- `target_granularity = matrix`
- shared candidate pool from `R2`
- proxy family comparisons across `weight` and `activation`
- static greedy allocation first

## Planned Files

Recommended Phase 2 config/reference files:

- `configs/phase2/phase2_action_schema.json`
- `configs/phase2/qwen3_0p6b_r2_top12_candidate_pool.json`
- `configs/phase2/phase2_first_pass_manifest.json`

The markdown file is the human-readable contract. JSON files are the machine-readable instances.
