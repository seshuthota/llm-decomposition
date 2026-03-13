# Modal Runner

This repo can now run one existing experiment config on a Modal GPU without changing the core experiment code.

## What It Does

- mounts the repo into a Modal container
- optionally mounts a Modal Volume that contains a locally uploaded model snapshot
- runs one existing config-driven Phase 1 or Phase 2 experiment remotely
- writes the remote JSON artifacts back into this repo under `results/modal/...`

The entrypoint is:

- [scripts/modal_experiment.py](/home/seshu/Documents/Python/llm-decomposition/scripts/modal_experiment.py)

## Recommended Flow

### 1. Upload the cached Qwen model once

Use:

- [scripts/modal_upload_qwen3_0p6b_base_rl.sh](/home/seshu/Documents/Python/llm-decomposition/scripts/modal_upload_qwen3_0p6b_base_rl.sh)
- [scripts/modal_upload_qwen3_1p7b_base_rl.sh](/home/seshu/Documents/Python/llm-decomposition/scripts/modal_upload_qwen3_1p7b_base_rl.sh)

Default behavior:

- creates or reuses Modal Volume `llm-decomposition-models`
- uploads the local cached snapshot for `Qwen/Qwen3-0.6B-Base`
- stores it under `/qwen3-0.6b-base` inside the volume

Command:

```bash
./scripts/modal_upload_qwen3_0p6b_base_rl.sh
```

For the next scale-up path:

```bash
./scripts/cache_qwen3_1p7b_base_rl.sh
./scripts/modal_upload_qwen3_1p7b_base_rl.sh
```

### 2. Run one experiment remotely

Use:

- [scripts/run_modal_experiment_rl.sh](/home/seshu/Documents/Python/llm-decomposition/scripts/run_modal_experiment_rl.sh)

Default command:

```bash
./scripts/run_modal_experiment_rl.sh P2R03 configs/phase2/phase2_matched_frontier_manifest.json qwen3-0.6b-base
```

This runs the existing `P2R03` config remotely, but overrides the model path so `transformers` loads from:

```text
/vol/models/qwen3-0.6b-base
```

inside the Modal container.

For the `1.7B` baseline:

```bash
./scripts/run_qwen3_1p7b_r2_modal.sh
```

## Environment Variables

You can override these when invoking `modal run` or the shell wrapper:

- `MODAL_GPU`
  - default: `T4`
- `MODAL_TIMEOUT`
  - default: `14400`
- `MODAL_MODEL_VOLUME`
  - default: `llm-decomposition-models`

Example:

```bash
MODAL_GPU=A100 ./scripts/run_modal_experiment_rl.sh P2R03 configs/phase2/phase2_matched_frontier_manifest.json qwen3-0.6b-base
```

## Output Location

Remote run artifacts are written back locally under:

```text
results/modal/<phase>/<run_id>/
```

The Modal wrapper also writes:

- `modal_run.log`

into that run directory with the captured remote stdout/stderr.

## Current Scope

This is designed for:

- existing config-driven runs
- a single remote run per invocation
- local model snapshots uploaded once to a Modal Volume

It does not yet:

- save modified model weights back to the volume
- fan out many runs in parallel
- use a separate Modal results volume
