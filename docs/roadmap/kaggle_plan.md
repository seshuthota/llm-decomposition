# Kaggle Execution Plan

This plan is for running the repo from a fresh Kaggle notebook session.

## What The Repo Already Supports

- config-driven execution through `scripts/run_manifest.py`
- environment-variable based Hugging Face authentication via `HF_TOKEN`
- direct Hugging Face model and dataset loading
- run-specific outputs in `results/`

## What Is Missing For Kaggle

The repo was originally built around:

- a local `conda` environment named `rl`
- Modal remote execution for larger runs

So the main Kaggle gaps are:

1. no notebook-first dependency bootstrap
2. no Kaggle-specific cache/result directory setup
3. no explicit output export step
4. no practical two-GPU execution path

## Recommended Repo Changes

The following lightweight changes are the right first step:

1. `requirements-kaggle-rtn.txt`
   - base runtime for the trusted RTN path
2. `requirements-kaggle-gptq.txt`
   - optional GPTQ add-on path
3. `scripts/kaggle/bootstrap_kaggle.sh`
   - installs dependencies
   - sets cache directories under `/kaggle/working`
4. `scripts/kaggle/run_manifest_on_gpu.sh`
   - pins a single run to one selected GPU
5. `scripts/kaggle/run_two_runs.sh`
   - uses two GPUs by running two independent runs in parallel
6. `scripts/kaggle/export_results.sh`
   - zips outputs for download

## Why Not True Multi-GPU Yet

The current code is single-process and single-device:

- model loading is `model.to(cuda)`
- evaluation is a simple per-sequence loop
- the RTN path deep-copies the full model in memory

That means true multi-GPU model execution would require a more invasive refactor:

- `accelerate`
- model sharding / `device_map`
- or distributed evaluation logic

For Kaggle, the low-risk approach is:

- use one GPU per run
- run two independent experiments in parallel if two GPUs are available

This uses both GPUs without destabilizing the core code.

## Fresh Notebook Workflow

### 1. Start The Notebook

Requirements:

- enable internet if downloading from Hugging Face directly
- add `HF_TOKEN` as a Kaggle secret if the model requires auth or higher rate limits

### 2. Clone The Repo

```bash
git clone <repo-url>
cd llm-decomposition
```

### 3. Bootstrap Dependencies

For RTN work:

```bash
bash scripts/kaggle/bootstrap_kaggle.sh rtn
```

For GPTQ work:

```bash
bash scripts/kaggle/bootstrap_kaggle.sh gptq
```

### 4. Set Hugging Face Token

In the notebook:

```python
import os
from kaggle_secrets import UserSecretsClient

client = UserSecretsClient()
os.environ["HF_TOKEN"] = client.get_secret("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
```

### 5. Run A Baseline

Example:

```bash
python scripts/run_manifest.py \
  --manifest configs/scaleup_1p7b/qwen3_1p7b_baselines_manifest.json \
  --run-id R2_Q17B
```

### 6. Use Two GPUs If Available

Example:

```bash
bash scripts/kaggle/run_two_runs.sh \
  configs/scaleup_1p7b/qwen3_1p7b_transfer_manifest.json P2B02_Q17B \
  configs/scaleup_1p7b/qwen3_1p7b_transfer_manifest.json P2R02_Q17B
```

This is the recommended first use of dual GPUs on Kaggle.

## Recommended Kaggle Experiment Order

### First Pass

Do not start with GPTQ.

Start with the already-trusted RTN path:

1. `R2_Q17B`
2. `P2B02_Q17B`
3. `P2R02_Q17B`

This validates:

- environment setup
- model download
- dataset download
- result persistence
- GPU pinning

### Second Pass

Only after RTN works cleanly on Kaggle:

1. try the GPTQ smoke baseline
2. then retry full `R3_Q17B`
3. only then run `G2B02_Q17B` / `G2R02_Q17B`

## Downloading Outputs

At the end:

```bash
bash scripts/kaggle/export_results.sh
```

Then download:

- `/kaggle/working/llm-decomposition-results.zip`

## Practical Recommendation

For the first Kaggle session:

- validate the repo with the `RTN` path first
- use both GPUs only for parallel independent runs
- do not attempt true multi-GPU model execution yet
- only move to `GPTQ` after the notebook setup is proven stable
