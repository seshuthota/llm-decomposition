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

Important note for GPTQ:

- the Kaggle GPTQ path now installs a prebuilt `gptqmodel` wheel matched to Kaggle's
  current stack (`torch 2.9`, `cu126`, `cp312`)
- this avoids the previous source-build failure
- if Kaggle changes its Python or Torch version later, set `GPTQMODEL_WHEEL_URL`
  manually before running the bootstrap

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

### GPTQ notebook sequence

After the RTN path is validated:

1. install GPTQ dependencies:

```bash
bash scripts/kaggle/bootstrap_kaggle.sh gptq
```

2. run the GPTQ smoke baseline on one GPU:

```bash
bash scripts/kaggle/run_qwen3_1p7b_gptq_smoke.sh 0
```

3. inspect the smoke result:

```bash
sed -n '1,160p' results/modal/qwen3_1p7b_gptq_smoke/R3S_Q17B/metrics.json
sed -n '1,200p' results/modal/qwen3_1p7b_gptq_smoke/R3S_Q17B/execution_status.json
```

4. only if the smoke baseline looks sane, run the full GPTQ baseline:

```bash
bash scripts/kaggle/run_qwen3_1p7b_gptq_baseline.sh 0
```

5. inspect the full baseline:

```bash
sed -n '1,160p' results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B/metrics.json
sed -n '1,200p' results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B/execution_status.json
```

6. only after a valid GPTQ baseline exists, resume the transfer runs

## Fresh Notebook Runbook

Use these cells in order.

### Cell 1: clone repo

```bash
!git clone https://github.com/seshuthota/llm-decomposition.git
%cd /kaggle/working/llm-decomposition
```

### Cell 2: set Hugging Face token

```python
import os
from kaggle_secrets import UserSecretsClient

client = UserSecretsClient()
os.environ["HF_TOKEN"] = client.get_secret("HF_TOKEN")
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
```

### Cell 3: bootstrap RTN dependencies

```bash
!bash scripts/kaggle/bootstrap_kaggle.sh rtn
```

### Cell 4: run the trusted RTN pair on two GPUs

```bash
!bash scripts/kaggle/run_two_runs.sh \
  configs/scaleup_1p7b/qwen3_1p7b_transfer_manifest.json P2B02_Q17B \
  configs/scaleup_1p7b/qwen3_1p7b_transfer_manifest.json P2R02_Q17B
```

### Cell 5: inspect RTN metrics

```bash
!sed -n '1,160p' results/qwen3_1p7b/P2B02/metrics.json
!sed -n '1,160p' results/qwen3_1p7b/P2R02/metrics.json
```

### Cell 6: bootstrap GPTQ dependencies

```bash
!bash scripts/kaggle/bootstrap_kaggle.sh gptq
```

### Cell 7: run GPTQ smoke

```bash
!bash scripts/kaggle/run_qwen3_1p7b_gptq_smoke.sh 0
```

### Cell 8: inspect GPTQ smoke result

```bash
!sed -n '1,160p' results/modal/qwen3_1p7b_gptq_smoke/R3S_Q17B/metrics.json
!sed -n '1,200p' results/modal/qwen3_1p7b_gptq_smoke/R3S_Q17B/execution_status.json
```

### Cell 9: export results

```bash
!bash scripts/kaggle/export_results.sh
```

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
- treat GPTQ smoke as the first real Kaggle GPTQ success criterion, not the full baseline
