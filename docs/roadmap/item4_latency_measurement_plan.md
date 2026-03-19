# Item 4 Latency Measurement Plan

This document is the working implementation plan and progress tracker for Item 4 of the paper-readiness roadmap.

Goal:

- measure practical inference cost for the paper-critical GPTQ comparison points
- report latency and peak VRAM honestly, rather than relying on the existing eval-pass timing metric

Status:

- overall status: benchmark contract frozen; latency module and Modal runner implemented; full 12-run latency matrix completed and saved
- owner: current paper-readiness branch
- canonical roadmap parent: `docs/roadmap/paper_readiness_plan.md`

## Why This Exists

The current `latency_ms_per_token` metric in `llm_decomposition/hf_utils.py` is derived from perplexity evaluation forward passes:

- `model(input_ids=..., labels=...)`

That is useful as a rough secondary signal, but it is not a decode benchmark. Item 4 needs a dedicated generation-style latency path so the paper can state the runtime cost of choosing rank repair over extra bits.

## Scope

Paper-critical benchmark matrix:

- Qwen3-1.7B GPTQ on `A10G`
  - baseline 4-bit: `R3_Q17B`
  - best bits-only: `G2B03_Q17B`
  - best rank-only: `G2R02_Q17B`
- Qwen3-8B GPTQ on `A100`
  - baseline 4-bit: `R3_Q8B`
  - best bits-only: `G2B02_Q8B`
  - best rank-only: `G2R02_Q8B`

For each model/policy point:

- batch size `1`
- batch size `8`

Total planned benchmark jobs:

- `6` policy points
- `2` batch sizes each
- `12` main latency jobs

## Benchmark Contract

These settings should be held fixed across all runs unless we explicitly revise the plan.

- inference mode: autoregressive `generate(...)` with KV cache enabled
- decoding policy: greedy decoding
  - `do_sample = False`
  - no temperature / top-p / top-k sampling path
- prompt length: `512`
- decode length: `128` new tokens
- fixed-length decode target:
  - benchmark implementation should force exactly `128` generated tokens when feasible
  - recommended implementation: `max_new_tokens = 128` and `min_new_tokens = 128`
- warmup iterations: `3`
- timed repetitions: `10`
- prompt construction:
  - use one fixed textual prompt template for all runs
  - tokenize separately per model family
  - truncate or pad to exactly `512` input tokens
  - batch `8` should duplicate the same prompt eight times to isolate policy effects from prompt-mixture effects
- precision/device placement: match the deployed GPTQ policy artifact being benchmarked
- tokenizer/model source:
  - use the same staged local model path / tokenizer path approach as the working GPTQ Modal runner
- hardware lock:
  - `1.7B` jobs run only on `A10G`
  - `8B` jobs run only on `A100`

Primary paper metrics:

- primary throughput metric: `decode_tokens_per_sec`
- primary latency metric: `decode_ms_per_token`
- primary memory metric: `peak_vram_mb`

Secondary diagnostic metrics:

- `end_to_end_tokens_per_sec`
- `first_token_latency_ms`

Metrics to report per job:

- `decode_tokens_per_sec`
- `decode_ms_per_token`
- `end_to_end_tokens_per_sec`
- `first_token_latency_ms`
- `peak_vram_mb`
- identifying metadata:
  - model
  - hardware
  - policy
  - run_id
  - batch_size
  - prompt_len
  - decode_len

### Measurement Rules

- warmup iterations do not contribute to reported metrics
- each timed repetition must:
  - reset CUDA peak-memory stats before measurement
  - synchronize CUDA before starting timing
  - synchronize CUDA after completion before reading elapsed time
- report both:
  - per-repetition raw measurements
  - aggregate mean and standard deviation
- benchmark should record the exact generated token count and treat deviations from `128` as a warning or error
- first-token latency should be measured explicitly, not inferred from total decode time

### Source Artifact Policy

Latency jobs should benchmark the canonical completed runs, not multiseed replicas.

Use these source artifacts:

- `Qwen3-1.7B`
  - baseline: `results/modal/qwen3_1p7b_gptq_baselines/R3_Q17B`
  - bits: `results/modal/qwen3_1p7b_gptq_transfer/G2B03_Q17B`
  - rank: `results/modal/qwen3_1p7b_gptq_transfer/G2R02_Q17B`
- `Qwen3-8B`
  - baseline: `results/modal/qwen3_8b_gptq_baselines/R3_Q8B`
  - bits: `results/modal/qwen3_8b_gptq_transfer/G2B02_Q8B`
  - rank: `results/modal/qwen3_8b_gptq_transfer/G2R02_Q8B`

These are the paper-critical point estimates already used elsewhere in the roadmap and analysis docs.

## Implementation Plan

### 1. Benchmark Contract Lock

- [x] Fixed benchmark settings chosen
- [x] Paper table primary numbers fixed to:
  - decode-only throughput / latency
  - peak VRAM
- [x] Secondary diagnostics retained:
  - end-to-end throughput
  - first-token latency
- [x] Prompt policy fixed:
  - one shared textual template
  - per-model tokenization
  - exact-token truncation/padding
- [x] Source run ids fixed for both scales

Deliverable:

- this file is now the frozen benchmark spec unless later explicitly revised

### 2. Dedicated Latency Module

- [x] Add `llm_decomposition/latency_benchmark.py`
- [x] Implement helpers to:
  - load a benchmark target from an existing run artifact or resolved config
  - prepare fixed prompt batches
  - run warmup + timed repetitions
  - measure first-token latency separately from steady-state decode throughput
  - read peak VRAM with `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()`
- [x] Ensure outputs are JSON-safe and stable

Deliverable:

- reusable latency benchmark library independent of the normal execute/eval path

### 3. Modal Runner

- [x] Add a dedicated Modal latency entrypoint
- [x] Reuse:
  - model volume
  - results volume
  - HF token wiring
  - GPU selection helpers
- [x] Keep latency artifacts separate from normal experiment execution artifacts

Recommended output root:

- `results/modal_latency/...`

Deliverable:

- one-shot remote latency benchmark runner for existing GPTQ artifacts

### 4. Benchmark Config Layer

- [x] Add latency configs under `configs/latency/`
- [x] Each config defines:
  - source run id or artifact path
  - hardware target
  - batch size
  - prompt length
  - decode length
  - warmup count
  - repetition count
- [x] Encode the paper-critical matrix explicitly so jobs are easy to audit
- [x] Add a dedicated launcher wrapper:
  - `scripts/run_modal_latency_gptq_rl.sh`
- [x] Add a top-level latency manifest:
  - `configs/latency/item4_latency_manifest.json`

Target benchmark set:

- `1.7B` / `A10G`
  - baseline
  - bits
  - rank
  - each at batch `1` and `8`
- `8B` / `A100`
  - baseline
  - bits
  - rank
  - each at batch `1` and `8`

### 5. Validation

- [x] Run one smoke benchmark first before launching the full matrix
- [x] Verify:
  - output JSON schema
  - stable metrics across repetitions
  - peak VRAM recording works
  - tokenizer/model loading is local when expected
- [x] Only then launch the full matrix

Recommended first smoke test:

- `Qwen3-8B GPTQ bits` on `A100`, batch `1`

Completed smoke validation:

- canonical artifact: `results/modal_latency/qwen3_8b_gptq_transfer/G2B02_Q8B__bs1`
- status: completed
- selection fallback fix required for targeted GPTQ latency runs:
  - force `selection_profile_source: "current_base_model"` in the latency runner
- launcher fix required:
  - pass Modal `-d/--detach` before the function reference in `scripts/run_modal_latency_gptq_rl.sh`
- first measured numbers for `G2B02_Q8B__bs1`:
  - `decode_tokens_per_sec = 13.7762`
  - `decode_ms_per_token = 72.5901`
  - `first_token_latency_ms = 77.7610`
  - `peak_vram_mb = 6255.19`

### 6. Main Execution

- [x] Run all `8B / A100` latency jobs
- [x] Run all `1.7B / A10G` latency jobs
- [x] Save and fetch all remote artifacts to the repo
- [x] Verify each job wrote final metrics cleanly

Current progress:

- completed and saved:
  - `results/modal_latency/qwen3_8b_gptq_baselines/R3_Q8B__bs1`
  - `results/modal_latency/qwen3_8b_gptq_baselines/R3_Q8B__bs8`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2B02_Q8B__bs1`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2B02_Q8B__bs8`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2R02_Q8B__bs1`
  - `results/modal_latency/qwen3_8b_gptq_transfer/G2R02_Q8B__bs8`
  - `results/modal_latency/qwen3_1p7b_gptq_baselines/R3_Q17B__bs1`
  - `results/modal_latency/qwen3_1p7b_gptq_baselines/R3_Q17B__bs8`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2B03_Q17B__bs1`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2B03_Q17B__bs8`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2R02_Q17B__bs1`
  - `results/modal_latency/qwen3_1p7b_gptq_transfer/G2R02_Q17B__bs8`
- next recommended job:
  - write `docs/experiments/item4_latency_analysis.md`

Current 8B summary:

- batch `1`:
  - baseline: `13.996 tok/s`, `71.45 ms/token`, `6178 MB`
  - bits: `13.776 tok/s`, `72.59 ms/token`, `6255 MB`
  - rank: `8.513 tok/s`, `117.48 ms/token`, `6445 MB`
- batch `8`:
  - baseline: `69.413 tok/s`, `14.41 ms/token`, `6958 MB`
  - bits: `112.051 tok/s`, `8.93 ms/token`, `22663 MB`
  - rank: `115.581 tok/s`, `8.65 ms/token`, `7230 MB`

Current interpretation:

- at low-batch interactive decode (`batch=1`), rank is materially slower than both baseline and bits
- at higher batching (`batch=8`), rank is not slower on decode throughput; both targeted policies exceed the baseline throughput
- practical latency conclusions are therefore workload-dependent, not one-dimensional
- peak VRAM still separates the policies and should be reported alongside throughput

Current 1.7B summary:

- batch `1`:
  - baseline: `15.689 tok/s`, `63.89 ms/token`, `1488 MB`
  - bits: `18.484 tok/s`, `54.12 ms/token`, `4825 MB`
  - rank: `19.978 tok/s`, `50.06 ms/token`, `1634 MB`
- batch `8`:
  - baseline: `138.031 tok/s`, `7.24 ms/token`, `2031 MB`
  - bits: `140.393 tok/s`, `7.13 ms/token`, `5366 MB`
  - rank: `118.776 tok/s`, `8.45 ms/token`, `2178 MB`

Current cross-scale interpretation:

- `8B`:
  - `batch=1`: rank is much slower than bits/baseline
  - `batch=8`: rank is not slower than bits on decode throughput, but policy behavior is batch-sensitive and VRAM differs materially
- `1.7B`:
  - `batch=1`: both targeted policies are faster than baseline on decode throughput
  - `batch=8`: bits is slightly faster than baseline; rank is slower than both bits and baseline
- latency conclusions are workload-dependent and must be reported jointly with peak VRAM

### 7. Summary + Analysis

- [x] Add `scripts/build_latency_summary.py`
- [x] Generate:
  - `results/analysis/latency_item4_summary.csv`
- [ ] Compute:
  - rank vs bits latency overhead percentage
  - rank vs baseline latency overhead percentage
  - bits vs baseline latency overhead percentage
  - peak VRAM deltas

Deliverable:

- one generated CSV that can feed the paper table and plots
- current CSV status:
  - `12` completed rows covering the full Item 4 matrix

### 8. Reporting

- [x] Add `docs/experiments/item4_latency_analysis.md`
- [x] Update `docs/experiments/README.md`
- [x] Update `docs/roadmap/paper_readiness_plan.md`
- [ ] Write the paper-facing conclusion:
  - whether rank repair’s quality gain is worth its latency overhead
  - whether bits is the practical default when throughput matters

## Proposed File-Level Changes

Expected additions:

- `llm_decomposition/latency_benchmark.py`
- `scripts/modal_latency_gptq.py` or a new latency entrypoint in `scripts/modal_experiment_gptq.py`
- `configs/latency/...`
- `scripts/build_latency_summary.py`
- `docs/experiments/item4_latency_analysis.md`
- `results/analysis/latency_item4_summary.csv`

Expected documentation updates:

- `docs/roadmap/paper_readiness_plan.md`
- `docs/experiments/README.md`

## Progress Log

- [x] Item 4 implementation plan drafted
- [x] benchmark contract frozen
- [x] latency module implemented
- [x] Modal runner implemented
- [x] latency configs created
- [x] smoke test completed
- [x] `8B / A100` matrix completed
- [x] `1.7B / A10G` matrix completed
- [x] summary CSV generated
- [x] analysis report written
- [x] roadmap updated with final Item 4 results

## Decision Notes

Current recommendation:

- do not reuse the existing eval-pass `latency_ms_per_token` as the paper’s main latency result
- implement a dedicated generation benchmark and keep the old metric only as a secondary diagnostic
- use decode-only throughput / latency as the primary reported numbers in the paper table

Implementation notes:

- current implementation files:
  - `llm_decomposition/latency_benchmark.py`
  - `scripts/modal_experiment_gptq.py` via `run_latency_remote`
  - `scripts/run_modal_latency_gptq_rl.sh`
  - `scripts/build_latency_summary.py`
  - `configs/latency/item4_latency_manifest.json`
- final summary status:
  - `results/analysis/latency_item4_summary.csv` contains all `12` completed latency runs
