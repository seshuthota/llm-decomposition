# Config-Driven Harness

The experiment harness is designed to be model-agnostic at the orchestration layer. A run is defined entirely by JSON config, and manifests simply point to a set of run configs.

## Entry Points

- `scripts/run_manifest.py` is the generic runner for any manifest.
- `scripts/run_phase1.py` is a phase-specific wrapper for the current baseline cycle.

Run them from the `conda` env `rl`:

```bash
conda run -n rl python scripts/run_manifest.py --manifest configs/phase1/phase1_manifest.json --dry-run
```

## Current Flow

1. Prepare run directories and template outputs from a manifest.
2. Validate method requirements for each selected run.
3. Record execution readiness in each run folder.
4. Later, plug in the actual Hugging Face backend behind the same interface.

## Why This Structure

This keeps the outer harness stable even when:

- model families change,
- quantization methods change,
- evaluation datasets expand,
- or multiple phases are added.

The config interface becomes the contract, and backend-specific code stays behind method dispatch.

## Minimal Run Config Shape

Each run config must include:

- `run_id`
- `phase`
- `description`
- `model`
- `method`
- `calibration`
- `evaluation`
- `profiling`
- `outputs`

## Method Dispatch

The generic executor currently knows about:

- `full_precision`
- `rtn`
- `gptq`

Method-specific dependency checks are handled in [llm_decomposition/methods.py](/home/seshu/Documents/Python/llm-decomposition/llm_decomposition/methods.py).

## Next Backend Step

The generic backend now supports:

- full-precision Hugging Face evaluation,
- RTN-style weight quantization for `nn.Linear`,
- perplexity evaluation,
- basic memory accounting,
- layerwise weight error summaries,
- activation-space error profiling,
- residual SVD profiling.

The remaining backend milestone is GPTQ support.
