# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `llm_decomposition/`; keep orchestration logic in files such as `config.py`, `executor.py`, and backend-specific evaluation code in the existing backend modules. Experiment definitions live under `configs/` as JSON manifests and per-run configs. Operational entrypoints live in `scripts/`, including local harness runners (`run_manifest.py`, `run_phase1.py`, `run_phase2.py`) and Modal/Kaggle wrappers. Research notes and references belong in `docs/`. Generated artifacts go in `results/`; treat those as outputs, not hand-edited source.

## Build, Test, and Development Commands

Use the existing `rl` Conda environment referenced in the docs.

```bash
conda run -n rl python scripts/run_manifest.py --manifest configs/phase2/phase2_matched_frontier_manifest.json --dry-run
conda run -n rl python scripts/run_phase1.py --prepare-only
conda run -n rl python scripts/summarize_phase1_results.py
./scripts/run_modal_experiment_rl.sh P2R03 configs/phase2/phase2_matched_frontier_manifest.json qwen3-0.6b-base
```

`--dry-run` validates dependencies and writes execution readiness without launching a model. `--prepare-only` creates run directories and manifest summaries. Modal shell wrappers are the standard remote execution path.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, type hints on public functions, `Path` over raw string paths, and `dataclass` for structured records where appropriate. Use `snake_case` for Python modules, functions, variables, and shell scripts. Keep config filenames descriptive and phase-oriented, for example `p2r02_qwen3_0p6b_rank_activation_1p0pct.json`. Prefer small, composable functions over large notebook-style scripts.

## Testing Guidelines

There is no committed `tests/` suite or coverage gate yet. The current validation standard is:

- run the relevant manifest with `--dry-run`
- execute the targeted script or Modal wrapper if dependencies are available
- inspect regenerated artifacts in `results/...`

If you add reusable logic, add `pytest`-style tests in a new `tests/` directory using `test_<module>.py` naming.

## Commit & Pull Request Guidelines

Keep commit subjects short, imperative, and capitalized, matching history such as `Add Kaggle execution scaffolding`. Scope each commit to one experiment harness change, config batch, or documentation update. PRs should describe the research or tooling impact, list changed manifests/scripts, link the relevant roadmap or proposal doc, and include representative output paths when results changed.

## Security & Configuration Tips

Do not commit `.env`, local caches, or `llm-decomposition-results/`. Keep secrets and provider tokens in local environment variables. Avoid rewriting historical `results/` directories unless you are intentionally rerunning that experiment and documenting the replacement.
