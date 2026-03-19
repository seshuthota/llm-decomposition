# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `llm_decomposition/`; keep orchestration logic in files such as `config.py`, `executor.py`, and backend-specific evaluation code in the existing backend modules. Experiment definitions live under `configs/` as JSON manifests and per-run configs. Operational entrypoints live in `scripts/`, including local harness runners (`run_manifest.py`, `run_phase1.py`, `run_phase2.py`) and Modal/Kaggle wrappers. Research notes and references belong in `docs/`. Generated artifacts go in `results/`; treat those as outputs, not hand-edited source.

## Build, Test, and Development Commands

All commands use the `rl` Conda environment:

```bash
conda activate rl

# Validate a manifest (dry-run - checks dependencies without running models)
python scripts/run_manifest.py --manifest configs/phase2/phase2_matched_frontier_manifest.json --dry-run

# Prepare run directories only (no execution)
python scripts/run_phase1.py --prepare-only

# Run a specific experiment from a manifest
python scripts/run_manifest.py --manifest configs/phase2/phase2_matched_frontier_manifest.json --run-id P2R03

# Modal execution (remote)
./scripts/run_modal_experiment_rl.sh P2R03 configs/phase2/phase2_matched_frontier_manifest.json qwen3-0.6b-base
```

### Running Tests

This project does not yet have a formal test suite. To validate changes:

```bash
# 1. Run dry-run to validate config parsing
python scripts/run_manifest.py --manifest <path/to/manifest.json> --dry-run

# 2. Run --prepare-only to validate run preparation
python scripts/run_manifest.py --manifest <path/to/manifest.json> --prepare-only

# 3. Execute the actual script if dependencies are available
python scripts/run_manifest.py --manifest <path/to/manifest.json>
```

If adding tests, place them in a `tests/` directory using pytest:
```bash
pytest tests/                          # Run all tests
pytest tests/test_config.py            # Run single test file
pytest tests/test_config.py::test_fn   # Run single test function
```

## Coding Style & Naming Conventions

### General Principles

- **4-space indentation** (no tabs)
- **Type hints** on all public functions and class methods
- Use `from __future__ import annotations` for forward references
- Use `Path` over raw string paths for file operations
- Use `dataclass` (frozen where appropriate) for structured records

### Imports

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_decomposition.config import RunConfig
from llm_decomposition.io import write_json
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Modules | `snake_case` | `hf_backend.py` |
| Functions | `snake_case` | `load_manifest()` |
| Variables | `snake_case` | `run_configs` |
| Classes | `PascalCase` | `RunConfig` |
| Constants | `UPPER_SNAKE` | `REQUIRED_TOP_LEVEL_KEYS` |
| Shell scripts | `snake_case` | `run_modal_experiment_rl.sh` |

### Functions

- Prefer small, composable functions over large scripts
- Use clear, descriptive names (e.g., `load_manifest`, `validate_run_config`)
- Keep functions focused on a single responsibility

### Error Handling

- Use specific exception types with clear messages
- Include context in error messages (e.g., `f"{config_path}: missing required keys: {missing_list}"`)
- Validate inputs early with clear error messages

### Docstrings

Use Google-style docstrings for public functions:

```python
def load_manifest(root: Path, manifest_rel_path: str) -> Manifest:
    """Load and parse an experiment manifest.

    Args:
        root: The repository root path.
        manifest_rel_path: Relative path to the manifest JSON file.

    Returns:
        A Manifest object containing the parsed configuration.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        ValueError: If the manifest format is invalid.
    """
```

## Testing Guidelines

There is no committed `tests/` suite or coverage gate yet. The current validation standard is:

1. Run the relevant manifest with `--dry-run` to validate dependencies
2. Execute the targeted script or Modal wrapper if dependencies are available
3. Inspect regenerated artifacts in `results/...`

If you add reusable logic, add `pytest`-style tests in a new `tests/` directory:
- Use `test_<module>.py` naming convention
- Place tests in a `tests/` directory at the repo root

## Commit & Pull Request Guidelines

Keep commit subjects short, imperative, and capitalized:
- `Add Kaggle execution scaffolding`
- `Fix config validation for missing model.name`
- `Refactor executor to support new method`

Scope each commit to one experiment harness change, config batch, or documentation update.

## Security & Configuration Tips

- **Do not commit** `.env`, local caches, or `llm-decomposition-results/`
- Keep secrets and provider tokens in local environment variables
- Avoid rewriting historical `results/` directories unless intentionally rerunning that experiment and documenting the replacement
- Treat `results/` as outputs, not hand-edited source

## Common Workflows

### Running a New Experiment

1. Create a config file in `configs/phase<N>/`
2. Add the config to a manifest file
3. Run with `--dry-run` to validate
4. Execute locally or via Modal
5. Document results in `results/<run_id>/notes.md`

### Adding a New Method

1. Add method spec to `llm_decomposition/methods.py`
2. Implement execution logic in appropriate backend
3. Add dispatch case in `executor.py`
4. Add config validation if needed
5. Test with a dry-run
