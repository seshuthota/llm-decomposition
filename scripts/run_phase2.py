#!/usr/bin/env python3
"""Prepare and optionally dry-run the first Phase-2 allocator runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_decomposition.config import load_manifest
from llm_decomposition.executor import ExperimentExecutor
from llm_decomposition.prepare import prepare_run, write_manifest_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase-2 allocator runs.")
    parser.add_argument(
        "--manifest",
        default="configs/phase2/phase2_first_pass_manifest.json",
        help="Path to the Phase-2 manifest JSON file.",
    )
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Optional run id filter. May be provided multiple times.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare run directories and metadata without starting execution.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check whether selected runs are executable without starting them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = REPO_ROOT
    manifest = load_manifest(root, args.manifest)
    run_filters = set(args.run_id)
    prepared_runs: list[dict[str, Any]] = []
    selected_configs = []
    for config in manifest.run_configs:
        if run_filters and config.run_id not in run_filters:
            continue
        prepared_runs.append(prepare_run(root, config))
        selected_configs.append(config)

    summary_path = write_manifest_summary(root, manifest, prepared_runs)

    if not args.prepare_only:
        print(f"Prepared {len(prepared_runs)} phase-2 runs.")
        for run in prepared_runs:
            bit_width = run["bit_width"]
            bit_label = "n/a" if bit_width is None else str(bit_width)
            print(
                f"- {run['run_id']}: method={run['method']} bits={bit_label} "
                f"results_dir={run['results_dir']}"
            )
        print(f"Summary written to {summary_path.as_posix()}")

    if args.prepare_only:
        return 0

    executor = ExperimentExecutor(root)
    for config in selected_configs:
        result = executor.execute(config, dry_run=args.dry_run)
        print(
            f"- {result.run_id}: status={result.status} method={config.method_name} "
            f"missing={','.join(result.missing_dependencies) or 'none'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
