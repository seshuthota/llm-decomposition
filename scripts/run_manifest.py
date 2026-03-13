#!/usr/bin/env python3
"""Generic config-driven harness for preparing and executing experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_decomposition.config import load_manifest
from llm_decomposition.executor import ExperimentExecutor
from llm_decomposition.prepare import prepare_run, write_manifest_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or execute experiment manifests.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to a manifest JSON file, relative to the repo root.",
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
        help="Prepare run directories and template outputs only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check execution readiness without starting the actual backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = REPO_ROOT
    manifest = load_manifest(root, args.manifest)
    selected_run_ids = set(args.run_id)

    prepared_runs = []
    selected_configs = []
    for config in manifest.run_configs:
        if selected_run_ids and config.run_id not in selected_run_ids:
            continue
        prepared_runs.append(prepare_run(root, config))
        selected_configs.append(config)

    summary_path = write_manifest_summary(root, manifest, prepared_runs)
    print(f"Prepared {len(prepared_runs)} runs from manifest {manifest.path.as_posix()}.")
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
