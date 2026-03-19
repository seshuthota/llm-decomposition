#!/usr/bin/env python3
"""Generate multi-seed configs from a base config for stability analysis.

This script reads a base run config, generates multiple copies with different
calibration seeds, and creates a manifest file for running the multi-seed
stability experiment.

Usage::

    conda run -n rl python scripts/generate_multiseed_configs.py \
        --base-config configs/scaleup_1p7b_gptq/g2r02w_qwen3_1p7b_gptq_rank_weight_1p0pct.json \
        --seeds 42 123 456 \
        --output-dir configs/multiseed

Example::

    # Generate multi-seed configs for rank policy
    conda run -n rl python scripts/generate_multiseed_configs.py \
        --base-config configs/scaleup_1p7b_gptq/g2r02w_qwen3_1p7b_gptq_rank_weight_1p0pct.json \
        --seeds 42 123 456 \
        --output-dir configs/multiseed

    # Generate multi-seed configs for bits policy
    conda run -n rl python scripts/generate_multiseed_configs.py \
        --base-config configs/scaleup_1p7b_gptq/g2b02w_qwen3_1p7b_gptq_bits_weight_1p0pct.json \
        --seeds 42 123 456 \
        --output-dir configs/multiseed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate multi-seed configs from a base config for stability analysis."
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Path to the base config JSON file.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="List of seeds to generate configs for (default: 42 123 456).",
    )
    parser.add_argument(
        "--output-dir",
        default="configs/multiseed",
        help="Output directory for generated configs (default: configs/multiseed).",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    if not base_path.exists():
        print(f"Error: Base config not found: {base_path}")
        return 1

    base_path = base_path.resolve()
    with open(base_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    base_run_id = base_config["run_id"]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = []
    for seed in args.seeds:
        new_config = dict(base_config)
        new_run_id = f"{base_run_id}_s{seed}"
        new_config["run_id"] = new_run_id

        if "calibration" not in new_config:
            new_config["calibration"] = {}

        new_config["calibration"]["sampling"] = "seeded_shuffle"
        new_config["calibration"]["seed"] = seed

        method_cfg = dict(new_config.get("method", {}))
        if (
            method_cfg.get("base_method") == "gptq"
            and method_cfg.get("name") in {"targeted_mixed_precision", "targeted_svd_rank", "hybrid_second_stage"}
            and "selection_profile_source" not in method_cfg
        ):
            method_cfg["selection_profile_source"] = "current_base_model"
        new_config["method"] = method_cfg

        new_outputs = dict(new_config.get("outputs", {}))
        base_results_dir = new_outputs.get("results_dir", f"results/{base_run_id}")
        new_outputs["results_dir"] = f"{base_results_dir}_s{seed}"
        new_config["outputs"] = new_outputs

        new_phase = new_config.get("phase", "multiseed")
        if not new_phase.endswith(f"_{seed}"):
            new_config["phase"] = f"{new_phase}_s{seed}"

        output_filename = f"{new_run_id}.json"
        output_path = output_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=2)

        relative_path = str(output_path.relative_to(REPO_ROOT))
        generated_paths.append(relative_path)
        print(f"  Generated: {relative_path} (seed={seed})")

    manifest_path = output_dir / f"{base_run_id}_multiseed_manifest.json"
    manifest = {
        "phase": f"multiseed_{base_run_id}",
        "description": f"Multi-seed stability analysis for {base_run_id}",
        "runs": generated_paths,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(args.seeds)} configs in {output_dir}")
    print(f"Manifest: {manifest_path}")
    print("\nTo run:")
    print(
        f"  python scripts/run_manifest.py --manifest {manifest_path.relative_to(output_dir.parent)} --dry-run"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
