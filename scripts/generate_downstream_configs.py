#!/usr/bin/env python3
"""Generate downstream evaluation configs from existing run configs.

This script reads existing run configs, adds a ``downstream`` section with
the standard task suite, assigns new run IDs with a ``DS_`` prefix, and
writes the augmented configs into ``configs/downstream/``.  It also produces
a manifest file for each model group.

Usage::

    conda run -n rl python scripts/generate_downstream_configs.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DOWNSTREAM_SECTION = {
    "enabled": True,
    "tasks": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "piqa",
        "boolq",
    ],
    "num_fewshot": {
        "hellaswag": 0,
        "arc_easy": 0,
        "arc_challenge": 25,
        "winogrande": 5,
        "piqa": 0,
        "boolq": 0,
    },
    "batch_size": 4,
    "output_file": "downstream_metrics.json",
}


# -----------------------------------------------------------------------
# Each entry:  (new_run_id, source_config_path_relative_to_repo)
# -----------------------------------------------------------------------

QWEN3_1P7B_GPTQ_RUNS = [
    # Full precision
    {
        "run_id": "DS_FP_Q17B",
        "phase": "downstream_qwen3_1p7b",
        "description": "Full-precision Qwen3-1.7B downstream evaluation.",
        "model": {
            "name": "Qwen/Qwen3-1.7B-Base",
            "tokenizer_name": "Qwen/Qwen3-1.7B-Base",
            "dtype_preference": ["float16"],
        },
        "method": {"name": "full_precision"},
        "calibration": {
            "dataset": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train",
            "num_sequences": 128,
            "sequence_length": 512,
            "sampling": "seeded_shuffle",
            "seed": 42,
        },
        "evaluation": {
            "dataset": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "test",
            "sequence_length": 512,
            "stride": 512,
            "metrics": ["perplexity", "latency_ms_per_token", "memory_total_bytes"],
        },
        "profiling": {
            "layerwise_weight_error": False,
            "layerwise_activation_error": False,
            "residual_svd_profile": False,
        },
        "outputs": {
            "results_dir": "results/downstream/DS_FP_Q17B",
            "metrics_file": "metrics.json",
            "layer_summary_file": "layer_errors.json",
            "residual_profile_file": "residual_profiles.json",
        },
    },
    # Existing configs to augment
    ("DS_R3_Q17B", "configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_baselines_manifest.json", "R3_Q17B"),
    ("DS_G2B03_Q17B", "configs/scaleup_1p7b_gptq/p2b03_qwen3_1p7b_gptq_bits_activation_2p0pct.json", None),
    ("DS_G2R02_Q17B", "configs/scaleup_1p7b_gptq/p2r02_qwen3_1p7b_gptq_rank_activation_1p0pct.json", None),
]

QWEN3_8B_GPTQ_RUNS = [
    # Full precision
    {
        "run_id": "DS_FP_Q8B",
        "phase": "downstream_qwen3_8b",
        "description": "Full-precision Qwen3-8B downstream evaluation.",
        "model": {
            "name": "Qwen/Qwen3-8B-Base",
            "tokenizer_name": "Qwen/Qwen3-8B-Base",
            "dtype_preference": ["float16"],
        },
        "method": {"name": "full_precision"},
        "calibration": {
            "dataset": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train",
            "num_sequences": 128,
            "sequence_length": 512,
            "sampling": "seeded_shuffle",
            "seed": 42,
        },
        "evaluation": {
            "dataset": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "test",
            "sequence_length": 512,
            "stride": 512,
            "metrics": ["perplexity", "latency_ms_per_token", "memory_total_bytes"],
        },
        "profiling": {
            "layerwise_weight_error": False,
            "layerwise_activation_error": False,
            "residual_svd_profile": False,
        },
        "outputs": {
            "results_dir": "results/downstream/DS_FP_Q8B",
            "metrics_file": "metrics.json",
            "layer_summary_file": "layer_errors.json",
            "residual_profile_file": "residual_profiles.json",
        },
    },
]

QWEN3_1P7B_RTN_RUNS = [
    ("DS_R2_Q17B", "configs/scaleup_1p7b/qwen3_1p7b_baselines_manifest.json", "R2_Q17B"),
    ("DS_P2B03_Q17B", "configs/scaleup_1p7b/p2b03_qwen3_1p7b_bits_activation_2p0pct.json", None),
    ("DS_P2R02_Q17B", "configs/scaleup_1p7b/p2r02_qwen3_1p7b_rank_activation_1p0pct.json", None),
]

SMOLLM3_3B_GPTQ_RUNS = [
    # Full precision
    {
        "run_id": "DS_FP_S3B",
        "phase": "downstream_smollm3_3b",
        "description": "Full-precision SmolLM3-3B downstream evaluation.",
        "model": {
            "name": "HuggingFaceTB/SmolLM3-3B-Base",
            "tokenizer_name": "HuggingFaceTB/SmolLM3-3B-Base",
            "dtype_preference": ["float16"],
        },
        "method": {"name": "full_precision"},
        "calibration": {
            "dataset": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "train",
            "num_sequences": 128,
            "sequence_length": 512,
            "sampling": "seeded_shuffle",
            "seed": 42,
        },
        "evaluation": {
            "dataset": "wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "test",
            "sequence_length": 512,
            "stride": 512,
            "metrics": ["perplexity", "latency_ms_per_token", "memory_total_bytes"],
        },
        "profiling": {
            "layerwise_weight_error": False,
            "layerwise_activation_error": False,
            "residual_svd_profile": False,
        },
        "outputs": {
            "results_dir": "results/downstream/DS_FP_S3B",
            "metrics_file": "metrics.json",
            "layer_summary_file": "layer_errors.json",
            "residual_profile_file": "residual_profiles.json",
        },
    },
]


def _augment_config(source_path: Path, new_run_id: str, phase: str) -> dict:
    """Read an existing config, inject downstream section, update IDs."""
    config = json.loads(source_path.read_text(encoding="utf-8"))
    config["run_id"] = new_run_id
    config["phase"] = phase
    config["downstream"] = dict(DOWNSTREAM_SECTION)
    config["outputs"]["results_dir"] = f"results/downstream/{new_run_id}"
    # Disable expensive profiling for downstream-only runs.
    config["profiling"]["layerwise_weight_error"] = False
    config["profiling"]["layerwise_activation_error"] = False
    config["profiling"]["residual_svd_profile"] = False
    config["profiling"]["profile_num_sequences"] = 0
    config["profiling"]["residual_top_k_layers"] = 0
    return config


def _find_config_by_run_id(manifest_path: Path, run_id: str) -> Path:
    """Look up a run config path from a manifest by run_id."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel in manifest["runs"]:
        candidate = Path(rel)
        cfg_path = candidate if candidate.exists() else manifest_path.parent / candidate.name
        if not cfg_path.exists():
            cfg_path = REPO_ROOT / rel
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg["run_id"] == run_id:
            return cfg_path
    raise ValueError(f"Run id {run_id!r} not found in {manifest_path}")


def _build_full_precision_config(spec: dict) -> dict:
    """Build a full-precision config from a direct spec dict."""
    config = dict(spec)
    config["downstream"] = dict(DOWNSTREAM_SECTION)
    return config


def generate_group(
    group_name: str,
    phase: str,
    runs: list,
    output_dir: Path,
) -> list[str]:
    """Generate configs for one model group and return relative config paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_paths: list[str] = []

    for entry in runs:
        if isinstance(entry, dict):
            # Full precision or direct spec
            config = _build_full_precision_config(entry)
            config["phase"] = phase
            filename = f"{config['run_id'].lower()}.json"
        elif isinstance(entry, tuple) and len(entry) == 3:
            new_run_id, source_ref, manifest_run_id = entry
            if manifest_run_id is not None:
                # Look up in manifest
                manifest_path = REPO_ROOT / source_ref
                source_path = _find_config_by_run_id(manifest_path, manifest_run_id)
            else:
                source_path = REPO_ROOT / source_ref
            config = _augment_config(source_path, new_run_id, phase)
            filename = f"{new_run_id.lower()}.json"
        else:
            raise ValueError(f"Unexpected entry format: {entry}")

        config_path = output_dir / filename
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        config_paths.append(str(config_path.relative_to(REPO_ROOT)))
        print(f"  wrote {config_path.relative_to(REPO_ROOT)}")

    return config_paths


def write_manifest(
    manifest_path: Path,
    phase: str,
    description: str,
    config_paths: list[str],
) -> None:
    manifest = {
        "phase": phase,
        "description": description,
        "runs": config_paths,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote manifest {manifest_path.relative_to(REPO_ROOT)}")


def main() -> None:
    ds_dir = REPO_ROOT / "configs" / "downstream"

    # --- Qwen3-1.7B GPTQ ---
    print("Generating Qwen3-1.7B GPTQ downstream configs...")

    # Need to find configs for runs referenced from manifests
    # First, find configs for 8B and 3B runs
    q8b_manifest_path = REPO_ROOT / "configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_transfer_manifest.json"
    s3b_manifest_path = REPO_ROOT / "configs/scaleup_smollm3_3b_gptq/smollm3_3b_gptq_transfer_manifest.json"

    # Add 8B transfer runs
    q8b_baselines_manifest_path = REPO_ROOT / "configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_baselines_manifest.json"
    q8b_additional = [
        ("DS_R3_Q8B", "configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_baselines_manifest.json", "R3_Q8B"),
        ("DS_G2B02_Q8B", str(q8b_manifest_path.relative_to(REPO_ROOT)), "G2B02_Q8B"),
        ("DS_G2R02_Q8B", str(q8b_manifest_path.relative_to(REPO_ROOT)), "G2R02_Q8B"),
    ]
    QWEN3_8B_GPTQ_RUNS.extend(q8b_additional)

    # Add 3B transfer runs
    s3b_additional = [
        ("DS_R3_S3B", "configs/scaleup_smollm3_3b_gptq/smollm3_3b_gptq_baselines_manifest.json", "R3_S3B"),
        ("DS_G3B02_S3B", str(s3b_manifest_path.relative_to(REPO_ROOT)), "G3B02_S3B"),
        ("DS_G3R02_S3B", str(s3b_manifest_path.relative_to(REPO_ROOT)), "G3R02_S3B"),
    ]
    SMOLLM3_3B_GPTQ_RUNS.extend(s3b_additional)

    # Also look up the hybrid configs for 1.7B and 8B
    q17b_hybrid_manifest = REPO_ROOT / "configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_matrix_hybrid_manifest.json"
    q8b_hybrid_manifest = REPO_ROOT / "configs/scaleup_qwen3_8b_gptq/qwen3_8b_gptq_hybrid_manifest.json"

    if q17b_hybrid_manifest.exists():
        QWEN3_1P7B_GPTQ_RUNS.append(
            ("DS_H2R02M_Q17B", str(q17b_hybrid_manifest.relative_to(REPO_ROOT)), "H2R02M_Q17B")
        )
    if q8b_hybrid_manifest.exists():
        QWEN3_8B_GPTQ_RUNS.append(
            ("DS_H2R02_Q8B", str(q8b_hybrid_manifest.relative_to(REPO_ROOT)), "H2R02_Q8B")
        )

    paths_1p7b = generate_group(
        "qwen3-1.7b-gptq",
        "downstream_qwen3_1p7b",
        QWEN3_1P7B_GPTQ_RUNS,
        ds_dir / "qwen3_1p7b",
    )
    write_manifest(
        ds_dir / "qwen3_1p7b_downstream_manifest.json",
        "downstream_qwen3_1p7b",
        "Downstream zero-shot evaluation for Qwen3-1.7B GPTQ regime points.",
        paths_1p7b,
    )

    print("\nGenerating Qwen3-8B GPTQ downstream configs...")
    paths_8b = generate_group(
        "qwen3-8b-gptq",
        "downstream_qwen3_8b",
        QWEN3_8B_GPTQ_RUNS,
        ds_dir / "qwen3_8b",
    )
    write_manifest(
        ds_dir / "qwen3_8b_downstream_manifest.json",
        "downstream_qwen3_8b",
        "Downstream zero-shot evaluation for Qwen3-8B GPTQ regime points.",
        paths_8b,
    )

    print("\nGenerating SmolLM3-3B GPTQ downstream configs...")
    paths_3b = generate_group(
        "smollm3-3b-gptq",
        "downstream_smollm3_3b",
        SMOLLM3_3B_GPTQ_RUNS,
        ds_dir / "smollm3_3b",
    )
    write_manifest(
        ds_dir / "smollm3_3b_downstream_manifest.json",
        "downstream_smollm3_3b",
        "Downstream zero-shot evaluation for SmolLM3-3B GPTQ regime points.",
        paths_3b,
    )

    print("\nGenerating Qwen3-1.7B RTN downstream configs...")
    paths_rtn = generate_group(
        "qwen3-1.7b-rtn",
        "downstream_qwen3_1p7b_rtn",
        QWEN3_1P7B_RTN_RUNS,
        ds_dir / "qwen3_1p7b_rtn",
    )
    write_manifest(
        ds_dir / "qwen3_1p7b_rtn_downstream_manifest.json",
        "downstream_qwen3_1p7b_rtn",
        "Downstream zero-shot evaluation for Qwen3-1.7B RTN cross-quantizer anchor runs.",
        paths_rtn,
    )

    print(
        f"\nDone. Generated "
        f"{len(paths_1p7b) + len(paths_8b) + len(paths_3b) + len(paths_rtn)} configs total."
    )


if __name__ == "__main__":
    main()
