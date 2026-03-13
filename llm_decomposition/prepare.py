from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_decomposition.config import Manifest, RunConfig
from llm_decomposition.io import write_json


def prepare_run(root: Path, config: RunConfig) -> dict[str, Any]:
    outputs = config.raw["outputs"]
    run_dir = root / outputs["results_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "resolved_config.json"
    metrics_template_path = run_dir / outputs.get("metrics_file", "metrics.json")
    layer_template_path = run_dir / outputs.get("layer_summary_file", "layer_errors.json")
    residual_template_path = run_dir / outputs.get("residual_profile_file", "residual_profiles.json")
    notes_path = run_dir / "notes.md"

    write_json(resolved_config_path, config.raw)
    write_json(
        metrics_template_path,
        {
            "run_id": config.run_id,
            "status": "pending",
            "model_name": config.model_name,
            "method": config.method_name,
            "bit_width": config.bit_width,
            "memory_total_bytes": None,
            "perplexity": None,
            "latency_ms_per_token": None,
        },
    )
    write_json(
        layer_template_path,
        {
            "run_id": config.run_id,
            "status": "pending",
            "layer_errors": [],
        },
    )
    write_json(
        residual_template_path,
        {
            "run_id": config.run_id,
            "status": "pending",
            "profiles": [],
        },
    )

    if not notes_path.exists():
        notes_path.write_text(
            "\n".join(
                [
                    f"# {config.run_id} Notes",
                    "",
                    f"- Model: `{config.model_name}`",
                    f"- Method: `{config.method_name}`",
                    f"- Config source: `{config.path.as_posix()}`",
                    "- Status: pending execution",
                    "",
                    "Use this file for run-specific observations or anomalies.",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    return {
        "run_id": config.run_id,
        "results_dir": run_dir.as_posix(),
        "method": config.method_name,
        "bit_width": config.bit_width,
    }


def write_manifest_summary(root: Path, manifest: Manifest, prepared_runs: list[dict[str, Any]]) -> Path:
    summary_path = root / f"results/{manifest.phase}/{manifest.phase}_preparation_summary.json"
    write_json(
        summary_path,
        {
            "phase": manifest.phase,
            "description": manifest.description,
            "prepared_runs": prepared_runs,
        },
    )
    return summary_path
