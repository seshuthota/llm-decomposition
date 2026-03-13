#!/usr/bin/env python3
"""Summarize phase-1 metrics into markdown and JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path


RUN_IDS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    rows = []
    for run_id in RUN_IDS:
        metrics_path = repo_root / f"results/phase1/{run_id}/metrics.json"
        metrics = load_metrics(metrics_path)
        if metrics is None:
            rows.append(
                {
                    "run_id": run_id,
                    "method": "missing",
                    "status": "missing",
                    "memory_total_bytes": None,
                    "perplexity": None,
                    "latency_ms_per_token": None,
                }
            )
            continue
        rows.append(
            {
                "run_id": run_id,
                "method": metrics.get("method"),
                "status": metrics.get("status"),
                "memory_total_bytes": metrics.get("memory_total_bytes"),
                "perplexity": metrics.get("perplexity"),
                "latency_ms_per_token": metrics.get("latency_ms_per_token"),
            }
        )

    markdown_lines = [
        "# Phase 1 Summary",
        "",
        "| Run | Method | Status | Memory (bytes) | Perplexity | Latency ms/token |",
        "|-----|--------|--------|----------------|------------|------------------|",
    ]
    for row in rows:
        markdown_lines.append(
            "| {run_id} | {method} | {status} | {memory_total_bytes} | {perplexity} | {latency_ms_per_token} |".format(
                run_id=row["run_id"],
                method=row["method"],
                status=row["status"],
                memory_total_bytes=row["memory_total_bytes"],
                perplexity=row["perplexity"],
                latency_ms_per_token=row["latency_ms_per_token"],
            )
        )
    markdown_lines.append("")

    summary_dir = repo_root / "results/phase1"
    summary_json_path = summary_dir / "phase1_summary.json"
    summary_md_path = summary_dir / "phase1_summary.md"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump({"runs": rows}, handle, indent=2)
        handle.write("\n")
    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines))

    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
