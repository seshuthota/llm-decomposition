#!/usr/bin/env python3
"""Aggregate latency benchmark JSON artifacts into a paper-friendly CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "results/analysis/latency_item4_summary.csv"


def _iter_benchmark_paths(root: Path) -> list[Path]:
    return sorted(root.glob("results/modal_latency/**/latency_benchmark.json"))


def _summary_value(payload: dict, key: str, stat: str = "mean") -> float | None:
    summary = payload.get("summary", {})
    metric = summary.get(key, {})
    value = metric.get(stat)
    return None if value is None else float(value)


def build_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _iter_benchmark_paths(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec = payload.get("benchmark_spec", {})
        meta = payload.get("policy_metadata", {})

        row = {
            "run_id": payload.get("run_id"),
            "method": payload.get("method"),
            "policy_type": meta.get("policy_type"),
            "device": payload.get("device"),
            "dtype": payload.get("dtype"),
            "batch_size": spec.get("batch_size"),
            "prompt_length": spec.get("prompt_length"),
            "decode_length": spec.get("decode_length"),
            "decode_tokens_per_sec_mean": _summary_value(payload, "decode_tokens_per_sec", "mean"),
            "decode_tokens_per_sec_std": _summary_value(payload, "decode_tokens_per_sec", "std"),
            "decode_ms_per_token_mean": _summary_value(payload, "decode_ms_per_token", "mean"),
            "decode_ms_per_token_std": _summary_value(payload, "decode_ms_per_token", "std"),
            "end_to_end_tokens_per_sec_mean": _summary_value(payload, "end_to_end_tokens_per_sec", "mean"),
            "first_token_latency_ms_mean": _summary_value(payload, "first_token_latency_ms", "mean"),
            "peak_vram_mb_mean": _summary_value(payload, "peak_vram_mb", "mean"),
            "peak_vram_mb_max": _summary_value(payload, "peak_vram_mb", "max"),
            "memory_total_bytes": meta.get("memory_total_bytes"),
            "reconstruction": meta.get("reconstruction"),
            "artifact_path": path.relative_to(root).as_posix(),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "method",
        "policy_type",
        "device",
        "dtype",
        "batch_size",
        "prompt_length",
        "decode_length",
        "decode_tokens_per_sec_mean",
        "decode_tokens_per_sec_std",
        "decode_ms_per_token_mean",
        "decode_ms_per_token_std",
        "end_to_end_tokens_per_sec_mean",
        "first_token_latency_ms_mean",
        "peak_vram_mb_mean",
        "peak_vram_mb_max",
        "memory_total_bytes",
        "reconstruction",
        "artifact_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows = build_rows(REPO_ROOT)
    write_csv(rows, DEFAULT_OUTPUT)
    print(f"Wrote {len(rows)} rows to {DEFAULT_OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
