#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Phase-2 candidate pool from layerwise activation error.")
    parser.add_argument("--layer-errors", required=True, help="Path to a layer_errors.json file.")
    parser.add_argument("--output", required=True, help="Path to write the candidate pool JSON.")
    parser.add_argument("--model-name", required=True, help="Model name for metadata.")
    parser.add_argument("--base-run-id", required=True, help="Anchor baseline run id.")
    parser.add_argument("--source-layer-errors-path", required=True, help="Repo-relative source layer error path for metadata.")
    parser.add_argument("--pool-name", required=True, help="Candidate pool name.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of top damaged layers to include.")
    parser.add_argument("--quantizer", default="RTN", help="Quantizer label for metadata.")
    parser.add_argument("--bit-width", type=int, default=4, help="Base quantizer bit width for metadata.")
    parser.add_argument("--group-size", type=int, default=128, help="Group size for metadata.")
    parser.add_argument("--symmetric", choices=["true", "false"], default="true", help="Whether quantization is symmetric.")
    parser.add_argument(
        "--control-layer",
        action="append",
        default=[],
        help="Optional control layer name. May be provided multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layer_errors_path = Path(args.layer_errors)
    payload = json.loads(layer_errors_path.read_text(encoding="utf-8"))
    rows = list(payload.get("layer_errors", []))
    rows.sort(
        key=lambda row: row["activation_relative_l2"] if "activation_relative_l2" in row else row.get("relative_fro_error", 0.0),
        reverse=True,
    )

    candidate_layers: list[str] = []
    for row in rows:
        layer_name = row["layer_name"]
        if layer_name in candidate_layers:
            continue
        candidate_layers.append(layer_name)
        if len(candidate_layers) >= args.top_k:
            break

    control_layers = []
    for layer_name in args.control_layer:
        if layer_name not in candidate_layers and layer_name not in control_layers:
            control_layers.append(layer_name)

    output = {
        "pool_name": args.pool_name,
        "description": (
            f"Top-{args.top_k} candidate pool built from {args.base_run_id} activation-space damage, "
            "plus shared metadata for targeted bits and targeted rank runs."
        ),
        "base_run_id": args.base_run_id,
        "source_layer_errors_path": args.source_layer_errors_path,
        "selection_rule": f"top_{args.top_k}_by_activation_relative_l2",
        "model_name": args.model_name,
        "quantizer": args.quantizer,
        "bit_width": args.bit_width,
        "group_size": args.group_size,
        "symmetric": args.symmetric == "true",
        "factor_dtype_bytes": 2,
        "candidate_layers": candidate_layers,
        "control_layers": control_layers,
        "budget_schedule_percent_of_r2": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        "bit_actions": {
            "target_granularity": "matrix",
            "candidate_bit_widths": [8],
            "future_candidate_bit_widths": [5, 6],
        },
        "rank_actions": {
            "target_granularity": "matrix",
            "candidate_ranks": [4, 8, 16, 32],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(output_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
