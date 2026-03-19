from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "modal" / "qwen3_1p7b_gptq_proxy_ablation"
ANALYSIS_ROOT = ROOT / "results" / "analysis"

RUNS = {
    "G2B02A_Q17B": {"policy": "bits", "proxy_family": "activation"},
    "G2B02W_Q17B": {"policy": "bits", "proxy_family": "weight"},
    "G2R02A_Q17B": {"policy": "rank", "proxy_family": "activation"},
    "G2R02W_Q17B": {"policy": "rank", "proxy_family": "weight"},
}


def _load_metrics(run_id: str) -> dict:
    return json.loads((RUN_ROOT / run_id / "metrics.json").read_text())


def _selected_targets(metrics: dict) -> list[str]:
    selected = metrics.get("selected_actions", [])
    return [action["target_name"] for action in selected]


def _selected_action_labels(metrics: dict) -> list[str]:
    selected = metrics.get("selected_actions", [])
    labels = []
    for action in selected:
        value = action.get("bit_to", action.get("rank_to"))
        labels.append(f"{action['target_name']}->{value}")
    return labels


def _write_csv(rows: list[dict]) -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_ROOT / "proxy_ablation_q17b_summary.csv"
    fieldnames = [
        "run_id",
        "policy",
        "proxy_family",
        "perplexity",
        "memory_total_bytes",
        "latency_ms_per_token",
        "profiling_wall_time_s",
        "selection_profiling_wall_time_s",
        "selected_action_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(payload: dict) -> None:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_ROOT / "proxy_ablation_q17b_selection_diff.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    rows: list[dict] = []
    loaded: dict[str, dict] = {}
    for run_id, meta in RUNS.items():
        metrics = _load_metrics(run_id)
        loaded[run_id] = metrics
        rows.append(
            {
                "run_id": run_id,
                "policy": meta["policy"],
                "proxy_family": meta["proxy_family"],
                "perplexity": metrics["perplexity"],
                "memory_total_bytes": metrics["memory_total_bytes"],
                "latency_ms_per_token": metrics["latency_ms_per_token"],
                "profiling_wall_time_s": metrics.get("profiling_wall_time_s", 0.0),
                "selection_profiling_wall_time_s": metrics.get("selection_profiling_wall_time_s", 0.0),
                "selected_action_count": len(metrics.get("selected_actions", [])),
            }
        )

    bits_a = loaded["G2B02A_Q17B"]
    bits_w = loaded["G2B02W_Q17B"]
    rank_a = loaded["G2R02A_Q17B"]
    rank_w = loaded["G2R02W_Q17B"]

    bits_a_targets = set(_selected_targets(bits_a))
    bits_w_targets = set(_selected_targets(bits_w))
    rank_a_targets = set(_selected_targets(rank_a))
    rank_w_targets = set(_selected_targets(rank_w))

    payload = {
        "bits": {
            "activation_targets": sorted(bits_a_targets),
            "weight_targets": sorted(bits_w_targets),
            "shared_targets": sorted(bits_a_targets & bits_w_targets),
            "activation_only_targets": sorted(bits_a_targets - bits_w_targets),
            "weight_only_targets": sorted(bits_w_targets - bits_a_targets),
            "same_perplexity": bits_a["perplexity"] == bits_w["perplexity"],
            "same_memory_total_bytes": bits_a["memory_total_bytes"] == bits_w["memory_total_bytes"],
        },
        "rank": {
            "activation_final_layer_ranks": rank_a.get("selected_layer_ranks", {}),
            "weight_final_layer_ranks": rank_w.get("selected_layer_ranks", {}),
            "shared_selected_layers": sorted(rank_a_targets & rank_w_targets),
            "activation_only_selected_layers": sorted(rank_a_targets - rank_w_targets),
            "weight_only_selected_layers": sorted(rank_w_targets - rank_a_targets),
            "same_final_layer_ranks": rank_a.get("selected_layer_ranks", {}) == rank_w.get("selected_layer_ranks", {}),
            "activation_first_12_actions": _selected_action_labels(rank_a)[:12],
            "weight_first_12_actions": _selected_action_labels(rank_w)[:12],
        },
    }

    _write_csv(rows)
    _write_json(payload)
    print("Wrote", ANALYSIS_ROOT / "proxy_ablation_q17b_summary.csv")
    print("Wrote", ANALYSIS_ROOT / "proxy_ablation_q17b_selection_diff.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
