from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "analysis"
ASSETS_DIR = ROOT / "docs" / "experiments" / "assets"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _policy_label(policy_type: str) -> str:
    if policy_type == "baseline_4bit":
        return "baseline"
    return policy_type


def _model_label(run_id: str) -> str:
    if run_id.endswith("Q17B"):
        return "Qwen3-1.7B"
    if run_id.endswith("Q8B"):
        return "Qwen3-8B"
    raise ValueError(f"Unrecognized run id for latency asset: {run_id}")


def _hardware_label(run_id: str) -> str:
    if run_id.endswith("Q17B"):
        return "A10G"
    if run_id.endswith("Q8B"):
        return "A100"
    raise ValueError(f"Unrecognized run id for latency asset: {run_id}")


def build_item3_errorbar_assets() -> None:
    source_rows = _read_csv(ANALYSIS_DIR / "multiseed_stability_all_summary.csv")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    for row in source_rows:
        grouped[(row["scale"], row["policy"])].append(float(row["perplexity"]))

    rows: list[dict[str, Any]] = []
    scale_order = ["1.7B", "3B", "8B"]
    policy_order = ["bits", "rank"]
    scale_to_x = {scale: idx for idx, scale in enumerate(scale_order)}
    policy_offsets = {"bits": -0.08, "rank": 0.08}
    policy_colors = {"bits": "#1f77b4", "rank": "#d62728"}

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)

    for policy in policy_order:
        xs: list[float] = []
        ys: list[float] = []
        errs: list[float] = []
        for scale in scale_order:
            values = grouped[(scale, policy)]
            mean_ppl = mean(values)
            std_ppl = stdev(values) if len(values) > 1 else 0.0
            rows.append(
                {
                    "scale": scale,
                    "policy": policy,
                    "n": len(values),
                    "mean_ppl": f"{mean_ppl:.6f}",
                    "std_ppl": f"{std_ppl:.6f}",
                    "min_ppl": f"{min(values):.6f}",
                    "max_ppl": f"{max(values):.6f}",
                }
            )
            xs.append(scale_to_x[scale] + policy_offsets[policy])
            ys.append(mean_ppl)
            errs.append(std_ppl)
        ax.errorbar(
            xs,
            ys,
            yerr=errs,
            fmt="o",
            capsize=4,
            color=policy_colors[policy],
            label=policy.title(),
            linewidth=1.8,
            markersize=6,
        )

    ax.set_xticks(range(len(scale_order)))
    ax.set_xticklabels(scale_order)
    ax.set_xlabel("Model Scale")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("GPTQ Multi-Seed Stability")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    _write_csv(
        ASSETS_DIR / "item3_multiseed_errorbar.csv",
        rows,
        ["scale", "policy", "n", "mean_ppl", "std_ppl", "min_ppl", "max_ppl"],
    )
    fig.savefig(ASSETS_DIR / "figure_item3_multiseed_errorbars.png", dpi=220)
    fig.savefig(ASSETS_DIR / "figure_item3_multiseed_errorbars.svg")
    plt.close(fig)


def build_item4_latency_assets() -> None:
    source_rows = _read_csv(ANALYSIS_DIR / "latency_item4_summary.csv")
    summary_rows: list[dict[str, Any]] = []
    grouped_by_model_batch: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)

    for row in source_rows:
        run_id = row["run_id"]
        model = _model_label(run_id)
        batch_size = row["batch_size"]
        policy = _policy_label(row["policy_type"])
        grouped_by_model_batch[(model, batch_size)][policy] = row
        summary_rows.append(
            {
                "model": model,
                "hardware": _hardware_label(run_id),
                "batch_size": batch_size,
                "policy": policy,
                "decode_tokens_per_sec": f"{float(row['decode_tokens_per_sec_mean']):.4f}",
                "decode_ms_per_token": f"{float(row['decode_ms_per_token_mean']):.4f}",
                "first_token_latency_ms": f"{float(row['first_token_latency_ms_mean']):.4f}",
                "peak_vram_mb": f"{float(row['peak_vram_mb_mean']):.2f}",
                "run_id": run_id,
            }
        )

    summary_rows.sort(key=lambda row: (row["model"], int(row["batch_size"]), row["policy"]))
    _write_csv(
        ASSETS_DIR / "item4_latency_table.csv",
        summary_rows,
        [
            "model",
            "hardware",
            "batch_size",
            "policy",
            "decode_tokens_per_sec",
            "decode_ms_per_token",
            "first_token_latency_ms",
            "peak_vram_mb",
            "run_id",
        ],
    )

    overhead_rows: list[dict[str, Any]] = []
    markdown_rows: list[list[str]] = []
    for (model, batch_size), policies in sorted(grouped_by_model_batch.items()):
        baseline = policies["baseline"]
        baseline_ms = float(baseline["decode_ms_per_token_mean"])
        baseline_tps = float(baseline["decode_tokens_per_sec_mean"])
        baseline_vram = float(baseline["peak_vram_mb_mean"])
        for policy in ("baseline", "bits", "rank"):
            row = policies[policy]
            markdown_rows.append(
                [
                    model,
                    _hardware_label(row["run_id"]),
                    batch_size,
                    policy,
                    f"{float(row['decode_tokens_per_sec_mean']):.2f}",
                    f"{float(row['decode_ms_per_token_mean']):.2f}",
                    f"{float(row['peak_vram_mb_mean']):.0f}",
                ]
            )
        for policy in ("bits", "rank"):
            row = policies[policy]
            ms = float(row["decode_ms_per_token_mean"])
            tps = float(row["decode_tokens_per_sec_mean"])
            vram = float(row["peak_vram_mb_mean"])
            overhead_rows.append(
                {
                    "model": model,
                    "hardware": _hardware_label(row["run_id"]),
                    "batch_size": batch_size,
                    "policy": policy,
                    "decode_ms_overhead_pct_vs_baseline": f"{((ms / baseline_ms) - 1.0) * 100.0:.2f}",
                    "decode_tps_delta_pct_vs_baseline": f"{((tps / baseline_tps) - 1.0) * 100.0:.2f}",
                    "peak_vram_overhead_pct_vs_baseline": f"{((vram / baseline_vram) - 1.0) * 100.0:.2f}",
                }
            )

    _write_csv(
        ASSETS_DIR / "item4_latency_overheads.csv",
        overhead_rows,
        [
            "model",
            "hardware",
            "batch_size",
            "policy",
            "decode_ms_overhead_pct_vs_baseline",
            "decode_tps_delta_pct_vs_baseline",
            "peak_vram_overhead_pct_vs_baseline",
        ],
    )
    _write_text(
        ASSETS_DIR / "table_item4_latency.md",
        _markdown_table(
            ["Model", "HW", "Batch", "Policy", "Tok/s", "ms/token", "Peak VRAM MB"],
            markdown_rows,
        ),
    )
    _write_text(
        ASSETS_DIR / "table_item4_latency_overheads.md",
        _markdown_table(
            ["Model", "HW", "Batch", "Policy", "ms/token vs base %", "tok/s vs base %", "VRAM vs base %"],
            [
                [
                    row["model"],
                    row["hardware"],
                    row["batch_size"],
                    row["policy"],
                    row["decode_ms_overhead_pct_vs_baseline"],
                    row["decode_tps_delta_pct_vs_baseline"],
                    row["peak_vram_overhead_pct_vs_baseline"],
                ]
                for row in overhead_rows
            ],
        ),
    )


def build_item2_ablation_assets() -> None:
    rows = _read_csv(ANALYSIS_DIR / "proxy_ablation_q17b_summary.csv")
    selection_diff = json.loads((ANALYSIS_DIR / "proxy_ablation_q17b_selection_diff.json").read_text(encoding="utf-8"))
    output_rows: list[dict[str, Any]] = []
    md_rows: list[list[str]] = []

    for row in rows:
        output_rows.append(
            {
                "policy": row["policy"],
                "proxy_family": row["proxy_family"],
                "perplexity": f"{float(row['perplexity']):.4f}",
                "profiling_wall_time_s": f"{float(row['profiling_wall_time_s']):.4f}",
                "selection_profiling_wall_time_s": f"{float(row['selection_profiling_wall_time_s']):.4f}",
                "selected_action_count": row["selected_action_count"],
            }
        )
        md_rows.append(
            [
                row["policy"],
                row["proxy_family"],
                f"{float(row['perplexity']):.4f}",
                f"{float(row['profiling_wall_time_s']):.2f}",
                f"{float(row['selection_profiling_wall_time_s']):.2f}",
                row["selected_action_count"],
            ]
        )

    _write_csv(
        ASSETS_DIR / "item2_activation_weight_ablation.csv",
        output_rows,
        [
            "policy",
            "proxy_family",
            "perplexity",
            "profiling_wall_time_s",
            "selection_profiling_wall_time_s",
            "selected_action_count",
        ],
    )
    notes = [
        "# Activation-vs-Weight Ablation Notes",
        "",
        f"- bits shared target set identical: `{selection_diff['bits']['same_perplexity']}`",
        f"- rank same final layer ranks: `{selection_diff['rank']['same_final_layer_ranks']}`",
        "- interpretation: the quality difference comes from incremental action ordering rather than a different final target set",
        "",
    ]
    _write_text(
        ASSETS_DIR / "table_item2_activation_weight_ablation.md",
        _markdown_table(
            ["Policy", "Proxy", "Perplexity", "Profiling s", "Selection s", "Actions"],
            md_rows,
        )
        + "\n"
        + "\n".join(notes)
        + "\n",
    )


def build_figure1_regime_map_assets() -> None:
    rows = [
        ["RTN", "Qwen3-0.6B", "rank", "n/a", "smallest RTN point favors rank"],
        ["RTN", "Qwen3-1.7B", "bits", "rank (slight)", "cross-quantizer flip vs GPTQ"],
        ["RTN", "SmolLM3-3B", "bits", "n/a", "bits-favoring RTN midpoint"],
        ["RTN", "Qwen3-8B", "bits", "n/a", "bits-favoring RTN large-scale point"],
        ["GPTQ", "Qwen3-1.7B", "rank (single-seed)", "bits", "multiseed says within noise"],
        ["GPTQ", "SmolLM3-3B", "mixed", "bits", "baseline best by PPL, bits best mean downstream"],
        ["GPTQ", "Qwen3-8B", "bits", "rank (near-tied)", "bits stable across seeds; downstream nearly tied"],
    ]
    _write_text(
        ASSETS_DIR / "table_figure1_regime_map_summary.md",
        _markdown_table(
            ["Quantizer", "Model", "Best PPL Policy", "Best Mean Downstream Policy", "Note"],
            rows,
        ),
    )
    _write_csv(
        ASSETS_DIR / "figure1_regime_map_summary.csv",
        [
            {
                "quantizer": quantizer,
                "model": model,
                "best_ppl_policy": best_ppl_policy,
                "best_mean_downstream_policy": best_downstream_policy,
                "note": note,
            }
            for quantizer, model, best_ppl_policy, best_downstream_policy, note in rows
        ],
        ["quantizer", "model", "best_ppl_policy", "best_mean_downstream_policy", "note"],
    )


def main() -> None:
    build_item3_errorbar_assets()
    build_item4_latency_assets()
    build_item2_ablation_assets()
    build_figure1_regime_map_assets()


if __name__ == "__main__":
    main()
