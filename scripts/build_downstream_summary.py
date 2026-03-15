#!/usr/bin/env python3
"""Build consolidated Item 1 downstream summaries from completed Modal runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "analysis"
REPORT_PATH = ROOT / "docs" / "experiments" / "downstream_item1_analysis.md"

TASK_METRICS = {
    "hellaswag": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "winogrande": "acc,none",
    "piqa": "acc_norm,none",
    "boolq": "acc,none",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    path: str
    model_label: str
    quantizer: str
    policy: str
    family: str
    is_full_precision: bool = False
    is_baseline: bool = False


RUN_SPECS: List[RunSpec] = [
    RunSpec("DS_FP_Q17B", "results/modal/downstream_qwen3_1p7b/DS_FP_Q17B", "Qwen3-1.7B", "GPTQ", "full_precision", "gptq_q17b", True, False),
    RunSpec("DS_R3_Q17B", "results/modal/downstream_qwen3_1p7b/DS_R3_Q17B", "Qwen3-1.7B", "GPTQ", "baseline_4bit", "gptq_q17b", False, True),
    RunSpec("DS_G2B03_Q17B", "results/modal/downstream_qwen3_1p7b/DS_G2B03_Q17B", "Qwen3-1.7B", "GPTQ", "bits", "gptq_q17b"),
    RunSpec("DS_G2R02_Q17B", "results/modal/downstream_qwen3_1p7b/DS_G2R02_Q17B", "Qwen3-1.7B", "GPTQ", "rank", "gptq_q17b"),
    RunSpec("DS_H2R02M_Q17B", "results/modal/downstream_qwen3_1p7b/DS_H2R02M_Q17B", "Qwen3-1.7B", "GPTQ", "hybrid", "gptq_q17b"),
    RunSpec("DS_FP_Q8B", "results/modal/downstream_qwen3_8b/DS_FP_Q8B", "Qwen3-8B", "GPTQ", "full_precision", "gptq_q8b", True, False),
    RunSpec("DS_R3_Q8B", "results/modal/downstream_qwen3_8b/DS_R3_Q8B", "Qwen3-8B", "GPTQ", "baseline_4bit", "gptq_q8b", False, True),
    RunSpec("DS_G2B02_Q8B", "results/modal/downstream_qwen3_8b/DS_G2B02_Q8B", "Qwen3-8B", "GPTQ", "bits", "gptq_q8b"),
    RunSpec("DS_G2R02_Q8B", "results/modal/downstream_qwen3_8b/DS_G2R02_Q8B", "Qwen3-8B", "GPTQ", "rank", "gptq_q8b"),
    RunSpec("DS_H2R02_Q8B", "results/modal/downstream_qwen3_8b/DS_H2R02_Q8B", "Qwen3-8B", "GPTQ", "hybrid", "gptq_q8b"),
    RunSpec("DS_FP_S3B", "results/modal/downstream_smollm3_3b/DS_FP_S3B", "SmolLM3-3B", "GPTQ", "full_precision", "gptq_s3b", True, False),
    RunSpec("DS_R3_S3B", "results/modal/downstream_smollm3_3b/DS_R3_S3B", "SmolLM3-3B", "GPTQ", "baseline_4bit", "gptq_s3b", False, True),
    RunSpec("DS_G3B02_S3B", "results/modal/downstream_smollm3_3b/DS_G3B02_S3B", "SmolLM3-3B", "GPTQ", "bits", "gptq_s3b"),
    RunSpec("DS_G3R02_S3B", "results/modal/downstream_smollm3_3b/DS_G3R02_S3B", "SmolLM3-3B", "GPTQ", "rank", "gptq_s3b"),
    RunSpec("DS_R2_Q17B", "results/modal/downstream_qwen3_1p7b_rtn/DS_R2_Q17B", "Qwen3-1.7B", "RTN", "baseline_4bit", "rtn_q17b", False, True),
    RunSpec("DS_P2B03_Q17B", "results/modal/downstream_qwen3_1p7b_rtn/DS_P2B03_Q17B", "Qwen3-1.7B", "RTN", "bits", "rtn_q17b"),
    RunSpec("DS_P2R02_Q17B", "results/modal/downstream_qwen3_1p7b_rtn/DS_P2R02_Q17B", "Qwen3-1.7B", "RTN", "rank", "rtn_q17b"),
]

GROUP_ORDER = ["gptq_q17b", "gptq_q8b", "gptq_s3b", "rtn_q17b"]
GROUP_LABELS = {
    "gptq_q17b": "GPTQ / Qwen3-1.7B",
    "gptq_q8b": "GPTQ / Qwen3-8B",
    "gptq_s3b": "GPTQ / SmolLM3-3B",
    "rtn_q17b": "RTN / Qwen3-1.7B",
}
FULL_PRECISION_REFERENCE = {
    "gptq_q17b": "DS_FP_Q17B",
    "gptq_q8b": "DS_FP_Q8B",
    "gptq_s3b": "DS_FP_S3B",
    "rtn_q17b": "DS_FP_Q17B",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def format_signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def md_table(headers: List[str], rows: Iterable[Iterable[str]]) -> str:
    rows = list(rows)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def avg_task_score(task_scores: Dict[str, float]) -> float:
    return mean(task_scores.values())


def compute_pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(np.array(xs), np.array(ys))[0, 1])


def build_rows() -> List[dict]:
    rows: List[dict] = []
    for spec in RUN_SPECS:
        root = ROOT / spec.path
        metrics = load_json(root / "metrics.json")
        downstream = load_json(root / "downstream_metrics.json")
        task_scores = {
            task: downstream["results"][task][metric_name]
            for task, metric_name in TASK_METRICS.items()
        }
        row = {
            "run_id": spec.run_id,
            "path": spec.path,
            "model_label": spec.model_label,
            "quantizer": spec.quantizer,
            "policy": spec.policy,
            "family": spec.family,
            "is_full_precision": spec.is_full_precision,
            "is_baseline": spec.is_baseline,
            "perplexity": float(metrics["perplexity"]),
            "memory_mb": float(metrics["memory_total_bytes"]) / (1024 * 1024),
            "latency_ms_per_token": float(metrics["latency_ms_per_token"]),
            "avg_downstream_score": avg_task_score(task_scores),
            "task_scores": task_scores,
        }
        rows.append(row)
    return rows


def add_group_deltas(rows: List[dict]) -> List[dict]:
    by_family: Dict[str, List[dict]] = {}
    by_run_id: Dict[str, dict] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
        by_run_id[row["run_id"]] = row

    delta_rows: List[dict] = []
    for family in GROUP_ORDER:
        group_rows = by_family[family]
        full_precision = by_run_id[FULL_PRECISION_REFERENCE[family]]
        baseline = next(row for row in group_rows if row["is_baseline"])
        fp_gap = full_precision["avg_downstream_score"] - baseline["avg_downstream_score"]

        for row in group_rows:
            row["delta_ppl_vs_baseline"] = baseline["perplexity"] - row["perplexity"]
            row["delta_avg_score_vs_baseline"] = row["avg_downstream_score"] - baseline["avg_downstream_score"]
            row["added_mb_vs_baseline"] = row["memory_mb"] - baseline["memory_mb"]
            if row["is_baseline"]:
                row["quality_recovered_fraction"] = 0.0
                row["quality_recovered_per_mb"] = 0.0
                row["avg_score_delta_per_mb"] = 0.0
            else:
                recovered = 0.0
                if fp_gap != 0:
                    recovered = row["delta_avg_score_vs_baseline"] / fp_gap
                row["quality_recovered_fraction"] = recovered
                added_mb = row["added_mb_vs_baseline"]
                if abs(added_mb) > 1e-12:
                    row["quality_recovered_per_mb"] = recovered / added_mb
                    row["avg_score_delta_per_mb"] = row["delta_avg_score_vs_baseline"] / added_mb
                else:
                    row["quality_recovered_per_mb"] = 0.0
                    row["avg_score_delta_per_mb"] = 0.0

            delta_rows.append(
                {
                    "family": family,
                    "group_label": GROUP_LABELS[family],
                    "run_id": row["run_id"],
                    "policy": row["policy"],
                    "is_full_precision": row["is_full_precision"],
                    "is_baseline": row["is_baseline"],
                    "perplexity": row["perplexity"],
                    "delta_ppl_vs_baseline": row["delta_ppl_vs_baseline"],
                    "avg_downstream_score": row["avg_downstream_score"],
                    "delta_avg_score_vs_baseline": row["delta_avg_score_vs_baseline"],
                    "memory_mb": row["memory_mb"],
                    "added_mb_vs_baseline": row["added_mb_vs_baseline"],
                    "quality_recovered_fraction": row["quality_recovered_fraction"],
                    "avg_score_delta_per_mb": row["avg_score_delta_per_mb"],
                    "quality_recovered_per_mb": row["quality_recovered_per_mb"],
                }
            )
    return delta_rows


def write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def task_winner_summary(rows: List[dict]) -> Dict[str, dict]:
    by_family: Dict[str, List[dict]] = {}
    for row in rows:
        if row["policy"] == "full_precision":
            continue
        by_family.setdefault(row["family"], []).append(row)

    summaries: Dict[str, dict] = {}
    for family in GROUP_ORDER:
        group_rows = by_family[family]
        wins = {row["run_id"]: 0 for row in group_rows}
        winners = {}
        for task in TASK_METRICS:
            best = max(group_rows, key=lambda row: row["task_scores"][task])
            winners[task] = best["run_id"]
            wins[best["run_id"]] += 1
        summaries[family] = {"wins": wins, "task_winners": winners}
    return summaries


def build_report(rows: List[dict], delta_rows: List[dict], task_winners: Dict[str, dict]) -> str:
    by_family: Dict[str, List[dict]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    group_tables = []
    for family in GROUP_ORDER:
        group_rows = sorted(
            by_family[family],
            key=lambda row: (
                0 if row["policy"] == "full_precision" else
                1 if row["policy"] == "baseline_4bit" else
                2 if row["policy"] == "bits" else
                3 if row["policy"] == "rank" else
                4
            )
        )
        table_rows = []
        for row in group_rows:
            table_rows.append([
                row["policy"],
                row["run_id"],
                format_float(row["perplexity"]),
                format_float(row["memory_mb"], 1),
                format_float(row["avg_downstream_score"]),
                format_float(row["latency_ms_per_token"]),
            ])
        group_tables.append(
            f"### {GROUP_LABELS[family]}\n\n" +
            md_table(
                ["policy", "run_id", "perplexity", "memory_mb", "avg_downstream", "latency_ms/token"],
                table_rows,
            )
        )

    delta_subset = [
        row for row in delta_rows
        if row["policy"] not in {"baseline_4bit", "full_precision"}
    ]
    quality_rows = []
    for family in GROUP_ORDER:
        for row in [r for r in delta_subset if r["family"] == family]:
            quality_rows.append([
                GROUP_LABELS[family],
                row["policy"],
                row["run_id"],
                format_signed(row["delta_ppl_vs_baseline"]),
                format_signed(row["delta_avg_score_vs_baseline"]),
                format_float(row["added_mb_vs_baseline"], 2),
                format_signed(row["quality_recovered_fraction"]),
                format_signed(row["quality_recovered_per_mb"], 5),
            ])

    all_points = [row for row in delta_rows if not row["is_baseline"]]
    policy_only = [row for row in delta_rows if row["policy"] not in {"baseline_4bit", "full_precision"}]
    all_corr = compute_pearson(
        [row["delta_ppl_vs_baseline"] for row in all_points],
        [row["delta_avg_score_vs_baseline"] for row in all_points],
    )
    policy_corr = compute_pearson(
        [row["delta_ppl_vs_baseline"] for row in policy_only],
        [row["delta_avg_score_vs_baseline"] for row in policy_only],
    )

    win_rows = []
    for family in GROUP_ORDER:
        summary = task_winner_summary(rows)[family]["wins"]
        for run_id, count in summary.items():
            win_rows.append([GROUP_LABELS[family], run_id, count])

    best_downstream_rows = []
    for family in GROUP_ORDER:
        group_rows = [row for row in rows if row["family"] == family and not row["is_full_precision"]]
        best = max(group_rows, key=lambda row: row["avg_downstream_score"])
        best_downstream_rows.append([
            GROUP_LABELS[family],
            best["run_id"],
            best["policy"],
            format_float(best["avg_downstream_score"]),
            format_float(best["perplexity"]),
        ])

    return f"""# Item 1 Downstream Analysis

This report aggregates the completed downstream evaluation matrix for the paper-readiness plan:

- GPTQ `Qwen3-1.7B`
- GPTQ `SmolLM3-3B`
- GPTQ `Qwen3-8B`
- RTN `Qwen3-1.7B` anchor

## Metric Definitions

For each run we compute:

- `avg_downstream`: the arithmetic mean of six task metrics
- task metric choice:
  - `hellaswag`, `arc_easy`, `arc_challenge`, `piqa`: `acc_norm`
  - `winogrande`, `boolq`: `acc`

For each quantizer/scale family, deltas are measured relative to the family baseline:

```text
ΔPPL = PPL_baseline - PPL_run
Δavg = avg_downstream_run - avg_downstream_baseline
quality_recovered_fraction = Δavg / (avg_downstream_full_precision - avg_downstream_baseline)
quality_recovered_per_mb = quality_recovered_fraction / added_mb
```

## Consolidated Run Table

{chr(10).join(group_tables)}

## Cross-Run Correlation

- Pearson correlation between `ΔPPL` and `Δavg` when full-precision references are included: `{format_float(all_corr)}`
- Pearson correlation between `ΔPPL` and `Δavg` for compressed policies only: `{format_float(policy_corr)}`

Interpretation:

- Full-precision anchors preserve a strong positive global trend: better perplexity generally corresponds to better downstream score when the comparison spans the entire quality range.
- Inside the compressed-policy regime, the relationship is weak and unstable. That is the paper-relevant result: small perplexity wins do not reliably transfer into uniformly better downstream task performance.

## Quality Recovered Per Added MB

{md_table(
    ["family", "policy", "run_id", "ΔPPL", "Δavg", "added_mb", "recovered_frac", "recovered_frac_per_mb"],
    quality_rows,
)}

## Task-Win Counts Within Each Family

{md_table(["family", "run_id", "task_wins"], win_rows)}

## Best Downstream Policy By Family

{md_table(["family", "best_run", "policy", "avg_downstream", "perplexity"], best_downstream_rows)}

## Conclusions

1. The downstream branch does not collapse the project into a universal winner.
2. Perplexity remains useful as a coarse global quality measure, but it is not sufficient to rank nearby compressed policies.
3. `Qwen3-1.7B` preserves the cross-quantizer contrast:
   - GPTQ: rank is best by perplexity, but bits wins the most task-level comparisons and the highest mean downstream score.
   - RTN: bits is best by perplexity and also wins the most task-level comparisons.
4. `SmolLM3-3B` remains the neutral midpoint:
   - baseline still has the best perplexity
   - bits slightly improves the mean downstream score over baseline
   - rank remains the weakest policy
5. `Qwen3-8B` stays bits-favoring by perplexity, but downstream is essentially tied between baseline, rank, and hybrid at the current budget slice.

These results are strong enough to mark Item 1 complete: the paper can now claim that the regime map holds beyond perplexity, but with a more nuanced downstream interpretation than the perplexity frontier alone.
"""


def main() -> int:
    rows = build_rows()
    delta_rows = add_group_deltas(rows)
    winners = task_winner_summary(rows)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    flat_rows = []
    for row in rows:
        base = {k: v for k, v in row.items() if k != "task_scores"}
        for task, score in row["task_scores"].items():
            base[f"task_{task}"] = score
        flat_rows.append(base)

    write_csv(
        ANALYSIS_DIR / "downstream_run_summary.csv",
        flat_rows,
        [
            "run_id",
            "path",
            "model_label",
            "quantizer",
            "policy",
            "family",
            "is_full_precision",
            "is_baseline",
            "perplexity",
            "memory_mb",
            "latency_ms_per_token",
            "avg_downstream_score",
            "delta_ppl_vs_baseline",
            "delta_avg_score_vs_baseline",
            "added_mb_vs_baseline",
            "quality_recovered_fraction",
            "avg_score_delta_per_mb",
            "quality_recovered_per_mb",
            "task_hellaswag",
            "task_arc_easy",
            "task_arc_challenge",
            "task_winogrande",
            "task_piqa",
            "task_boolq",
        ],
    )

    write_csv(
        ANALYSIS_DIR / "downstream_group_deltas.csv",
        delta_rows,
        [
            "family",
            "group_label",
            "run_id",
            "policy",
            "perplexity",
            "delta_ppl_vs_baseline",
            "avg_downstream_score",
            "delta_avg_score_vs_baseline",
            "memory_mb",
            "added_mb_vs_baseline",
            "quality_recovered_fraction",
            "avg_score_delta_per_mb",
            "quality_recovered_per_mb",
        ],
    )

    summary_payload = {
        "task_metrics": TASK_METRICS,
        "task_winners": winners,
        "generated_files": [
            str((ANALYSIS_DIR / "downstream_run_summary.csv").relative_to(ROOT)),
            str((ANALYSIS_DIR / "downstream_group_deltas.csv").relative_to(ROOT)),
            str(REPORT_PATH.relative_to(ROOT)),
        ],
    }
    (ANALYSIS_DIR / "downstream_item1_summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n"
    )

    REPORT_PATH.write_text(build_report(rows, delta_rows, winners))
    print(f"Wrote {(ANALYSIS_DIR / 'downstream_run_summary.csv').relative_to(ROOT)}")
    print(f"Wrote {(ANALYSIS_DIR / 'downstream_group_deltas.csv').relative_to(ROOT)}")
    print(f"Wrote {(ANALYSIS_DIR / 'downstream_item1_summary.json').relative_to(ROOT)}")
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
