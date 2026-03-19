#!/usr/bin/env python3
"""Collect multi-seed experiment results and generate summary CSV."""

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def collect_results():
    """Collect all multi-seed results and write summary CSV."""

    results = []

    # Define experiments to check
    experiments = [
        # 1.7B scale
        ("G2R02W_Q17B", "rank", "1.7B"),
        ("G2B02W_Q17B", "bits", "1.7B"),
        # 8B scale
        ("G2R02_Q8B", "rank", "8B"),
        ("G2B02_Q8B", "bits", "8B"),
        # 3B scale
        ("G3R02_S3B", "rank", "3B"),
        ("G3B02_S3B", "bits", "3B"),
    ]

    seeds = [42, 123, 456]

    for base_run_id, policy, scale in experiments:
        for seed in seeds:
            run_id = f"{base_run_id}_s{seed}"

            # Look for results in various locations
            paths_to_check = [
                # Proxy ablation results (1.7B)
                REPO_ROOT
                / f"results/modal/qwen3_1p7b_gptq_proxy_ablation_s{seed}/{run_id}/metrics.json",
                # Transfer results (8B)
                REPO_ROOT
                / f"results/modal/qwen3_8b_gptq_transfer_s{seed}/{run_id}/metrics.json",
                # Transfer results (3B)
                REPO_ROOT
                / f"results/modal/smollm3_3b_gptq_transfer_s{seed}/{run_id}/metrics.json",
                # Direct modal results
                REPO_ROOT / f"results/modal/{run_id}/metrics.json",
            ]

            perplexity = None
            memory_mb = None
            latency_ms = None

            for metrics_path in paths_to_check:
                if metrics_path.exists():
                    print(f"Found metrics at: {metrics_path}")
                    try:
                        with open(metrics_path, "r") as f:
                            data = json.load(f)
                            perplexity = data.get("perplexity")
                            memory_bytes = data.get("memory_total_bytes")
                            if memory_bytes is not None:
                                memory_mb = memory_bytes / (1024 * 1024)
                            latency_ms = data.get("latency_ms_per_token")
                            if perplexity is not None:
                                break
                    except (json.JSONDecodeError, KeyError):
                        continue

            results.append(
                {
                    "run_id": run_id,
                    "policy": policy,
                    "scale": scale,
                    "seed": seed,
                    "perplexity": perplexity,
                    "memory_mb": memory_mb,
                    "latency_ms_per_token": latency_ms,
                    "status": "completed"
                    if perplexity is not None
                    else "pending/missing",
                }
            )

    # Write CSV
    output_path = REPO_ROOT / "results/analysis/multiseed_stability_all_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "policy",
                "scale",
                "seed",
                "perplexity",
                "memory_mb",
                "latency_ms_per_token",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to: {output_path}")
    print(f"Total experiments tracked: {len(results)}")

    # Print summary
    print("\n=== Results Summary ===")
    for base_run_id, policy, scale in experiments:
        print(f"\n{scale}B {policy.upper()}:")
        for seed in seeds:
            run_id = f"{base_run_id}_s{seed}"
            result = next((r for r in results if r["run_id"] == run_id), None)
            if result and result["perplexity"] is not None:
                print(f"  Seed {seed}: {result['perplexity']:.6f}")
            else:
                print(f"  Seed {seed}: PENDING/MISSING")


if __name__ == "__main__":
    collect_results()
