from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_decomposition.config import RunConfig
from llm_decomposition.io import write_json
from llm_decomposition.methods import method_spec, missing_modules


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    status: str
    message: str
    missing_dependencies: list[str]


class ExperimentExecutor:
    """Generic execution entrypoint for config-driven runs.

    This version is intentionally model-agnostic at the orchestration layer.
    Actual Hugging Face evaluation backends can be added behind method-specific
    implementations without changing the config interface.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def execute(self, config: RunConfig, dry_run: bool = False) -> ExecutionResult:
        spec = method_spec(config.method_name)
        missing = missing_modules(spec.required_modules)
        run_dir = self.root / config.results_dir
        execution_path = run_dir / "execution_status.json"

        if missing:
            result = ExecutionResult(
                run_id=config.run_id,
                status="blocked",
                message=(
                    "Execution backend is not ready because required Python modules "
                    f"are missing for method '{config.method_name}'."
                ),
                missing_dependencies=missing,
            )
            write_json(execution_path, _result_payload(config, result, dry_run=dry_run))
            return result

        if dry_run:
            result = ExecutionResult(
                run_id=config.run_id,
                status="dry_run",
                message=(
                    "Dependencies are available and the config is executable. "
                    "No model run was started because dry-run mode is enabled."
                ),
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=True))
            return result

        if config.method_name == "full_precision":
            from llm_decomposition.hf_backend import execute_full_precision

            metrics = execute_full_precision(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message=(
                    "Full-precision evaluation completed successfully with the "
                    "generic Hugging Face backend."
                ),
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        if config.method_name == "rtn":
            from llm_decomposition.hf_backend import execute_rtn

            metrics = execute_rtn(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message="RTN quantization evaluation completed successfully.",
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        if config.method_name == "gptq":
            from llm_decomposition.hf_backend import execute_gptq

            metrics = execute_gptq(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message="GPTQ quantization evaluation completed successfully.",
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        if config.method_name == "uniform_svd_repair":
            from llm_decomposition.hf_backend import execute_uniform_svd_repair

            metrics = execute_uniform_svd_repair(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message="Uniform SVD repair evaluation completed successfully.",
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        if config.method_name == "mixed_precision_budget_match":
            from llm_decomposition.hf_backend import execute_mixed_precision_budget_match

            metrics = execute_mixed_precision_budget_match(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message="Mixed-precision budget-matched evaluation completed successfully.",
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        if config.method_name == "targeted_mixed_precision":
            from llm_decomposition.hf_backend import execute_targeted_mixed_precision

            metrics = execute_targeted_mixed_precision(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message="Phase-2 targeted mixed-precision evaluation completed successfully.",
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        if config.method_name == "targeted_svd_rank":
            from llm_decomposition.hf_backend import execute_targeted_svd_rank

            metrics = execute_targeted_svd_rank(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message="Phase-2 targeted SVD rank evaluation completed successfully.",
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        if config.method_name == "hybrid_second_stage":
            from llm_decomposition.hf_backend import execute_hybrid_second_stage

            metrics = execute_hybrid_second_stage(self.root, config)
            result = ExecutionResult(
                run_id=config.run_id,
                status="completed",
                message="Hybrid second-stage evaluation completed successfully.",
                missing_dependencies=[],
            )
            write_json(execution_path, _result_payload(config, result, dry_run=False, metrics=metrics))
            return result

        result = ExecutionResult(
            run_id=config.run_id,
            status="not_implemented",
            message=(
                f"Method '{config.method_name}' passed dependency checks but does not "
                "yet have an execution backend."
            ),
            missing_dependencies=[],
        )
        write_json(execution_path, _result_payload(config, result, dry_run=False))
        return result


def _result_payload(
    config: RunConfig,
    result: ExecutionResult,
    dry_run: bool,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": config.run_id,
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": config.bit_width,
        "status": result.status,
        "message": result.message,
        "dry_run": dry_run,
        "missing_dependencies": result.missing_dependencies,
    }
    if metrics is not None:
        payload["metrics_summary"] = {
            "perplexity": metrics.get("perplexity"),
            "memory_total_bytes": metrics.get("memory_total_bytes"),
            "latency_ms_per_token": metrics.get("latency_ms_per_token"),
        }
    return payload
