#!/usr/bin/env python3
from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import importlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO_ROOT = "/root/project"
REMOTE_MODEL_ROOT = "/vol/models"
REMOTE_RESULTS_ROOT = "/resultsvol"
DEFAULT_GPU = os.environ.get("MODAL_GPU", "A100")
DEFAULT_TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "21600"))
DEFAULT_MODEL_VOLUME = os.environ.get("MODAL_MODEL_VOLUME", "llm-decomposition-models")
DEFAULT_RESULTS_VOLUME = os.environ.get("MODAL_RESULTS_VOLUME", "llm-decomposition-results")
INSTALL_GPTQMODEL = os.environ.get("MODAL_INSTALL_GPTQMODEL", "1") == "1"


def _read_local_hf_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            env[key] = value

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                continue
            parsed = value.strip().strip("'").strip('"')
            if parsed and key not in env:
                env[key] = parsed

    if env.get("HF_TOKEN") and "HUGGING_FACE_HUB_TOKEN" not in env:
        env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
    return env


HF_SECRET = modal.Secret.from_dict(_read_local_hf_env()) if modal.is_local() else modal.Secret.from_dict({})


def _resolve_gpu_spec(gpu_name: str):
    normalized = gpu_name.strip().upper()
    if normalized == "T4":
        return "T4"
    if normalized == "A10G":
        return "A10G"
    if normalized in {"A100", "A100-40GB"}:
        return "A100-40GB"
    if normalized in {"A100-80GB", "A100_80GB", "A100:80GB"}:
        return "A100-80GB"
    if normalized == "L40S":
        return "L40S"
    if normalized == "H100":
        return "H100"
    return gpu_name


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pip",
    "setuptools",
    "wheel",
    "torch==2.7.1",
    "transformers==5.3.0",
    "datasets==4.5.0",
    "accelerate==1.13.0",
    "huggingface_hub==1.6.0",
    "safetensors==0.5.2",
    "numpy",
    "optimum==2.1.0",
    "lm-eval[hf]",
)
image = image.run_commands("python -m pip install --upgrade setuptools wheel")
if INSTALL_GPTQMODEL:
    image = image.run_commands("python -m pip install --no-build-isolation gptqmodel")
image = image.add_local_dir(
    REPO_ROOT,
    remote_path=REMOTE_REPO_ROOT,
    copy=False,
    ignore=[
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".modal_fetch",
        "results",
        "llm-decomposition-results",
    ],
)

app = modal.App("llm-decomposition-gptq-experiments")
model_volume = modal.Volume.from_name(DEFAULT_MODEL_VOLUME, create_if_missing=True)
results_volume = modal.Volume.from_name(DEFAULT_RESULTS_VOLUME, create_if_missing=True)


def _stage_marker(run_dir: Path, stage: str, **extra: Any) -> None:
    payload = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{stage}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results_volume.commit()


def _resolve_remote_target_memory_bytes(run_id: str) -> int | None:
    candidate_paths = [
        Path(REMOTE_RESULTS_ROOT) / f"results/modal/qwen3_1p7b_gptq_baselines/{run_id}/metrics.json",
        Path(REMOTE_RESULTS_ROOT) / f"results/modal/qwen3_1p7b_gptq_transfer/{run_id}/metrics.json",
        Path(REMOTE_RESULTS_ROOT) / f"results/modal/qwen3_1p7b_gptq_proxy_ablation/{run_id}/metrics.json",
        Path(REMOTE_RESULTS_ROOT) / f"results/modal/qwen3_8b_gptq_baselines/{run_id}/metrics.json",
        Path(REMOTE_RESULTS_ROOT) / f"results/modal/qwen3_8b_gptq_transfer/{run_id}/metrics.json",
        Path(REMOTE_RESULTS_ROOT) / f"results/modal/smollm3_3b_gptq_baselines/{run_id}/metrics.json",
        Path(REMOTE_RESULTS_ROOT) / f"results/modal/smollm3_3b_gptq_transfer/{run_id}/metrics.json",
    ]
    for metrics_path in candidate_paths:
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        memory_total_bytes = payload.get("memory_total_bytes")
        if memory_total_bytes is not None:
            return int(memory_total_bytes)

    for metrics_path in sorted((Path(REMOTE_RESULTS_ROOT) / "results").glob("**/metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("run_id") != run_id:
            continue
        memory_total_bytes = payload.get("memory_total_bytes")
        if memory_total_bytes is not None:
            return int(memory_total_bytes)
    return None


def _collect_gptq_stack_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "modal_install_gptqmodel": INSTALL_GPTQMODEL,
    }

    for module_name in ("torch", "transformers", "optimum", "gptqmodel"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            status[f"{module_name}_import_ok"] = False
            status[f"{module_name}_error_type"] = exc.__class__.__name__
            status[f"{module_name}_error_message"] = str(exc)
            continue

        status[f"{module_name}_import_ok"] = True
        status[f"{module_name}_version"] = getattr(module, "__version__", "unknown")

    try:
        from gptqmodel import QuantizeConfig  # type: ignore

        status["gptqmodel_quantize_config_import_ok"] = True
        status["gptqmodel_quantize_config_repr"] = repr(QuantizeConfig)
    except Exception as exc:  # pragma: no cover - diagnostic path
        status["gptqmodel_quantize_config_import_ok"] = False
        status["gptqmodel_quantize_config_error_type"] = exc.__class__.__name__
        status["gptqmodel_quantize_config_error_message"] = str(exc)

    try:
        quantizer_module = importlib.import_module("optimum.gptq.quantizer")
        status["optimum_gptq_quantizer_import_ok"] = True
        status["optimum_quantizer_has_QuantizeConfig"] = hasattr(quantizer_module, "QuantizeConfig")
        status["optimum_quantizer_QuantizeConfig_repr"] = (
            repr(getattr(quantizer_module, "QuantizeConfig"))
            if hasattr(quantizer_module, "QuantizeConfig")
            else None
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        status["optimum_gptq_quantizer_import_ok"] = False
        status["optimum_gptq_quantizer_error_type"] = exc.__class__.__name__
        status["optimum_gptq_quantizer_error_message"] = str(exc)

    return status


@app.function(
    image=image,
    gpu=_resolve_gpu_spec(DEFAULT_GPU),
    timeout=1800,
    secrets=[HF_SECRET],
    volumes={
        REMOTE_MODEL_ROOT: model_volume,
        REMOTE_RESULTS_ROOT: results_volume,
    },
)
def run_diagnostic_remote(
    run_id: str = "diag_q8b",
    model_subpath: str = "",
    results_prefix: str = "results/modal",
) -> dict[str, Any]:
    repo_root = Path(REMOTE_REPO_ROOT)
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    phase = "diagnostics"
    local_results_dir = Path(results_prefix) / phase / run_id
    run_dir = Path(REMOTE_RESULTS_ROOT) / local_results_dir
    _stage_marker(
        run_dir,
        "stage_01_diagnostic_start",
        run_id=run_id,
        model_subpath=model_subpath,
    )

    stack_status = _collect_gptq_stack_status()
    if not all(
        stack_status.get(key, False)
        for key in ("torch_import_ok", "transformers_import_ok", "optimum_import_ok")
    ):
        _stage_marker(run_dir, "stage_02_import_failure", **stack_status)
        return stack_status

    _stage_marker(run_dir, "stage_02_stack_status", **stack_status)

    volume_model_dir = Path(REMOTE_MODEL_ROOT) / model_subpath if model_subpath else None
    preflight = {
        "gptq_stack_status": stack_status,
        "model_subpath": model_subpath,
        "volume_model_dir": None if volume_model_dir is None else volume_model_dir.as_posix(),
        "volume_model_dir_exists": False if volume_model_dir is None else volume_model_dir.exists(),
        "config_exists": False if volume_model_dir is None else (volume_model_dir / "config.json").exists(),
        "tokenizer_config_exists": False if volume_model_dir is None else (volume_model_dir / "tokenizer_config.json").exists(),
    }
    _stage_marker(run_dir, "stage_03_model_preflight", **preflight)
    return preflight


@app.function(
    image=image,
    gpu=_resolve_gpu_spec(DEFAULT_GPU),
    timeout=DEFAULT_TIMEOUT,
    secrets=[HF_SECRET],
    volumes={
        REMOTE_MODEL_ROOT: model_volume,
        REMOTE_RESULTS_ROOT: results_volume,
    },
)
def run_config_remote(
    run_id: str,
    manifest: str,
    model_subpath: str = "",
    results_prefix: str = "results/modal",
) -> dict[str, Any]:
    import traceback
    from pathlib import Path

    repo_root = Path(REMOTE_REPO_ROOT)
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from llm_decomposition.config import load_manifest
    from llm_decomposition.executor import ExperimentExecutor
    from llm_decomposition.prepare import prepare_run

    manifest_obj = load_manifest(repo_root, manifest)
    selected = [config for config in manifest_obj.run_configs if config.run_id == run_id]
    if not selected:
        raise ValueError(f"Run id '{run_id}' was not found in manifest '{manifest}'.")
    config = selected[0]
    phase = config.raw.get("phase", "adhoc")
    local_results_dir = Path(results_prefix) / phase / run_id
    volume_results_dir = Path(REMOTE_RESULTS_ROOT) / local_results_dir
    run_dir = volume_results_dir
    _stage_marker(
        run_dir,
        "stage_01_config_loaded",
        run_id=run_id,
        manifest=manifest,
        local_results_dir=local_results_dir.as_posix(),
    )
    model_preflight: dict[str, Any] | None = None
    gptq_stack_status = _collect_gptq_stack_status()

    if model_subpath:
        requested_model = config.raw["model"]["name"]
        requested_tokenizer = config.raw["model"].get("tokenizer_name")
        volume_model_dir = Path(REMOTE_MODEL_ROOT) / model_subpath
        config_path = volume_model_dir / "config.json"
        tokenizer_config_path = volume_model_dir / "tokenizer_config.json"
        model_preflight = {
            "requested_model": requested_model,
            "requested_tokenizer": requested_tokenizer,
            "model_subpath": model_subpath,
            "volume_model_dir": volume_model_dir.as_posix(),
            "volume_model_dir_exists": volume_model_dir.exists(),
            "config_exists": config_path.exists(),
            "tokenizer_config_exists": tokenizer_config_path.exists(),
        }
        if volume_model_dir.exists() and config_path.exists():
            volume_model_path = str(volume_model_dir)
            config.raw["model"]["name"] = volume_model_path
            if tokenizer_config_path.exists():
                config.raw["model"]["tokenizer_name"] = volume_model_path
    _stage_marker(
        run_dir,
        "stage_02_model_preflight",
        run_id=run_id,
        requested_model=config.raw["model"]["name"],
        requested_tokenizer=config.raw["model"].get("tokenizer_name"),
        model_subpath=model_subpath,
        model_preflight=model_preflight,
        gptq_stack_status=gptq_stack_status,
    )
    config.raw["outputs"]["results_dir"] = volume_results_dir.as_posix()
    prepared = prepare_run(repo_root, config)
    run_dir = Path(config.results_dir)
    _stage_marker(
        run_dir,
        "stage_03_prepare_run_done",
        run_id=run_id,
        prepared=prepared,
        remote_results_dir=config.results_dir,
    )
    results_volume.commit()

    stdout_buffer = io.StringIO()
    executor = ExperimentExecutor(repo_root)
    _stage_marker(
        run_dir,
        "stage_04_before_execute",
        run_id=run_id,
        method=config.method_name,
        model_name=config.model_name,
        gptq_stack_status=gptq_stack_status,
    )
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
        try:
            execution = executor.execute(config, dry_run=False)
            error = None
        except Exception as exc:  # pragma: no cover - remote failure payload
            execution = None
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
    _stage_marker(
        run_dir,
        "stage_05_after_execute",
        run_id=run_id,
        execution_status=None if execution is None else execution.status,
        error_type=None if error is None else error["type"],
    )

    log_path = run_dir / "modal_run.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(stdout_buffer.getvalue(), encoding="utf-8")
    results_volume.commit()

    metrics_summary = None
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8"))

    artifact_files: dict[str, str] = {}
    for filename in (
        "metrics.json",
        "layer_errors.json",
        "residual_profiles.json",
        "resolved_config.json",
        "gptq_validation.json",
        "notes.md",
        "execution_status.json",
        "modal_run.log",
        "actions.json",
        "downstream_metrics.json",
    ):
        path = run_dir / filename
        if path.exists():
            artifact_files[filename] = path.read_text(encoding="utf-8", errors="ignore")

    execution_payload = None
    if execution is not None:
        execution_payload = {
            "run_id": execution.run_id,
            "status": execution.status,
            "message": execution.message,
            "missing_dependencies": execution.missing_dependencies,
        }

    payload = {
        "run_id": run_id,
        "manifest": manifest,
        "phase": phase,
        "prepared": prepared,
        "remote_results_dir": config.results_dir,
        "local_results_dir": local_results_dir.as_posix(),
        "model_source": model_subpath or config.model_name,
        "model_preflight": model_preflight,
        "gptq_stack_status": gptq_stack_status,
        "gpu": DEFAULT_GPU,
        "timeout": DEFAULT_TIMEOUT,
        "stdout_tail": stdout_buffer.getvalue()[-8000:],
        "execution": execution_payload,
        "metrics": metrics_summary,
        "artifact_files": artifact_files,
        "error": error,
    }
    (run_dir / "modal_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results_volume.commit()
    return payload


@app.function(
    image=image,
    gpu=_resolve_gpu_spec(DEFAULT_GPU),
    timeout=DEFAULT_TIMEOUT,
    secrets=[HF_SECRET],
    volumes={
        REMOTE_MODEL_ROOT: model_volume,
        REMOTE_RESULTS_ROOT: results_volume,
    },
)
def run_latency_remote(
    run_id: str,
    manifest: str,
    batch_size: int,
    model_subpath: str = "",
    prompt_length: int = 512,
    decode_length: int = 128,
    warmup_iterations: int = 3,
    timed_iterations: int = 10,
    prompt_template: str = (
        "Summarize the key engineering tradeoffs of deploying a quantized language model "
        "under a fixed GPU memory budget. Focus on accuracy, memory footprint, and "
        "inference latency, and explain why a practitioner might choose extra bits or "
        "low-rank repair for a single-device deployment setting."
    ),
    results_prefix: str = "results/modal_latency",
) -> dict[str, Any]:
    import traceback
    from pathlib import Path

    repo_root = Path(REMOTE_REPO_ROOT)
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from llm_decomposition.config import RunConfig, load_manifest
    from llm_decomposition.latency_benchmark import LatencyBenchmarkSpec, run_latency_benchmark

    manifest_obj = load_manifest(repo_root, manifest)
    selected = [config for config in manifest_obj.run_configs if config.run_id == run_id]
    if not selected:
        raise ValueError(f"Run id '{run_id}' was not found in manifest '{manifest}'.")
    config = RunConfig(path=selected[0].path, raw=deepcopy(selected[0].raw))
    phase = config.raw.get("phase", "adhoc")
    benchmark_run_id = f"{run_id}__bs{batch_size}"
    local_results_dir = Path(results_prefix) / phase / benchmark_run_id
    volume_results_dir = Path(REMOTE_RESULTS_ROOT) / local_results_dir
    run_dir = volume_results_dir
    _stage_marker(
        run_dir,
        "stage_01_config_loaded",
        run_id=run_id,
        benchmark_run_id=benchmark_run_id,
        manifest=manifest,
        batch_size=batch_size,
        local_results_dir=local_results_dir.as_posix(),
    )
    model_preflight: dict[str, Any] | None = None
    gptq_stack_status = _collect_gptq_stack_status()

    if model_subpath:
        requested_model = config.raw["model"]["name"]
        requested_tokenizer = config.raw["model"].get("tokenizer_name")
        volume_model_dir = Path(REMOTE_MODEL_ROOT) / model_subpath
        config_path = volume_model_dir / "config.json"
        tokenizer_config_path = volume_model_dir / "tokenizer_config.json"
        model_preflight = {
            "requested_model": requested_model,
            "requested_tokenizer": requested_tokenizer,
            "model_subpath": model_subpath,
            "volume_model_dir": volume_model_dir.as_posix(),
            "volume_model_dir_exists": volume_model_dir.exists(),
            "config_exists": config_path.exists(),
            "tokenizer_config_exists": tokenizer_config_path.exists(),
        }
        if volume_model_dir.exists() and config_path.exists():
            volume_model_path = str(volume_model_dir)
            config.raw["model"]["name"] = volume_model_path
            if tokenizer_config_path.exists():
                config.raw["model"]["tokenizer_name"] = volume_model_path
    method_cfg = config.raw.get("method", {})
    if config.method_name in {"targeted_mixed_precision", "targeted_svd_rank"} and method_cfg.get(
        "base_method", "rtn"
    ) == "gptq":
        method_cfg.setdefault("selection_profile_source", "current_base_model")
    if method_cfg.get("budget_bytes") is None and method_cfg.get("base_run_id"):
        resolved_budget = _resolve_remote_target_memory_bytes(method_cfg["base_run_id"])
        if resolved_budget is not None:
            percent = method_cfg.get("budget_percent_of_base")
            if percent is None:
                method_cfg["budget_bytes"] = resolved_budget
            else:
                method_cfg["budget_bytes"] = max(int(round(resolved_budget * (float(percent) / 100.0))), 0)
    config.raw["outputs"]["results_dir"] = volume_results_dir.as_posix()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config.raw, indent=2), encoding="utf-8")
    _stage_marker(
        run_dir,
        "stage_02_model_preflight",
        run_id=run_id,
        benchmark_run_id=benchmark_run_id,
        batch_size=batch_size,
        requested_model=config.raw["model"]["name"],
        requested_tokenizer=config.raw["model"].get("tokenizer_name"),
        model_subpath=model_subpath,
        model_preflight=model_preflight,
        gptq_stack_status=gptq_stack_status,
    )

    spec = LatencyBenchmarkSpec(
        batch_size=batch_size,
        prompt_length=prompt_length,
        decode_length=decode_length,
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        prompt_template=prompt_template,
    )
    stdout_buffer = io.StringIO()
    _stage_marker(
        run_dir,
        "stage_03_before_benchmark",
        run_id=run_id,
        benchmark_run_id=benchmark_run_id,
        benchmark_spec=asdict(spec),
        model_name=config.model_name,
        method=config.method_name,
    )
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
        try:
            benchmark = run_latency_benchmark(
                repo_root,
                config,
                spec,
                output_path=run_dir / "latency_benchmark.json",
            )
            error = None
        except Exception as exc:  # pragma: no cover - remote failure payload
            benchmark = None
            error = {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
    _stage_marker(
        run_dir,
        "stage_04_after_benchmark",
        run_id=run_id,
        benchmark_run_id=benchmark_run_id,
        status=None if benchmark is None else benchmark.get("status"),
        error_type=None if error is None else error["type"],
    )

    log_path = run_dir / "modal_run.log"
    log_path.write_text(stdout_buffer.getvalue(), encoding="utf-8")
    results_volume.commit()

    artifact_files: dict[str, str] = {}
    for filename in (
        "resolved_config.json",
        "latency_benchmark.json",
        "modal_run.log",
    ):
        path = run_dir / filename
        if path.exists():
            artifact_files[filename] = path.read_text(encoding="utf-8", errors="ignore")

    payload = {
        "run_id": run_id,
        "benchmark_run_id": benchmark_run_id,
        "manifest": manifest,
        "phase": phase,
        "remote_results_dir": volume_results_dir.as_posix(),
        "local_results_dir": local_results_dir.as_posix(),
        "model_source": model_subpath or config.model_name,
        "model_preflight": model_preflight,
        "gptq_stack_status": gptq_stack_status,
        "gpu": DEFAULT_GPU,
        "timeout": DEFAULT_TIMEOUT,
        "benchmark_spec": asdict(spec),
        "stdout_tail": stdout_buffer.getvalue()[-8000:],
        "benchmark_summary": None if benchmark is None else benchmark.get("summary"),
        "artifact_files": artifact_files,
        "error": error,
    }
    (run_dir / "modal_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results_volume.commit()
    return payload


def _write_local_artifacts(payload: dict[str, Any]) -> None:
    run_dir = REPO_ROOT / payload.get("local_results_dir", payload["remote_results_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in payload.get("artifact_files", {}).items():
        (run_dir / filename).write_text(contents, encoding="utf-8")
    (run_dir / "modal_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


@app.local_entrypoint()
def main(
    run_id: str,
    manifest: str = "configs/scaleup_1p7b_gptq/qwen3_1p7b_gptq_transfer_manifest.json",
    model_subpath: str = "",
    results_prefix: str = "results/modal",
) -> None:
    payload = run_config_remote.remote(
        run_id=run_id,
        manifest=manifest,
        model_subpath=model_subpath,
        results_prefix=results_prefix,
    )
    _write_local_artifacts(payload)

    print(f"Run: {payload['run_id']}")
    print(f"Phase: {payload['phase']}")
    print(f"Remote results dir: {payload['remote_results_dir']}")
    print(f"Model source: {payload['model_source']}")
    if payload.get("execution") is not None:
        print(f"Status: {payload['execution']['status']}")
    if payload.get("metrics") is not None:
        metrics = payload["metrics"]
        print(
            "Metrics: "
            f"perplexity={metrics.get('perplexity')} "
            f"memory_total_bytes={metrics.get('memory_total_bytes')}"
        )
    if payload.get("error") is not None:
        print("Remote error:")
        print(payload["error"]["traceback"])
    else:
        print("Remote log tail:")
        print(payload.get("stdout_tail", ""))
