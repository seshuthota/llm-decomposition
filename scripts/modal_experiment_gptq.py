#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
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


def _resolve_gpu_spec(gpu_name: str):
    normalized = gpu_name.strip().upper()
    if normalized == "T4":
        return modal.gpu.T4()
    if normalized == "A10G":
        return modal.gpu.A10G()
    if normalized == "A100":
        return modal.gpu.A100()
    if normalized in {"A100-80GB", "A100_80GB", "A100:80GB"}:
        return modal.gpu.A100(size="80GB")
    if normalized == "L40S":
        return modal.gpu.L40S()
    if normalized == "H100":
        return modal.gpu.H100()
    return gpu_name


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
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
    .run_commands(
        "python -m pip install --upgrade setuptools wheel",
        "python -m pip install --no-build-isolation gptqmodel"
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_REPO_ROOT,
        copy=False,
        ignore=[".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"],
    )
)

app = modal.App("llm-decomposition-gptq-experiments")
model_volume = modal.Volume.from_name(DEFAULT_MODEL_VOLUME, create_if_missing=True)
results_volume = modal.Volume.from_name(DEFAULT_RESULTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=_resolve_gpu_spec(DEFAULT_GPU),
    timeout=DEFAULT_TIMEOUT,
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

    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    from llm_decomposition.config import load_manifest
    from llm_decomposition.executor import ExperimentExecutor
    from llm_decomposition.prepare import prepare_run

    manifest_obj = load_manifest(repo_root, manifest)
    selected = [config for config in manifest_obj.run_configs if config.run_id == run_id]
    if not selected:
        raise ValueError(f"Run id '{run_id}' was not found in manifest '{manifest}'.")
    config = selected[0]
    model_preflight: dict[str, Any] | None = None

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
            tokenizer_name = config.raw["model"].get("tokenizer_name")
            if tokenizer_name is None:
                config.raw["model"]["tokenizer_name"] = volume_model_path

    phase = config.raw.get("phase", "adhoc")
    local_results_dir = Path(results_prefix) / phase / run_id
    volume_results_dir = Path(REMOTE_RESULTS_ROOT) / local_results_dir
    config.raw["outputs"]["results_dir"] = volume_results_dir.as_posix()
    prepared = prepare_run(repo_root, config)
    results_volume.commit()

    stdout_buffer = io.StringIO()
    executor = ExperimentExecutor(repo_root)
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

    run_dir = repo_root / config.results_dir
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
