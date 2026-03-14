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

DEFAULT_GPU = os.environ.get("MODAL_GPU", "T4")
DEFAULT_TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", "14400"))
DEFAULT_MODEL_VOLUME = os.environ.get("MODAL_MODEL_VOLUME", "llm-decomposition-models")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==5.3.0",
        "datasets==4.5.0",
        "accelerate==1.13.0",
        "huggingface_hub==1.6.0",
        "safetensors==0.5.2",
        "numpy",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path=REMOTE_REPO_ROOT,
        copy=False,
        ignore=[
            ".git",
            ".venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            "llm-decomposition-results",
            ".cache",
        ],
    )
)

app = modal.App("llm-decomposition-experiments")
model_volume = modal.Volume.from_name(DEFAULT_MODEL_VOLUME, create_if_missing=True)


class _TeeBuffer(io.StringIO):
    def __init__(self, mirror: io.TextIOBase) -> None:
        super().__init__()
        self._mirror = mirror

    def write(self, s: str) -> int:
        self._mirror.write(s)
        self._mirror.flush()
        return super().write(s)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=DEFAULT_TIMEOUT,
    volumes={REMOTE_MODEL_ROOT: model_volume},
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

    # Keep the remote environment deterministic. Public datasets work without
    # an HF token, and volume-backed local model paths do not need one.
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

    if model_subpath:
        volume_model_path = str((Path(REMOTE_MODEL_ROOT) / model_subpath).resolve())
        config.raw["model"]["name"] = volume_model_path
        if config.raw["model"].get("tokenizer_name") is not None:
            config.raw["model"]["tokenizer_name"] = volume_model_path

    phase = config.raw.get("phase", "adhoc")
    config.raw["outputs"]["results_dir"] = f"{results_prefix}/{phase}/{run_id}"
    prepared = prepare_run(repo_root, config)

    stdout_buffer = _TeeBuffer(sys.__stdout__)
    stderr_buffer = _TeeBuffer(sys.__stderr__)
    executor = ExperimentExecutor(repo_root)
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
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
    artifact_files: dict[str, str] = {}
    if run_dir.exists():
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(repo_root).as_posix()
            try:
                artifact_files[relative] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

    metrics_summary = None
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics_summary = json.loads(metrics_path.read_text(encoding="utf-8"))

    execution_payload = None
    if execution is not None:
        execution_payload = {
            "run_id": execution.run_id,
            "status": execution.status,
            "message": execution.message,
            "missing_dependencies": execution.missing_dependencies,
        }

    return {
        "run_id": run_id,
        "manifest": manifest,
        "phase": phase,
        "prepared": prepared,
        "remote_results_dir": config.results_dir,
        "model_source": model_subpath or config.model_name,
        "gpu": DEFAULT_GPU,
        "timeout": DEFAULT_TIMEOUT,
        "stdout": stdout_buffer.getvalue() + stderr_buffer.getvalue(),
        "execution": execution_payload,
        "metrics": metrics_summary,
        "artifacts": artifact_files,
        "error": error,
    }


def _write_local_artifacts(payload: dict[str, Any]) -> None:
    repo_root = REPO_ROOT
    for relative, content in payload.get("artifacts", {}).items():
        destination = repo_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")

    log_path = repo_root / payload["remote_results_dir"] / "modal_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(payload.get("stdout", ""), encoding="utf-8")


@app.local_entrypoint()
def main(
    run_id: str,
    manifest: str = "configs/phase2/phase2_matched_frontier_manifest.json",
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
        print("Remote log:")
        print(payload.get("stdout", ""))
