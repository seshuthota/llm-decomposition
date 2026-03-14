#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import modal


REMOTE_MODEL_ROOT = "/vol/models"
DEFAULT_TIMEOUT = int(os.environ.get("MODAL_STAGE_TIMEOUT", "14400"))
DEFAULT_MODEL_VOLUME = os.environ.get("MODAL_MODEL_VOLUME", "llm-decomposition-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub==1.6.0",
        "hf_transfer>=0.1.9",
    )
)

app = modal.App("llm-decomposition-model-staging")
model_volume = modal.Volume.from_name(DEFAULT_MODEL_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    timeout=DEFAULT_TIMEOUT,
    volumes={REMOTE_MODEL_ROOT: model_volume},
)
def stage_model(repo_id: str, model_subpath: str) -> str:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    target_dir = Path(REMOTE_MODEL_ROOT) / model_subpath
    target_dir.mkdir(parents=True, exist_ok=True)

    existing = target_dir / "config.json"
    if existing.exists():
        return target_dir.as_posix()

    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir.as_posix(),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    model_volume.commit()
    return target_dir.as_posix()


@app.local_entrypoint()
def main(repo_id: str, model_subpath: str) -> None:
    staged_path = stage_model.remote(repo_id=repo_id, model_subpath=model_subpath)
    print(staged_path)
