"""Downstream zero-shot / few-shot evaluation using lm-eval-harness.

This module wraps the ``lm_eval`` library to evaluate models that have
already been loaded and optionally quantized / repaired in-memory.  The
key design choice is to pass the **pre-built model object** to the HFLM
wrapper rather than a model path, because our quantised+repaired models
cannot be trivially serialised back to a Hugging Face checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Default task suite and per-task few-shot settings.  These can be
# overridden by the ``downstream`` config section in each run config.
# ---------------------------------------------------------------------------

DEFAULT_TASKS: list[str] = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "piqa",
    "boolq",
]

DEFAULT_NUM_FEWSHOT: dict[str, int] = {
    "hellaswag": 0,
    "arc_easy": 0,
    "arc_challenge": 25,
    "winogrande": 5,
    "piqa": 0,
    "boolq": 0,
}

DEFAULT_BATCH_SIZE: int = 4


def evaluate_downstream(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    tasks: list[str] | None = None,
    num_fewshot: dict[str, int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Run lm-eval-harness on *model* for the requested *tasks*.

    Parameters
    ----------
    model:
        An already-loaded ``PreTrainedModel`` (or quantised variant).
    tokenizer:
        The corresponding tokenizer.
    tasks:
        List of lm-eval task names.  Defaults to ``DEFAULT_TASKS``.
    num_fewshot:
        Per-task few-shot count overrides.  Defaults to
        ``DEFAULT_NUM_FEWSHOT``.
    batch_size:
        Evaluation batch size passed to lm-eval.
    device:
        Device the model resides on.

    Returns
    -------
    dict
        ``{"results": {task: {metric: value}}}``
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    resolved_tasks = tasks or list(DEFAULT_TASKS)
    resolved_fewshot = dict(DEFAULT_NUM_FEWSHOT)
    if num_fewshot is not None:
        resolved_fewshot.update(num_fewshot)

    device_str = str(device) if isinstance(device, torch.device) else device

    # Build an HFLM wrapper around the pre-loaded model.
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device_str,
    )

    # Run evaluations one task at a time so we can supply per-task
    # few-shot values (simple_evaluate only accepts a single num_fewshot).
    merged_results: dict[str, dict[str, Any]] = {}
    for task_name in resolved_tasks:
        fewshot = resolved_fewshot.get(task_name, 0)
        print(f"  [downstream] evaluating {task_name} ({fewshot}-shot)")
        output = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=fewshot,
            batch_size=batch_size,
            device=device_str,
            log_samples=False,
        )
        task_results = output.get("results", {}).get(task_name, {})
        merged_results[task_name] = task_results

    return {
        "results": merged_results,
    }


def write_downstream_metrics(
    run_dir: Path,
    run_id: str,
    downstream_results: dict[str, Any],
    output_file: str = "downstream_metrics.json",
) -> Path:
    """Persist downstream evaluation results to the run directory."""
    output_path = run_dir / output_file
    payload = {
        "run_id": run_id,
        "status": "completed",
        **downstream_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return output_path
