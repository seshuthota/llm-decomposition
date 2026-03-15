from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_repo_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)

    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def _hf_token() -> str | None:
    _load_repo_env()
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    dtype: torch.dtype
    device_label: str


def resolve_runtime_context(dtype_preferences: list[str] | None) -> RuntimeContext:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = _resolve_dtype(dtype_preferences or ["bfloat16", "float16", "float32"])
        return RuntimeContext(device=device, dtype=dtype, device_label="cuda")
    return RuntimeContext(device=torch.device("cpu"), dtype=torch.float32, device_label="cpu")


def load_tokenizer(model_name: str):
    token = _hf_token()
    tokenizer_kwargs = {"use_fast": True}
    model_path = Path(model_name)
    if model_path.is_absolute():
        tokenizer_kwargs["local_files_only"] = True
    elif token is not None:
        tokenizer_kwargs["token"] = token
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(model_name: str, runtime: RuntimeContext):
    kwargs = {
        "low_cpu_mem_usage": True,
    }
    if runtime.device.type == "cuda":
        kwargs["torch_dtype"] = runtime.dtype
    token = _hf_token()
    model_path = Path(model_name)
    if model_path.is_absolute():
        kwargs["local_files_only"] = True
    elif token is not None:
        kwargs["token"] = token
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(runtime.device)
    model.eval()
    return model


def load_text_split(dataset_name: str, subset: str, split: str):
    token = _hf_token()
    if token is not None:
        return load_dataset(dataset_name, subset, split=split, token=token)
    return load_dataset(dataset_name, subset, split=split)


def build_fixed_length_sequences(
    tokenizer,
    dataset,
    sequence_length: int,
    num_sequences: int | None,
) -> list[torch.Tensor]:
    text_field = _detect_text_field(dataset.column_names)
    texts = [text for text in dataset[text_field] if isinstance(text, str) and text.strip()]
    separator_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        separator_ids = [tokenizer.eos_token_id]

    token_chunks: list[torch.Tensor] = []
    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        ids = encoded["input_ids"]
        if ids:
            token_chunks.append(torch.tensor(ids, dtype=torch.long))
            if separator_ids:
                token_chunks.append(torch.tensor(separator_ids, dtype=torch.long))

    if not token_chunks:
        return []

    input_ids = torch.cat(token_chunks, dim=0)

    sequences: list[torch.Tensor] = []
    max_tokens = input_ids.numel()
    for start in range(0, max_tokens - sequence_length + 1, sequence_length):
        end = start + sequence_length
        sequences.append(input_ids[start:end].clone())
        if num_sequences is not None and len(sequences) >= num_sequences:
            break
    return sequences


def evaluate_perplexity(
    model,
    sequences: Iterable[torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    durations: list[float] = []

    with torch.no_grad():
        for index, sequence in enumerate(sequences):
            inputs = sequence.unsqueeze(0).to(device)
            start = time.perf_counter()
            outputs = model(input_ids=inputs, labels=inputs)
            logits = outputs.logits.detach()
            loss_tensor = outputs.loss.detach()
            if not torch.isfinite(loss_tensor).all().item():
                raise RuntimeError(
                    f"Non-finite loss detected during evaluation at batch {index}."
                )
            if not torch.isfinite(logits).all().item():
                raise RuntimeError(
                    f"Non-finite logits detected during evaluation at batch {index}."
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            if index > 0:
                durations.append(end - start)

            loss = float(loss_tensor.cpu())
            token_count = max(inputs.numel() - 1, 1)
            total_nll += loss * token_count
            total_tokens += token_count

    mean_nll = total_nll / max(total_tokens, 1)
    perplexity = math.exp(mean_nll)
    total_eval_tokens = total_tokens
    total_duration = sum(durations)
    latency_ms_per_token = None
    if total_duration > 0 and total_eval_tokens > 0:
        latency_ms_per_token = 1000.0 * total_duration / total_eval_tokens

    return {
        "perplexity": perplexity,
        "latency_ms_per_token": latency_ms_per_token,
        "evaluated_tokens": float(total_eval_tokens),
    }


def validate_finite_outputs(
    model,
    sequences: Iterable[torch.Tensor],
    device: torch.device,
    max_batches: int = 1,
) -> dict[str, object]:
    checked_batches = 0
    batch_summaries: list[dict[str, object]] = []

    with torch.no_grad():
        for sequence in sequences:
            if checked_batches >= max_batches:
                break
            inputs = sequence.unsqueeze(0).to(device)
            outputs = model(input_ids=inputs, labels=inputs)
            logits = outputs.logits.detach()
            loss_tensor = outputs.loss.detach()

            logits_finite = torch.isfinite(logits)
            loss_finite = torch.isfinite(loss_tensor)
            finite_logits = logits[logits_finite]

            batch_summary = {
                "batch_index": checked_batches,
                "sequence_length": int(sequence.numel()),
                "loss": float(loss_tensor.cpu()),
                "loss_is_finite": bool(loss_finite.all().item()),
                "logits_all_finite": bool(logits_finite.all().item()),
                "finite_logit_count": int(logits_finite.sum().item()),
                "total_logit_count": int(logits.numel()),
                "finite_logit_min": float(finite_logits.min().cpu()) if finite_logits.numel() else None,
                "finite_logit_max": float(finite_logits.max().cpu()) if finite_logits.numel() else None,
            }
            batch_summaries.append(batch_summary)
            checked_batches += 1

            if not batch_summary["loss_is_finite"] or not batch_summary["logits_all_finite"]:
                return {
                    "all_finite": False,
                    "checked_batches": checked_batches,
                    "batches": batch_summaries,
                }

    return {
        "all_finite": True,
        "checked_batches": checked_batches,
        "batches": batch_summaries,
    }


def _detect_text_field(column_names: list[str]) -> str:
    for candidate in ("text", "content", "sentence"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Could not detect a text column from columns: {column_names}")


def _resolve_dtype(dtype_preferences: list[str]) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    for dtype_name in dtype_preferences:
        if dtype_name in mapping:
            return mapping[dtype_name]
    return torch.float32
