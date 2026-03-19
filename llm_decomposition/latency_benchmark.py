from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from llm_decomposition.config import RunConfig
from llm_decomposition.gptq_backend import estimate_gptq_layer_stats, quantize_model_gptq
from llm_decomposition.hf_backend import (
    _apply_targeted_bit_actions,
    _apply_targeted_rank_actions,
    _build_bit_actions,
    _build_gptq_base_model,
    _build_rank_actions,
    _candidate_layers_from_pool,
    _clear_cuda_cache,
    _collapse_selected_rank_actions,
    _load_candidate_pool,
    _move_model_to_runtime_device,
    _resolve_candidate_bit_widths,
    _resolve_runtime_for_config,
    _resolve_selection_layer_error_map,
    _select_bit_actions,
    _select_rank_actions,
)
from llm_decomposition.hf_utils import load_causal_lm, load_tokenizer
from llm_decomposition.io import write_json
from llm_decomposition.profiling import profile_residual_svd


DEFAULT_PROMPT_TEMPLATE = (
    "Summarize the key engineering tradeoffs of deploying a quantized language model "
    "under a fixed GPU memory budget. Focus on accuracy, memory footprint, and "
    "inference latency, and explain why a practitioner might choose extra bits or "
    "low-rank repair for a single-device deployment setting."
)


@dataclass(frozen=True)
class LatencyBenchmarkSpec:
    """Fixed benchmark settings for one latency job."""

    batch_size: int
    prompt_length: int = 512
    decode_length: int = 128
    warmup_iterations: int = 3
    timed_iterations: int = 10
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE


@dataclass
class PreparedBenchmarkModel:
    """Prepared model bundle for a latency benchmark."""

    model: nn.Module
    tokenizer: Any
    runtime: Any
    metadata: dict[str, Any]


def run_latency_benchmark(
    root: Path,
    config: RunConfig,
    spec: LatencyBenchmarkSpec,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a benchmark model and run deterministic generation latency measurement.

    Args:
        root: Repository root path.
        config: Existing experiment config describing the policy to benchmark.
        spec: Frozen latency benchmark settings.
        output_path: Optional path where the resulting payload should be written.

    Returns:
        A JSON-safe dictionary containing raw repetition measurements and aggregate
        latency statistics.

    Raises:
        NotImplementedError: If the method/policy is not yet supported by the
            benchmark reconstruction path.
        RuntimeError: If generation returns an unexpected token count.
    """

    prepared = prepare_benchmark_model(root, config)
    try:
        payload = _benchmark_prepared_model(prepared, config, spec)
        if output_path is not None:
            write_json(output_path, payload)
        return payload
    finally:
        prepared.model.to("cpu")
        _clear_cuda_cache()


def prepare_benchmark_model(root: Path, config: RunConfig) -> PreparedBenchmarkModel:
    """Reconstruct a benchmarkable in-memory model from an existing experiment config."""

    if config.method_name == "gptq":
        return _prepare_gptq_model(config)
    if config.method_name == "targeted_mixed_precision":
        return _prepare_targeted_bits_model(root, config)
    if config.method_name == "targeted_svd_rank":
        return _prepare_targeted_rank_model(root, config)
    raise NotImplementedError(
        f"Latency benchmark currently supports only GPTQ baseline, targeted bits, and "
        f"targeted rank. Got method '{config.method_name}'."
    )


def _prepare_gptq_model(config: RunConfig) -> PreparedBenchmarkModel:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.tokenizer_name)
    method_cfg = config.raw["method"]

    fp_model = load_causal_lm(config.model_name, runtime)
    layer_stats = estimate_gptq_layer_stats(
        model=fp_model,
        bit_width=method_cfg["bit_width"],
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
    )
    memory_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
    fp_model.to("cpu")
    _clear_cuda_cache()

    quantized_model = quantize_model_gptq(
        None,
        config,
        tokenizer,
        model_name=config.model_name,
        runtime=runtime,
    )
    _prepare_model_for_generation(quantized_model, tokenizer, runtime.device)

    return PreparedBenchmarkModel(
        model=quantized_model,
        tokenizer=tokenizer,
        runtime=runtime,
        metadata={
            "policy_type": "baseline_4bit",
            "memory_total_bytes": memory_total_bytes,
            "reconstruction": "gptq_baseline",
        },
    )


def _prepare_targeted_bits_model(root: Path, config: RunConfig) -> PreparedBenchmarkModel:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.tokenizer_name)
    method_cfg = config.raw["method"]
    base_method = method_cfg.get("base_method", "rtn")
    if base_method != "gptq":
        raise NotImplementedError("Latency benchmark currently supports GPTQ-backed targeted bits only.")

    fp_model = load_causal_lm(config.model_name, runtime)
    candidate_pool = _load_candidate_pool(root, method_cfg["candidate_pool_path"])
    candidate_layers = _candidate_layers_from_pool(candidate_pool)
    candidate_bit_widths = _resolve_candidate_bit_widths(candidate_pool, method_cfg)
    if not candidate_bit_widths:
        raise ValueError("No candidate bit widths are available for targeted mixed precision.")

    fp_model.to("cpu")
    _clear_cuda_cache()
    quantized_model, base_total_bytes, layer_stats = _build_gptq_base_model(
        config=config,
        tokenizer=tokenizer,
        source_model=None,
        fp_reference_model=fp_model,
    )
    _move_model_to_runtime_device(fp_model, runtime.device)
    _move_model_to_runtime_device(quantized_model, runtime.device)
    budget_bytes = _resolve_benchmark_budget_bytes(base_total_bytes, method_cfg)
    layer_error_map, selection_profiling_wall_time_s = _resolve_selection_layer_error_map(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
        target_layers=candidate_layers,
        fallback_source_path=candidate_pool["source_layer_errors_path"],
    )
    actions = _build_bit_actions(
        fp_model=fp_model,
        candidate_layers=candidate_layers,
        layer_error_map=layer_error_map,
        base_bit_width=method_cfg.get("base_bit_width", 4),
        target_bit_widths=candidate_bit_widths,
        group_size=candidate_pool.get("group_size", 128),
        symmetric=candidate_pool.get("symmetric", True),
        proxy_family=method_cfg.get("proxy_family", "activation"),
        target_granularity=method_cfg.get("target_granularity", "matrix"),
        row_block_size=method_cfg.get("row_block_size"),
        column_block_size=method_cfg.get("column_block_size"),
    )
    selected_actions = _select_bit_actions(
        actions,
        allocator=method_cfg.get("allocator", "greedy_activation"),
        budget_bytes=budget_bytes,
    )
    upgraded_layers = _apply_targeted_bit_actions(
        fp_model=fp_model,
        quantized_model=quantized_model,
        layer_stats=layer_stats,
        selected_actions=selected_actions,
        target_bit_width=max(candidate_bit_widths),
        group_size=candidate_pool.get("group_size", 128),
        symmetric=candidate_pool.get("symmetric", True),
    )
    memory_total_bytes = base_total_bytes + sum(action.byte_cost for action in selected_actions)

    _prepare_model_for_generation(quantized_model, tokenizer, runtime.device)

    return PreparedBenchmarkModel(
        model=quantized_model,
        tokenizer=tokenizer,
        runtime=runtime,
        metadata={
            "policy_type": "bits",
            "memory_total_bytes": memory_total_bytes,
            "selection_profiling_wall_time_s": selection_profiling_wall_time_s,
            "selected_action_count": len(selected_actions),
            "upgraded_layers": upgraded_layers,
            "reconstruction": "targeted_mixed_precision_gptq",
        },
    )


def _prepare_targeted_rank_model(root: Path, config: RunConfig) -> PreparedBenchmarkModel:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.tokenizer_name)
    method_cfg = config.raw["method"]
    base_method = method_cfg.get("base_method", "rtn")
    if base_method != "gptq":
        raise NotImplementedError("Latency benchmark currently supports GPTQ-backed targeted rank only.")

    fp_model = load_causal_lm(config.model_name, runtime)
    candidate_pool = _load_candidate_pool(root, method_cfg["candidate_pool_path"])
    candidate_layers = _candidate_layers_from_pool(candidate_pool)

    fp_model.to("cpu")
    _clear_cuda_cache()
    quantized_model, base_total_bytes, layer_stats = _build_gptq_base_model(
        config=config,
        tokenizer=tokenizer,
        source_model=None,
        fp_reference_model=fp_model,
    )
    budget_bytes = _resolve_benchmark_budget_bytes(base_total_bytes, method_cfg)
    layer_error_map, selection_profiling_wall_time_s = _resolve_selection_layer_error_map(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
        target_layers=candidate_layers,
        fallback_source_path=candidate_pool["source_layer_errors_path"],
    )
    residual_profiles = profile_residual_svd(
        fp_model,
        quantized_model,
        candidate_layers,
        method_cfg.get("candidate_ranks", candidate_pool["rank_actions"]["candidate_ranks"]),
    )
    actions = _build_rank_actions(
        fp_model=fp_model,
        candidate_layers=candidate_layers,
        layer_error_map=layer_error_map,
        residual_profiles=residual_profiles,
        candidate_ranks=method_cfg.get("candidate_ranks", candidate_pool["rank_actions"]["candidate_ranks"]),
        factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
        proxy_family=method_cfg.get("proxy_family", "activation"),
    )
    selected_actions = _select_rank_actions(
        actions,
        allocator=method_cfg.get("allocator", "greedy_activation"),
        budget_bytes=budget_bytes,
        family_rounds=int(method_cfg.get("family_rounds", 1)),
    )
    selected_layer_ranks = _collapse_selected_rank_actions(selected_actions)
    _apply_targeted_rank_actions(
        fp_model=fp_model,
        quantized_model=quantized_model,
        layer_stats=layer_stats,
        selected_actions=selected_actions,
        factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
    )
    repair_factor_bytes = sum(item.get("repair_factor_bytes", 0) for item in layer_stats.values())
    memory_total_bytes = base_total_bytes + repair_factor_bytes

    _prepare_model_for_generation(quantized_model, tokenizer, runtime.device)

    return PreparedBenchmarkModel(
        model=quantized_model,
        tokenizer=tokenizer,
        runtime=runtime,
        metadata={
            "policy_type": "rank",
            "memory_total_bytes": memory_total_bytes,
            "selection_profiling_wall_time_s": selection_profiling_wall_time_s,
            "selected_action_count": len(selected_actions),
            "selected_layer_ranks": selected_layer_ranks,
            "repair_factor_bytes": repair_factor_bytes,
            "reconstruction": "targeted_svd_rank_gptq",
        },
    )


def _prepare_model_for_generation(model: nn.Module, tokenizer: Any, runtime_device: torch.device) -> None:
    _move_model_to_runtime_device(model, runtime_device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.do_sample = False
        generation_config.use_cache = True
        if getattr(generation_config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            generation_config.pad_token_id = tokenizer.pad_token_id
        if getattr(generation_config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
            generation_config.eos_token_id = tokenizer.eos_token_id


def _resolve_benchmark_budget_bytes(base_total_bytes: int, method_cfg: dict[str, Any]) -> int:
    explicit_budget = method_cfg.get("budget_bytes")
    if explicit_budget is not None:
        return max(int(explicit_budget), 0)
    budget_percent = method_cfg.get("budget_percent_of_base")
    if budget_percent is None:
        raise ValueError("Latency benchmark requires either budget_bytes or budget_percent_of_base.")
    return max(int(round(base_total_bytes * (float(budget_percent) / 100.0))), 0)


def _benchmark_prepared_model(
    prepared: PreparedBenchmarkModel,
    config: RunConfig,
    spec: LatencyBenchmarkSpec,
) -> dict[str, Any]:
    prompt_batch = build_prompt_batch(
        prepared.tokenizer,
        prompt_template=spec.prompt_template,
        prompt_length=spec.prompt_length,
        batch_size=spec.batch_size,
        device=prepared.runtime.device,
    )
    generate_kwargs = {
        "input_ids": prompt_batch["input_ids"],
        "attention_mask": prompt_batch["attention_mask"],
        "max_new_tokens": spec.decode_length,
        "min_new_tokens": spec.decode_length,
        "do_sample": False,
        "use_cache": True,
    }
    if prepared.tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = prepared.tokenizer.pad_token_id
    if prepared.tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = prepared.tokenizer.eos_token_id

    first_token_kwargs = dict(generate_kwargs)
    first_token_kwargs["max_new_tokens"] = 1
    first_token_kwargs["min_new_tokens"] = 1

    for _ in range(spec.warmup_iterations):
        _run_generate(prepared.model, generate_kwargs, prepared.runtime.device)
        _run_generate(prepared.model, first_token_kwargs, prepared.runtime.device)

    repetitions: list[dict[str, float]] = []
    for _ in range(spec.timed_iterations):
        if prepared.runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(prepared.runtime.device)

        first_elapsed, first_generated_tokens, _ = _run_generate(
            prepared.model,
            first_token_kwargs,
            prepared.runtime.device,
        )
        total_elapsed, total_generated_tokens, _ = _run_generate(
            prepared.model,
            generate_kwargs,
            prepared.runtime.device,
        )
        if total_generated_tokens != spec.batch_size * spec.decode_length:
            raise RuntimeError(
                f"Expected {spec.batch_size * spec.decode_length} generated tokens for benchmark "
                f"{config.run_id}, but observed {total_generated_tokens}."
            )

        steady_tokens = max(total_generated_tokens - first_generated_tokens, 1)
        decode_elapsed = max(total_elapsed - first_elapsed, 1e-9)
        end_to_end_tokens_per_sec = total_generated_tokens / max(total_elapsed, 1e-9)
        decode_tokens_per_sec = steady_tokens / decode_elapsed
        peak_vram_bytes = _read_peak_vram_bytes(prepared.runtime.device)

        repetitions.append(
            {
                "first_token_latency_ms": 1000.0 * first_elapsed,
                "end_to_end_elapsed_s": total_elapsed,
                "decode_elapsed_s": decode_elapsed,
                "end_to_end_tokens_per_sec": end_to_end_tokens_per_sec,
                "decode_tokens_per_sec": decode_tokens_per_sec,
                "decode_ms_per_token": 1000.0 / decode_tokens_per_sec,
                "peak_vram_mb": peak_vram_bytes / (1024 * 1024),
                "generated_tokens": float(total_generated_tokens),
            }
        )

    summary = _summarize_repetitions(repetitions)
    return {
        "run_id": config.run_id,
        "model_name": config.model_name,
        "tokenizer_name": config.tokenizer_name,
        "method": config.method_name,
        "benchmark_spec": asdict(spec),
        "device": prepared.runtime.device_label,
        "dtype": str(prepared.runtime.dtype).replace("torch.", ""),
        "policy_metadata": prepared.metadata,
        "prompt_template": spec.prompt_template,
        "summary": summary,
        "repetitions": repetitions,
        "status": "completed",
    }


def build_prompt_batch(
    tokenizer: Any,
    *,
    prompt_template: str,
    prompt_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a deterministic fixed-length prompt batch."""

    encoded = tokenizer(prompt_template, add_special_tokens=False, truncation=False)
    input_ids = list(encoded["input_ids"])
    if not input_ids:
        raise ValueError("Prompt template produced no token ids.")

    repeated_ids: list[int] = []
    while len(repeated_ids) < prompt_length:
        repeated_ids.extend(input_ids)
        if tokenizer.eos_token_id is not None:
            repeated_ids.append(int(tokenizer.eos_token_id))
    repeated_ids = repeated_ids[:prompt_length]

    prompt_tensor = torch.tensor(repeated_ids, dtype=torch.long)
    input_batch = prompt_tensor.unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones_like(input_batch, dtype=torch.long, device=device)
    return {
        "input_ids": input_batch,
        "attention_mask": attention_mask,
    }


def _run_generate(
    model: nn.Module,
    generate_kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[float, int, torch.Tensor]:
    with torch.no_grad():
        _synchronize_if_cuda(device)
        start = time.perf_counter()
        outputs = model.generate(**generate_kwargs)
        _synchronize_if_cuda(device)
        elapsed = time.perf_counter() - start
    input_len = int(generate_kwargs["input_ids"].shape[1])
    generated_tokens = int(outputs.shape[1] - input_len) * int(outputs.shape[0])
    return elapsed, generated_tokens, outputs


def _read_peak_vram_bytes(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize_repetitions(repetitions: list[dict[str, float]]) -> dict[str, Any]:
    def _metric_summary(key: str) -> dict[str, float]:
        values = [float(item[key]) for item in repetitions]
        summary = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
        summary["std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        return summary

    return {
        "first_token_latency_ms": _metric_summary("first_token_latency_ms"),
        "decode_tokens_per_sec": _metric_summary("decode_tokens_per_sec"),
        "decode_ms_per_token": _metric_summary("decode_ms_per_token"),
        "end_to_end_tokens_per_sec": _metric_summary("end_to_end_tokens_per_sec"),
        "peak_vram_mb": _metric_summary("peak_vram_mb"),
        "generated_tokens": _metric_summary("generated_tokens"),
    }
