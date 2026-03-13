from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from llm_decomposition.config import RunConfig
from llm_decomposition.hf_utils import (
    build_fixed_length_sequences,
    evaluate_perplexity,
    load_causal_lm,
    load_text_split,
    load_tokenizer,
    resolve_runtime_context,
)
from llm_decomposition.io import write_json
from llm_decomposition.profiling import (
    measure_activation_error,
    merge_layer_metrics,
    profile_residual_svd,
    summarize_layer_errors,
)
from llm_decomposition.quantization import (
    apply_uniform_svd_repair,
    quantize_linear_weight,
    quantize_model_mixed_precision,
    quantize_model_rtn,
)


def execute_full_precision(root: Path, config: RunConfig) -> dict[str, Any]:
    print(f"[{config.run_id}] loading tokenizer and full-precision model")
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    tokenizer = load_tokenizer(config.model_name)
    model = load_causal_lm(config.model_name, runtime)

    print(f"[{config.run_id}] loading evaluation split and building sequences")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating full-precision perplexity on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(model, eval_sequences, runtime.device)

    metrics = {
        "run_id": config.run_id,
        "status": "completed",
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": config.bit_width,
        "device": runtime.device_label,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "memory_total_bytes": _full_precision_memory_bytes(model),
        "memory_metadata_bytes": 0,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
    }
    _write_outputs(root, config, metrics, [], [])
    return metrics


def execute_rtn(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    tokenizer = load_tokenizer(config.model_name)

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)

    method_cfg = config.raw["method"]
    print(f"[{config.run_id}] quantizing model with RTN {method_cfg['bit_width']}-bit")
    quantized_model, layer_stats = quantize_model_rtn(
        fp_model,
        bit_width=method_cfg["bit_width"],
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
    )
    quantized_model.to(runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating quantized model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling layerwise errors")
    merged_layer_metrics, residual_profiles = _profile_model_pair(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
    )

    memory_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
    memory_metadata_bytes = sum(item["metadata_bytes"] for item in layer_stats.values())
    metrics = {
        "run_id": config.run_id,
        "status": "completed",
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": config.bit_width,
        "device": runtime.device_label,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "memory_total_bytes": memory_total_bytes,
        "memory_metadata_bytes": memory_metadata_bytes,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    return metrics


def execute_gptq(root: Path, config: RunConfig) -> dict[str, Any]:
    from llm_decomposition.gptq_backend import estimate_gptq_layer_stats, quantize_model_gptq

    runtime = _resolve_gptq_runtime(config)
    tokenizer = load_tokenizer(config.model_name)
    method_cfg = config.raw["method"]
    print(
        f"[{config.run_id}] GPTQ runtime "
        f"device={runtime.device_label} dtype={str(runtime.dtype).replace('torch.', '')}"
    )

    print(f"[{config.run_id}] loading full-precision reference model")
    fp_model = load_causal_lm(config.model_name, runtime)
    layer_stats = estimate_gptq_layer_stats(
        model=fp_model,
        bit_width=method_cfg["bit_width"],
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
    )

    print(f"[{config.run_id}] loading GPTQ source model")
    gptq_source_model = load_causal_lm(config.model_name, runtime)

    print(f"[{config.run_id}] quantizing model with GPTQ {method_cfg['bit_width']}-bit")
    quantized_model, saved_artifact_bytes = quantize_model_gptq(gptq_source_model, config, tokenizer)
    quantized_model.to(runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating quantized model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling layerwise errors")
    merged_layer_metrics, residual_profiles = _profile_model_pair(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
    )

    memory_metadata_bytes = sum(item["metadata_bytes"] for item in layer_stats.values())
    metrics = {
        "run_id": config.run_id,
        "status": "completed",
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": config.bit_width,
        "device": runtime.device_label,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "memory_total_bytes": saved_artifact_bytes,
        "memory_metadata_bytes": memory_metadata_bytes,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    return metrics


def execute_uniform_svd_repair(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    tokenizer = load_tokenizer(config.model_name)
    method_cfg = config.raw["method"]

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)

    print(f"[{config.run_id}] building RTN base model")
    quantized_model, layer_stats = quantize_model_rtn(
        fp_model,
        bit_width=method_cfg.get("base_bit_width", 4),
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
    )

    selected_layers = _load_selected_layers(root, method_cfg)
    print(
        f"[{config.run_id}] applying uniform SVD repair rank={method_cfg['rank']} "
        f"to {len(selected_layers)} layers"
    )
    apply_uniform_svd_repair(
        fp_model=fp_model,
        quantized_model=quantized_model,
        layer_stats=layer_stats,
        layer_names=selected_layers,
        rank=method_cfg["rank"],
        factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", 2),
    )
    quantized_model.to(runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating repaired model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling repaired model")
    merged_layer_metrics, residual_profiles = _profile_model_pair(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
        target_layers=selected_layers,
    )

    memory_total_bytes = sum(item.get("total_effective_bytes", item["total_quantized_bytes"]) for item in layer_stats.values())
    memory_metadata_bytes = sum(item["metadata_bytes"] for item in layer_stats.values())
    metrics = {
        "run_id": config.run_id,
        "status": "completed",
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": method_cfg.get("base_bit_width", 4),
        "device": runtime.device_label,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "memory_total_bytes": memory_total_bytes,
        "memory_metadata_bytes": memory_metadata_bytes,
        "repair_factor_bytes": sum(item.get("repair_factor_bytes", 0) for item in layer_stats.values()),
        "selected_layers": selected_layers,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    return metrics


def execute_mixed_precision_budget_match(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    tokenizer = load_tokenizer(config.model_name)
    method_cfg = config.raw["method"]

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)
    selected_layers = _load_selected_layers(root, method_cfg)
    target_total_bytes = _load_target_memory_bytes(root, method_cfg["memory_match_run_id"])

    print(
        f"[{config.run_id}] building mixed-precision model to match {method_cfg['memory_match_run_id']} "
        f"target memory {target_total_bytes} bytes"
    )
    layer_bit_overrides = _build_layer_bit_overrides_for_budget(
        fp_model=fp_model,
        selected_layers=selected_layers,
        default_bit_width=method_cfg.get("base_bit_width", 4),
        upgraded_bit_width=method_cfg.get("target_bit_width", 8),
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
        target_total_bytes=target_total_bytes,
    )

    quantized_model, layer_stats = quantize_model_mixed_precision(
        model=fp_model,
        default_bit_width=method_cfg.get("base_bit_width", 4),
        layer_bit_overrides=layer_bit_overrides,
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
    )
    quantized_model.to(runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating mixed-precision model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling mixed-precision model")
    merged_layer_metrics, residual_profiles = _profile_model_pair(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
        target_layers=selected_layers,
    )

    memory_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
    memory_metadata_bytes = sum(item["metadata_bytes"] for item in layer_stats.values())
    upgraded_layers = sorted(layer_bit_overrides)
    metrics = {
        "run_id": config.run_id,
        "status": "completed",
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": method_cfg.get("base_bit_width", 4),
        "device": runtime.device_label,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "memory_total_bytes": memory_total_bytes,
        "memory_metadata_bytes": memory_metadata_bytes,
        "upgraded_layers": upgraded_layers,
        "target_memory_match_run_id": method_cfg["memory_match_run_id"],
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    return metrics


def execute_targeted_mixed_precision(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.model_name)
    method_cfg = config.raw["method"]
    base_method = method_cfg.get("base_method", "rtn")

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)
    candidate_pool = _load_candidate_pool(root, method_cfg["candidate_pool_path"])
    layer_error_map = _load_layer_error_map(root, candidate_pool["source_layer_errors_path"])
    candidate_layers = _candidate_layers_from_pool(candidate_pool)
    budget_bytes = _resolve_budget_bytes(root, method_cfg["base_run_id"], method_cfg["budget_percent_of_base"])

    print(
        f"[{config.run_id}] building targeted mixed-precision action set "
        f"for {len(candidate_layers)} matrices with budget {budget_bytes} bytes"
    )
    actions = _build_bit_actions(
        fp_model=fp_model,
        candidate_layers=candidate_layers,
        layer_error_map=layer_error_map,
        base_bit_width=method_cfg.get("base_bit_width", 4),
        target_bit_width=candidate_pool["bit_actions"]["candidate_bit_widths"][0],
        group_size=candidate_pool.get("group_size", 128),
        symmetric=candidate_pool.get("symmetric", True),
        proxy_family=method_cfg.get("proxy_family", "activation"),
    )
    selected_actions = _select_bit_actions(actions, allocator=method_cfg.get("allocator", "greedy_activation"), budget_bytes=budget_bytes)
    layer_bit_overrides = {action["target_name"]: action["bit_to"] for action in selected_actions}

    if base_method == "gptq":
        from llm_decomposition.gptq_backend import apply_targeted_bit_upgrades

        print(
            f"[{config.run_id}] GPTQ mixed-precision runtime "
            f"device={runtime.device_label} dtype={str(runtime.dtype).replace('torch.', '')}"
        )
        print(f"[{config.run_id}] building GPTQ base model for targeted mixed-precision allocation")
        gptq_source_model = load_causal_lm(config.model_name, runtime)
        quantized_model, base_total_bytes, layer_stats = _build_gptq_base_model(
            config=config,
            tokenizer=tokenizer,
            source_model=gptq_source_model,
            fp_reference_model=fp_model,
        )
        print(f"[{config.run_id}] applying {len(selected_actions)} matrix-level bit upgrades on GPTQ base")
        upgraded_layers = apply_targeted_bit_upgrades(
            fp_model=fp_model,
            quantized_model=quantized_model,
            layer_stats=layer_stats,
            target_layers=sorted(layer_bit_overrides),
            target_bit_width=candidate_pool["bit_actions"]["candidate_bit_widths"][0],
            group_size=candidate_pool.get("group_size", 128),
            symmetric=candidate_pool.get("symmetric", True),
        )
        memory_total_bytes = base_total_bytes + sum(action["byte_cost"] for action in selected_actions)
    else:
        print(f"[{config.run_id}] applying {len(selected_actions)} matrix-level bit upgrades")
        quantized_model, layer_stats = quantize_model_mixed_precision(
            model=fp_model,
            default_bit_width=method_cfg.get("base_bit_width", 4),
            layer_bit_overrides=layer_bit_overrides,
            group_size=candidate_pool.get("group_size", 128),
            symmetric=candidate_pool.get("symmetric", True),
        )
        upgraded_layers = sorted(layer_bit_overrides)
        memory_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
    quantized_model.to(runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating targeted mixed-precision model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling targeted mixed-precision model")
    merged_layer_metrics, residual_profiles = _profile_model_pair(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
        target_layers=candidate_layers,
    )

    memory_metadata_bytes = sum(item["metadata_bytes"] for item in layer_stats.values())
    metrics = {
        "run_id": config.run_id,
        "status": "completed",
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": method_cfg.get("base_bit_width", 4),
        "device": runtime.device_label,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "memory_total_bytes": memory_total_bytes,
        "memory_metadata_bytes": memory_metadata_bytes,
        "extra_budget_bytes": budget_bytes,
        "candidate_pool_path": method_cfg["candidate_pool_path"],
        "allocator": method_cfg.get("allocator"),
        "proxy_family": method_cfg.get("proxy_family"),
        "selected_actions": selected_actions,
        "upgraded_layers": upgraded_layers,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    _write_actions(root, config, actions, selected_actions)
    return metrics


def execute_targeted_svd_rank(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.model_name)
    method_cfg = config.raw["method"]
    base_method = method_cfg.get("base_method", "rtn")

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)
    candidate_pool = _load_candidate_pool(root, method_cfg["candidate_pool_path"])
    layer_error_map = _load_layer_error_map(root, candidate_pool["source_layer_errors_path"])
    candidate_layers = _candidate_layers_from_pool(candidate_pool)
    budget_bytes = _resolve_budget_bytes(root, method_cfg["base_run_id"], method_cfg["budget_percent_of_base"])

    if base_method == "gptq":
        print(
            f"[{config.run_id}] GPTQ rank runtime "
            f"device={runtime.device_label} dtype={str(runtime.dtype).replace('torch.', '')}"
        )
        print(f"[{config.run_id}] building GPTQ base model for targeted rank allocation")
        gptq_source_model = load_causal_lm(config.model_name, runtime)
        quantized_model, base_total_bytes, layer_stats = _build_gptq_base_model(
            config=config,
            tokenizer=tokenizer,
            source_model=gptq_source_model,
            fp_reference_model=fp_model,
        )
    else:
        print(f"[{config.run_id}] building RTN base model for targeted rank allocation")
        quantized_model, layer_stats = quantize_model_rtn(
            fp_model,
            bit_width=method_cfg.get("base_bit_width", 4),
            group_size=candidate_pool.get("group_size", 128),
            symmetric=candidate_pool.get("symmetric", True),
        )
        base_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
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
    selected_actions = _select_rank_actions(actions, allocator=method_cfg.get("allocator", "greedy_activation"), budget_bytes=budget_bytes)

    selected_layer_ranks = _collapse_selected_rank_actions(selected_actions)
    print(
        f"[{config.run_id}] applying {len(selected_layer_ranks)} targeted rank repairs "
        f"with total selected rank bytes {sum(action['byte_cost'] for action in selected_actions)}"
    )
    _apply_targeted_svd_repairs(
        fp_model=fp_model,
        quantized_model=quantized_model,
        layer_stats=layer_stats,
        selected_layer_ranks=selected_layer_ranks,
        factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
    )
    quantized_model.to(runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating targeted rank model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling targeted rank model")
    merged_layer_metrics, final_residual_profiles = _profile_model_pair(
        root=root,
        config=config,
        tokenizer=tokenizer,
        runtime_device=runtime.device,
        fp_model=fp_model,
        eval_model=quantized_model,
        layer_stats=layer_stats,
        target_layers=candidate_layers,
    )

    memory_metadata_bytes = sum(item["metadata_bytes"] for item in layer_stats.values())
    repair_factor_bytes = sum(item.get("repair_factor_bytes", 0) for item in layer_stats.values())
    if base_method == "gptq":
        memory_total_bytes = base_total_bytes + repair_factor_bytes
    else:
        memory_total_bytes = sum(item.get("total_effective_bytes", item["total_quantized_bytes"]) for item in layer_stats.values())
    metrics = {
        "run_id": config.run_id,
        "status": "completed",
        "model_name": config.model_name,
        "method": config.method_name,
        "bit_width": method_cfg.get("base_bit_width", 4),
        "device": runtime.device_label,
        "dtype": str(runtime.dtype).replace("torch.", ""),
        "memory_total_bytes": memory_total_bytes,
        "memory_metadata_bytes": memory_metadata_bytes,
        "extra_budget_bytes": budget_bytes,
        "candidate_pool_path": method_cfg["candidate_pool_path"],
        "allocator": method_cfg.get("allocator"),
        "proxy_family": method_cfg.get("proxy_family"),
        "selected_actions": selected_actions,
        "selected_layer_ranks": selected_layer_ranks,
        "repair_factor_bytes": repair_factor_bytes,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, final_residual_profiles)
    _write_actions(root, config, actions, selected_actions)
    return metrics


def _profile_model_pair(
    root: Path,
    config: RunConfig,
    tokenizer,
    runtime_device,
    fp_model,
    eval_model,
    layer_stats: dict[str, dict[str, Any]],
    target_layers: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    profiling_cfg = config.raw["profiling"]
    layer_summaries = summarize_layer_errors(layer_stats)

    activation_errors = {}
    if profiling_cfg.get("layerwise_activation_error", False):
        calibration_sequences = _load_profile_sequences(tokenizer, config)
        activation_errors = measure_activation_error(
            fp_model,
            eval_model,
            calibration_sequences,
            runtime_device,
            target_layers=target_layers,
        )

    merged_layer_metrics = merge_layer_metrics(layer_summaries, activation_errors)

    residual_profiles = []
    if profiling_cfg.get("residual_svd_profile", False):
        if target_layers is None:
            top_k = profiling_cfg.get("residual_top_k_layers", 8)
            candidate_layers = [item["layer_name"] for item in merged_layer_metrics[:top_k]]
        else:
            candidate_layers = target_layers[: profiling_cfg.get("residual_top_k_layers", len(target_layers))]
        residual_profiles = profile_residual_svd(
            fp_model,
            eval_model,
            candidate_layers,
            profiling_cfg.get("candidate_ranks", [4, 8, 16, 32]),
        )
    return merged_layer_metrics, residual_profiles


def _resolve_runtime_for_config(config: RunConfig):
    method_cfg = config.raw.get("method", {})
    if config.method_name == "gptq" or method_cfg.get("base_method") == "gptq":
        return _resolve_gptq_runtime(config)
    return resolve_runtime_context(config.raw["model"].get("dtype_preference"))


def _resolve_gptq_runtime(config: RunConfig):
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    if runtime.device.type == "cuda" and runtime.dtype == torch.bfloat16:
        return type(runtime)(device=runtime.device, dtype=torch.float16, device_label=runtime.device_label)
    return runtime


def _build_gptq_base_model(
    config: RunConfig,
    tokenizer,
    source_model,
    fp_reference_model,
) -> tuple[nn.Module, int, dict[str, dict[str, Any]]]:
    from llm_decomposition.gptq_backend import estimate_gptq_layer_stats, quantize_model_gptq

    quantized_model, base_total_bytes = quantize_model_gptq(source_model, config, tokenizer)
    method_cfg = config.raw["method"]
    layer_stats = estimate_gptq_layer_stats(
        model=fp_reference_model,
        bit_width=method_cfg.get("base_bit_width", method_cfg.get("bit_width", 4)),
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
    )
    return quantized_model, base_total_bytes, layer_stats


def _load_eval_sequences(tokenizer, config: RunConfig):
    eval_cfg = config.raw["evaluation"]
    eval_dataset = load_text_split(eval_cfg["dataset"], eval_cfg["subset"], eval_cfg["split"])
    return build_fixed_length_sequences(
        tokenizer,
        eval_dataset,
        sequence_length=eval_cfg["sequence_length"],
        num_sequences=eval_cfg.get("num_sequences"),
    )


def _load_profile_sequences(tokenizer, config: RunConfig):
    profiling_cfg = config.raw["profiling"]
    calibration_cfg = config.raw["calibration"]
    calibration_dataset = load_text_split(
        calibration_cfg["dataset"],
        calibration_cfg["subset"],
        calibration_cfg["split"],
    )
    return build_fixed_length_sequences(
        tokenizer,
        calibration_dataset,
        sequence_length=profiling_cfg.get("profile_sequence_length", 128),
        num_sequences=profiling_cfg.get("profile_num_sequences", 8),
    )


def _load_selected_layers(root: Path, method_cfg: dict[str, Any]) -> list[str]:
    source_path = (root / method_cfg["selected_layers_source"]).resolve()
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    layer_errors = payload["layer_errors"]
    layer_errors = sorted(
        layer_errors,
        key=lambda item: item.get("activation_relative_l2", item.get("relative_fro_error", 0.0)),
        reverse=True,
    )
    top_k = method_cfg.get("selected_layers_top_k", len(layer_errors))
    return [item["layer_name"] for item in layer_errors[:top_k]]


def _load_candidate_pool(root: Path, rel_path: str) -> dict[str, Any]:
    with (root / rel_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_layer_error_map(root: Path, rel_path: str) -> dict[str, dict[str, Any]]:
    with (root / rel_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {item["layer_name"]: item for item in payload["layer_errors"]}


def _candidate_layers_from_pool(candidate_pool: dict[str, Any]) -> list[str]:
    candidate_layers = list(candidate_pool.get("candidate_layers", []))
    candidate_layers.extend(candidate_pool.get("control_layers", []))
    return candidate_layers


def _load_target_memory_bytes(root: Path, run_id: str) -> int:
    candidate_paths = [
        root / f"results/phase1/{run_id}/metrics.json",
        root / f"results/phase2/{run_id}/metrics.json",
        root / f"results/qwen3_1p7b/{run_id.replace('_Q17B', '')}/metrics.json",
        root / f"results/modal/qwen3_1p7b_baselines/{run_id}/metrics.json",
        root / f"results/modal/qwen3_1p7b_transfer/{run_id}/metrics.json",
        root / f"results/modal/phase2/{run_id}/metrics.json",
    ]

    for metrics_path in candidate_paths:
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        memory_total_bytes = payload.get("memory_total_bytes")
        if memory_total_bytes is not None:
            return int(memory_total_bytes)

    searched = "\n".join(path.as_posix() for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not locate a metrics.json with memory_total_bytes for run_id='{run_id}'. "
        f"Searched:\n{searched}"
    )


def _resolve_budget_bytes(root: Path, base_run_id: str, budget_percent_of_base: float) -> int:
    base_total_bytes = _load_target_memory_bytes(root, base_run_id)
    return max(int(round(base_total_bytes * (budget_percent_of_base / 100.0))), 0)


def _build_layer_bit_overrides_for_budget(
    fp_model,
    selected_layers: list[str],
    default_bit_width: int,
    upgraded_bit_width: int,
    group_size: int,
    symmetric: bool,
    target_total_bytes: int,
) -> dict[str, int]:
    fp_modules = dict(fp_model.named_modules())
    base_total_bytes = 0
    per_layer_upgrade_costs: dict[str, int] = {}

    for name, module in fp_modules.items():
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach()
        _, base_stats = quantize_linear_weight(
            weight.to(torch.float32),
            bit_width=default_bit_width,
            group_size=group_size,
            symmetric=symmetric,
        )
        base_total_bytes += base_stats["total_quantized_bytes"]
        if name in selected_layers:
            _, upgraded_stats = quantize_linear_weight(
                weight.to(torch.float32),
                bit_width=upgraded_bit_width,
                group_size=group_size,
                symmetric=symmetric,
            )
            extra_cost = upgraded_stats["total_quantized_bytes"] - base_stats["total_quantized_bytes"]
            per_layer_upgrade_costs[name] = extra_cost

    remaining_budget = max(target_total_bytes - base_total_bytes, 0)
    overrides: dict[str, int] = {}
    for layer_name in selected_layers:
        extra_cost = per_layer_upgrade_costs.get(layer_name)
        if extra_cost is None:
            continue
        if extra_cost <= remaining_budget:
            overrides[layer_name] = upgraded_bit_width
            remaining_budget -= extra_cost
    return overrides


def _build_bit_actions(
    fp_model,
    candidate_layers: list[str],
    layer_error_map: dict[str, dict[str, Any]],
    base_bit_width: int,
    target_bit_width: int,
    group_size: int,
    symmetric: bool,
    proxy_family: str,
) -> list[dict[str, Any]]:
    fp_modules = dict(fp_model.named_modules())
    actions = []
    for layer_name in candidate_layers:
        module = fp_modules.get(layer_name)
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach().to(torch.float32)
        _, base_stats = quantize_linear_weight(weight, bit_width=base_bit_width, group_size=group_size, symmetric=symmetric)
        _, target_stats = quantize_linear_weight(weight, bit_width=target_bit_width, group_size=group_size, symmetric=symmetric)
        extra_cost = target_stats["total_quantized_bytes"] - base_stats["total_quantized_bytes"]
        if extra_cost <= 0:
            continue
        proxy_score = _action_proxy_score(layer_error_map.get(layer_name, {}), proxy_family)
        actions.append(
            {
                "action_id": f"bit_{layer_name.replace('.', '_')}_{base_bit_width}to{target_bit_width}",
                "action_type": "bit_upgrade",
                "target_granularity": "matrix",
                "target_name": layer_name,
                "byte_cost": extra_cost,
                "proxy_family": proxy_family,
                "proxy_score": proxy_score,
                "predicted_gain_per_byte": proxy_score / max(extra_cost, 1),
                "bit_from": base_bit_width,
                "bit_to": target_bit_width,
            }
        )
    return actions


def _build_rank_actions(
    fp_model,
    candidate_layers: list[str],
    layer_error_map: dict[str, dict[str, Any]],
    residual_profiles: list[dict[str, Any]],
    candidate_ranks: list[int],
    factor_dtype_bytes: int,
    proxy_family: str,
) -> list[dict[str, Any]]:
    fp_modules = dict(fp_model.named_modules())
    profile_map = {item["layer_name"]: item for item in residual_profiles}
    actions = []
    for layer_name in candidate_layers:
        module = fp_modules.get(layer_name)
        if not isinstance(module, nn.Linear):
            continue
        rows, cols = module.weight.shape
        proxy_score = _action_proxy_score(layer_error_map.get(layer_name, {}), proxy_family)
        rank_energy = profile_map.get(layer_name, {}).get("rank_energy", {})
        previous_rank = 0
        previous_energy = 0.0
        for rank in sorted(candidate_ranks):
            energy_multiplier = float(rank_energy.get(str(rank), 0.0))
            delta_rank = rank - previous_rank
            delta_energy = max(energy_multiplier - previous_energy, 0.0)
            extra_cost = (rows * delta_rank + delta_rank * cols) * factor_dtype_bytes
            predicted_score = proxy_score * delta_energy
            if delta_rank <= 0 or extra_cost <= 0:
                previous_rank = rank
                previous_energy = energy_multiplier
                continue
            actions.append(
                {
                    "action_id": f"rank_{layer_name.replace('.', '_')}_r{previous_rank}_to_r{rank}",
                    "action_type": "rank_repair",
                    "target_granularity": "matrix",
                    "target_name": layer_name,
                    "byte_cost": extra_cost,
                    "proxy_family": proxy_family,
                    "proxy_score": predicted_score,
                    "predicted_gain_per_byte": predicted_score / max(extra_cost, 1),
                    "rank": rank,
                    "rank_from": previous_rank,
                    "rank_to": rank,
                    "rank_delta": delta_rank,
                }
            )
            previous_rank = rank
            previous_energy = energy_multiplier
    return actions


def _action_proxy_score(layer_error: dict[str, Any], proxy_family: str) -> float:
    if proxy_family == "activation":
        return float(layer_error.get("activation_relative_l2", layer_error.get("relative_fro_error", 0.0)))
    if proxy_family == "weight":
        return float(layer_error.get("relative_fro_error", 0.0))
    return float(layer_error.get("activation_relative_l2", layer_error.get("relative_fro_error", 0.0)))


def _select_bit_actions(actions: list[dict[str, Any]], allocator: str, budget_bytes: int) -> list[dict[str, Any]]:
    ordered = list(actions)
    if allocator == "greedy_activation":
        ordered.sort(key=lambda item: item["predicted_gain_per_byte"], reverse=True)
    selected = []
    used = 0
    for action in ordered:
        if used + action["byte_cost"] > budget_bytes:
            continue
        selected.append(_selected_action_payload(action, selection_order=len(selected) + 1, cumulative_budget=used + action["byte_cost"]))
        used += action["byte_cost"]
    return selected


def _select_rank_actions(actions: list[dict[str, Any]], allocator: str, budget_bytes: int) -> list[dict[str, Any]]:
    if allocator == "uniform_rank":
        candidate_ranks = sorted({int(action["rank_to"]) for action in actions})
        best_selection: list[dict[str, Any]] = []
        best_score = -1.0
        for rank in candidate_ranks:
            per_layer_sequences = _build_uniform_rank_sequences(actions, target_rank=rank)
            ordered_sequences = sorted(per_layer_sequences, key=lambda item: item["total_score"], reverse=True)
            chosen_sequences = []
            used = 0
            total_score = 0.0
            for sequence in ordered_sequences:
                if used + sequence["total_cost"] > budget_bytes:
                    continue
                chosen_sequences.append(sequence)
                total_score += sequence["total_score"]
                used += sequence["total_cost"]
            if total_score > best_score:
                best_score = total_score
                best_selection = [action for sequence in chosen_sequences for action in sequence["actions"]]
        selected_payloads = []
        used = 0
        for index, action in enumerate(best_selection, start=1):
            used += action["byte_cost"]
            selected_payloads.append(_selected_action_payload(action, selection_order=index, cumulative_budget=used))
        return selected_payloads

    return _select_rank_actions_incremental(actions, budget_bytes)


def _selected_action_payload(action: dict[str, Any], selection_order: int, cumulative_budget: int) -> dict[str, Any]:
    selected = dict(action)
    selected["selected"] = True
    selected["selection_order"] = selection_order
    selected["cumulative_budget_bytes"] = cumulative_budget
    selected["status"] = "selected"
    return selected


def _select_rank_actions_incremental(actions: list[dict[str, Any]], budget_bytes: int) -> list[dict[str, Any]]:
    actions_by_layer: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        actions_by_layer.setdefault(action["target_name"], []).append(action)
    for layer_actions in actions_by_layer.values():
        layer_actions.sort(key=lambda item: int(item["rank_to"]))

    current_rank_by_layer = {layer: 0 for layer in actions_by_layer}
    selected: list[dict[str, Any]] = []
    used = 0

    while True:
        available: list[dict[str, Any]] = []
        for layer_name, layer_actions in actions_by_layer.items():
            current_rank = current_rank_by_layer[layer_name]
            for action in layer_actions:
                if int(action["rank_from"]) == current_rank:
                    available.append(action)
                    break

        fitting = [action for action in available if used + action["byte_cost"] <= budget_bytes]
        if not fitting:
            break

        best_action = max(fitting, key=lambda item: item["predicted_gain_per_byte"])
        used += best_action["byte_cost"]
        current_rank_by_layer[best_action["target_name"]] = int(best_action["rank_to"])
        selected.append(
            _selected_action_payload(
                best_action,
                selection_order=len(selected) + 1,
                cumulative_budget=used,
            )
        )

    return selected


def _build_uniform_rank_sequences(actions: list[dict[str, Any]], target_rank: int) -> list[dict[str, Any]]:
    actions_by_layer: dict[str, list[dict[str, Any]]] = {}
    for action in actions:
        actions_by_layer.setdefault(action["target_name"], []).append(action)

    sequences = []
    for layer_name, layer_actions in actions_by_layer.items():
        prefix = [action for action in sorted(layer_actions, key=lambda item: int(item["rank_to"])) if int(action["rank_to"]) <= target_rank]
        if not prefix:
            continue
        if int(prefix[-1]["rank_to"]) != target_rank:
            continue
        sequences.append(
            {
                "layer_name": layer_name,
                "actions": prefix,
                "total_cost": sum(action["byte_cost"] for action in prefix),
                "total_score": sum(action["proxy_score"] for action in prefix),
            }
        )
    return sequences


def _collapse_selected_rank_actions(selected_actions: list[dict[str, Any]]) -> dict[str, int]:
    selected_layer_ranks: dict[str, int] = {}
    for action in selected_actions:
        selected_layer_ranks[action["target_name"]] = int(action.get("rank_to", action["rank"]))
    return selected_layer_ranks


def _apply_targeted_svd_repairs(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    layer_stats: dict[str, dict[str, Any]],
    selected_layer_ranks: dict[str, int],
    factor_dtype_bytes: int,
) -> None:
    from llm_decomposition.quantization import compute_low_rank_repair

    fp_modules = dict(fp_model.named_modules())
    quant_modules = dict(quantized_model.named_modules())
    for layer_name, rank in selected_layer_ranks.items():
        if layer_name not in fp_modules or layer_name not in quant_modules:
            continue
        fp_weight = fp_modules[layer_name].weight.detach().to(torch.float32)
        quant_weight = quant_modules[layer_name].weight.detach().to(torch.float32)
        repaired_weight, stats = compute_low_rank_repair(
            fp_weight=fp_weight,
            quant_weight=quant_weight,
            rank=rank,
            factor_dtype_bytes=factor_dtype_bytes,
        )
        quant_modules[layer_name].weight.data.copy_(repaired_weight.to(quant_modules[layer_name].weight.dtype))
        base_stats = layer_stats[layer_name]
        updated_stats = dict(base_stats)
        updated_stats.update(stats)
        updated_stats["total_effective_bytes"] = base_stats["total_quantized_bytes"] + stats["repair_factor_bytes"]
        layer_stats[layer_name] = updated_stats
    for stats in layer_stats.values():
        if "total_effective_bytes" not in stats:
            stats["repair_factor_bytes"] = 0
            stats["repair_rank"] = 0
            stats["total_effective_bytes"] = stats["total_quantized_bytes"]


def _write_outputs(
    root: Path,
    config: RunConfig,
    metrics: dict[str, Any],
    layer_errors: list[dict[str, Any]],
    residual_profiles: list[dict[str, Any]],
) -> None:
    output_cfg = config.raw["outputs"]
    run_dir = root / output_cfg["results_dir"]
    write_json(run_dir / output_cfg.get("metrics_file", "metrics.json"), metrics)
    write_json(
        run_dir / output_cfg.get("layer_summary_file", "layer_errors.json"),
        {
            "run_id": config.run_id,
            "status": "completed",
            "layer_errors": layer_errors,
        },
    )
    write_json(
        run_dir / output_cfg.get("residual_profile_file", "residual_profiles.json"),
        {
            "run_id": config.run_id,
            "status": "completed",
            "profiles": residual_profiles,
        },
    )


def _write_actions(
    root: Path,
    config: RunConfig,
    actions: list[dict[str, Any]],
    selected_actions: list[dict[str, Any]],
) -> None:
    run_dir = root / config.raw["outputs"]["results_dir"]
    write_json(
        run_dir / "actions.json",
        {
            "run_id": config.run_id,
            "status": "completed",
            "actions": actions,
            "selected_actions": selected_actions,
        },
    )


def _full_precision_memory_bytes(model) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
