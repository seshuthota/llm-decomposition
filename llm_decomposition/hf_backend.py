from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn

from llm_decomposition.actions import ActionRecord
from llm_decomposition.config import RunConfig
from llm_decomposition.hf_utils import (
    build_fixed_length_sequences,
    evaluate_perplexity,
    load_causal_lm,
    load_text_split,
    load_tokenizer,
    resolve_runtime_context,
    validate_finite_outputs,
)
from llm_decomposition.io import write_json
from llm_decomposition.profiling import (
    extract_aligned_module_weight,
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
    tokenizer = load_tokenizer(config.tokenizer_name)
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
    _maybe_run_downstream(root, config, model, tokenizer, runtime.device, metrics)
    return metrics


def execute_rtn(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    tokenizer = load_tokenizer(config.tokenizer_name)

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
    if _should_sequential_offload(config):
        print(f"[{config.run_id}] offloading full-precision reference model to CPU before evaluation")
        fp_model.to("cpu")
        _clear_cuda_cache()
    _move_model_to_runtime_device(quantized_model, runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating quantized model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling layerwise errors")
    merged_layer_metrics, residual_profiles, profiling_wall_time_s = _profile_model_pair(
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
        "profiling_wall_time_s": profiling_wall_time_s,
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    _maybe_run_downstream(root, config, quantized_model, tokenizer, runtime.device, metrics)
    return metrics


def execute_gptq(root: Path, config: RunConfig) -> dict[str, Any]:
    from llm_decomposition.gptq_backend import estimate_gptq_layer_stats, quantize_model_gptq

    runtime = _resolve_gptq_runtime(config)
    tokenizer = load_tokenizer(config.tokenizer_name)
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
    quantized_model = quantize_model_gptq(
        gptq_source_model,
        config,
        tokenizer,
        model_name=config.model_name,
        runtime=runtime,
    )
    _move_model_to_runtime_device(quantized_model, runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    _validate_gptq_outputs(root, config, quantized_model, eval_sequences, runtime.device)
    print(f"[{config.run_id}] evaluating quantized model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling layerwise errors")
    merged_layer_metrics, residual_profiles, profiling_wall_time_s = _profile_model_pair(
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
        "memory_total_bytes": sum(item["total_quantized_bytes"] for item in layer_stats.values()),
        "memory_metadata_bytes": memory_metadata_bytes,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
        "profiling_wall_time_s": profiling_wall_time_s,
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    _maybe_run_downstream(root, config, quantized_model, tokenizer, runtime.device, metrics)
    return metrics


def execute_uniform_svd_repair(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    tokenizer = load_tokenizer(config.tokenizer_name)
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
    _move_model_to_runtime_device(quantized_model, runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating repaired model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling repaired model")
    merged_layer_metrics, residual_profiles, profiling_wall_time_s = _profile_model_pair(
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
        "profiling_wall_time_s": profiling_wall_time_s,
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    _maybe_run_downstream(root, config, quantized_model, tokenizer, runtime.device, metrics)
    return metrics


def execute_mixed_precision_budget_match(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = resolve_runtime_context(config.raw["model"].get("dtype_preference"))
    tokenizer = load_tokenizer(config.tokenizer_name)
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
    _move_model_to_runtime_device(quantized_model, runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    print(f"[{config.run_id}] evaluating mixed-precision model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling mixed-precision model")
    merged_layer_metrics, residual_profiles, profiling_wall_time_s = _profile_model_pair(
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
        "profiling_wall_time_s": profiling_wall_time_s,
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    _maybe_run_downstream(root, config, quantized_model, tokenizer, runtime.device, metrics)
    return metrics


def execute_targeted_mixed_precision(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.tokenizer_name)
    method_cfg = config.raw["method"]
    base_method = method_cfg.get("base_method", "rtn")

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)
    candidate_pool = _load_candidate_pool(root, method_cfg["candidate_pool_path"])
    candidate_layers = _candidate_layers_from_pool(candidate_pool)
    budget_bytes = _resolve_budget_bytes(root, method_cfg)
    candidate_bit_widths = _resolve_candidate_bit_widths(candidate_pool, method_cfg)
    if not candidate_bit_widths:
        raise ValueError("No candidate bit widths are available for targeted mixed precision.")

    selection_profiling_wall_time_s = 0.0

    if base_method == "gptq":
        from llm_decomposition.gptq_backend import apply_targeted_bit_upgrades

        print(
            f"[{config.run_id}] GPTQ mixed-precision runtime "
            f"device={runtime.device_label} dtype={str(runtime.dtype).replace('torch.', '')}"
        )
        print(f"[{config.run_id}] building GPTQ base model for targeted mixed-precision allocation")
        # Optimization: Move fp_model to CPU to free VRAM for GPTQ loading
        fp_model.to("cpu")
        _clear_cuda_cache()

        quantized_model, base_total_bytes, layer_stats = _build_gptq_base_model(
            config=config,
            tokenizer=tokenizer,
            source_model=None,  # Redundant for transformers_gptq_config
            fp_reference_model=fp_model,
        )
        # Move fp_model back to runtime device for profiling
        _move_model_to_runtime_device(fp_model, runtime.device)

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
        print(
            f"[{config.run_id}] building targeted mixed-precision action set "
            f"for {len(candidate_layers)} matrices with budget {budget_bytes} bytes"
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
        print(f"[{config.run_id}] applying {len(selected_actions)} matrix-level bit upgrades on GPTQ base")
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
    else:
        print(f"[{config.run_id}] building RTN base model for targeted mixed-precision allocation")
        quantized_model, layer_stats = quantize_model_rtn(
            fp_model,
            bit_width=method_cfg.get("base_bit_width", 4),
            group_size=candidate_pool.get("group_size", 128),
            symmetric=candidate_pool.get("symmetric", True),
        )
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
        print(
            f"[{config.run_id}] building targeted mixed-precision action set "
            f"for {len(candidate_layers)} matrices with budget {budget_bytes} bytes"
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
        layer_bit_overrides = {action.target_name: action.bit_to for action in selected_actions if action.target_granularity == "matrix"}
        if all(action.target_granularity == "matrix" for action in selected_actions):
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
        else:
            print(f"[{config.run_id}] applying {len(selected_actions)} targeted bit actions")
            quantized_model, layer_stats = quantize_model_rtn(
                fp_model,
                bit_width=method_cfg.get("base_bit_width", 4),
                group_size=candidate_pool.get("group_size", 128),
                symmetric=candidate_pool.get("symmetric", True),
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
            memory_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
    if _should_sequential_offload(config):
        print(f"[{config.run_id}] offloading full-precision reference model to CPU before evaluation")
        fp_model.to("cpu")
        _clear_cuda_cache()
    _move_model_to_runtime_device(quantized_model, runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    if base_method == "gptq":
        _validate_gptq_outputs(root, config, quantized_model, eval_sequences, runtime.device)
    print(f"[{config.run_id}] evaluating targeted mixed-precision model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling targeted mixed-precision model")
    merged_layer_metrics, residual_profiles, profiling_wall_time_s = _profile_model_pair(
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
        "candidate_bit_widths": candidate_bit_widths,
        "selected_actions": _serialize_actions(selected_actions),
        "upgraded_layers": upgraded_layers,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
        "selection_profiling_wall_time_s": selection_profiling_wall_time_s,
        "profiling_wall_time_s": profiling_wall_time_s,
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, residual_profiles)
    _write_actions(root, config, actions, selected_actions)
    _maybe_run_downstream(root, config, quantized_model, tokenizer, runtime.device, metrics)
    return metrics


def execute_targeted_svd_rank(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.tokenizer_name)
    method_cfg = config.raw["method"]
    base_method = method_cfg.get("base_method", "rtn")

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)
    candidate_pool = _load_candidate_pool(root, method_cfg["candidate_pool_path"])
    candidate_layers = _candidate_layers_from_pool(candidate_pool)
    budget_bytes = _resolve_budget_bytes(root, method_cfg)
    selection_profiling_wall_time_s = 0.0

    if base_method == "gptq":
        print(
            f"[{config.run_id}] GPTQ rank runtime "
            f"device={runtime.device_label} dtype={str(runtime.dtype).replace('torch.', '')}"
        )
        print(f"[{config.run_id}] building GPTQ base model for targeted rank allocation")
        # Optimization: Move fp_model to CPU to free VRAM for GPTQ loading
        fp_model.to("cpu")
        _clear_cuda_cache()

        quantized_model, base_total_bytes, layer_stats = _build_gptq_base_model(
            config=config,
            tokenizer=tokenizer,
            source_model=None,  # Redundant for transformers_gptq_config
            fp_reference_model=fp_model,
        )
        # Move fp_model back to runtime device for profiling
        _move_model_to_runtime_device(fp_model, runtime.device)
    else:
        print(f"[{config.run_id}] building RTN base model for targeted rank allocation")
        quantized_model, layer_stats = quantize_model_rtn(
            fp_model,
            bit_width=method_cfg.get("base_bit_width", 4),
            group_size=candidate_pool.get("group_size", 128),
            symmetric=candidate_pool.get("symmetric", True),
        )
        base_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
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
    target_granularity = method_cfg.get("target_granularity", "matrix")
    candidate_ranks = method_cfg.get("candidate_ranks", candidate_pool["rank_actions"]["candidate_ranks"])
    if target_granularity == "row_block":
        actions = _build_row_block_rank_actions(
            fp_model=fp_model,
            eval_model=quantized_model,
            candidate_layers=candidate_layers,
            layer_error_map=layer_error_map,
            candidate_ranks=candidate_ranks,
            factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
            proxy_family=method_cfg.get("proxy_family", "activation"),
            row_block_size=method_cfg.get("row_block_size"),
        )
    elif target_granularity == "column_block":
        actions = _build_column_block_rank_actions(
            fp_model=fp_model,
            eval_model=quantized_model,
            candidate_layers=candidate_layers,
            layer_error_map=layer_error_map,
            candidate_ranks=candidate_ranks,
            factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
            proxy_family=method_cfg.get("proxy_family", "activation"),
            column_block_size=method_cfg.get("column_block_size"),
        )
    else:
        residual_profiles = profile_residual_svd(
            fp_model,
            quantized_model,
            candidate_layers,
            candidate_ranks,
        )
        actions = _build_rank_actions(
            fp_model=fp_model,
            candidate_layers=candidate_layers,
            layer_error_map=layer_error_map,
            residual_profiles=residual_profiles,
            candidate_ranks=candidate_ranks,
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
    print(
        f"[{config.run_id}] applying {len(selected_layer_ranks)} targeted rank repairs "
        f"with total selected rank bytes {sum(action.byte_cost for action in selected_actions)}"
    )
    _apply_targeted_rank_actions(
        fp_model=fp_model,
        quantized_model=quantized_model,
        layer_stats=layer_stats,
        selected_actions=selected_actions,
        factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
    )
    if _should_sequential_offload(config):
        print(f"[{config.run_id}] offloading full-precision reference model to CPU before evaluation")
        fp_model.to("cpu")
        _clear_cuda_cache()
    _move_model_to_runtime_device(quantized_model, runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    if base_method == "gptq":
        _validate_gptq_outputs(root, config, quantized_model, eval_sequences, runtime.device)
    print(f"[{config.run_id}] evaluating targeted rank model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling targeted rank model")
    merged_layer_metrics, final_residual_profiles, profiling_wall_time_s = _profile_model_pair(
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
        "selected_actions": _serialize_actions(selected_actions),
        "selected_layer_ranks": selected_layer_ranks,
        "repair_factor_bytes": repair_factor_bytes,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
        "selection_profiling_wall_time_s": selection_profiling_wall_time_s,
        "profiling_wall_time_s": profiling_wall_time_s,
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, final_residual_profiles)
    _write_actions(root, config, actions, selected_actions)
    _maybe_run_downstream(root, config, quantized_model, tokenizer, runtime.device, metrics)
    return metrics


def execute_hybrid_second_stage(root: Path, config: RunConfig) -> dict[str, Any]:
    runtime = _resolve_runtime_for_config(config)
    tokenizer = load_tokenizer(config.tokenizer_name)
    method_cfg = config.raw["method"]
    base_method = method_cfg.get("base_method", "rtn")

    print(f"[{config.run_id}] loading full-precision model")
    fp_model = load_causal_lm(config.model_name, runtime)
    candidate_pool = _load_candidate_pool(root, method_cfg["candidate_pool_path"])
    layer_error_map = _load_layer_error_map(root, candidate_pool["source_layer_errors_path"])
    candidate_layers = _candidate_layers_from_pool(candidate_pool)
    budget_bytes = _resolve_budget_bytes(root, method_cfg)
    prior_bit_actions = _load_prior_selected_bit_actions(root, method_cfg["prior_run_id"])

    if base_method == "gptq":
        print(
            f"[{config.run_id}] GPTQ hybrid runtime "
            f"device={runtime.device_label} dtype={str(runtime.dtype).replace('torch.', '')}"
        )
        print(f"[{config.run_id}] building GPTQ base model for hybrid second stage")
        gptq_source_model = load_causal_lm(config.model_name, runtime)
        quantized_model, base_total_bytes, layer_stats = _build_gptq_base_model(
            config=config,
            tokenizer=tokenizer,
            source_model=gptq_source_model,
            fp_reference_model=fp_model,
        )
    else:
        print(f"[{config.run_id}] building RTN base model for hybrid second stage")
        quantized_model, layer_stats = quantize_model_rtn(
            fp_model,
            bit_width=method_cfg.get("base_bit_width", 4),
            group_size=candidate_pool.get("group_size", 128),
            symmetric=candidate_pool.get("symmetric", True),
        )
        base_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())

    print(
        f"[{config.run_id}] applying {len(prior_bit_actions)} prior bit actions "
        f"from {method_cfg['prior_run_id']}"
    )
    upgraded_layers = _apply_targeted_bit_actions(
        fp_model=fp_model,
        quantized_model=quantized_model,
        layer_stats=layer_stats,
        selected_actions=prior_bit_actions,
        target_bit_width=method_cfg.get(
            "target_bit_width",
            candidate_pool["bit_actions"]["candidate_bit_widths"][0],
        ),
        group_size=candidate_pool.get("group_size", 128),
        symmetric=candidate_pool.get("symmetric", True),
    )
    prior_bit_action_bytes = sum(action.byte_cost for action in prior_bit_actions)

    print(
        f"[{config.run_id}] building second-stage rank action set on top of {method_cfg['prior_run_id']} "
        f"with extra budget {budget_bytes} bytes"
    )
    target_granularity = method_cfg.get("target_granularity", "matrix")
    candidate_ranks = method_cfg.get("candidate_ranks", candidate_pool["rank_actions"]["candidate_ranks"])
    if target_granularity == "row_block":
        rank_actions = _build_row_block_rank_actions(
            fp_model=fp_model,
            eval_model=quantized_model,
            candidate_layers=candidate_layers,
            layer_error_map=layer_error_map,
            candidate_ranks=candidate_ranks,
            factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
            proxy_family=method_cfg.get("proxy_family", "activation"),
            row_block_size=method_cfg.get("row_block_size"),
        )
    elif target_granularity == "column_block":
        rank_actions = _build_column_block_rank_actions(
            fp_model=fp_model,
            eval_model=quantized_model,
            candidate_layers=candidate_layers,
            layer_error_map=layer_error_map,
            candidate_ranks=candidate_ranks,
            factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
            proxy_family=method_cfg.get("proxy_family", "activation"),
            column_block_size=method_cfg.get("column_block_size"),
        )
    else:
        residual_profiles = profile_residual_svd(
            fp_model,
            quantized_model,
            candidate_layers,
            candidate_ranks,
        )
        rank_actions = _build_rank_actions(
            fp_model=fp_model,
            candidate_layers=candidate_layers,
            layer_error_map=layer_error_map,
            residual_profiles=residual_profiles,
            candidate_ranks=candidate_ranks,
            factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
            proxy_family=method_cfg.get("proxy_family", "activation"),
        )
    selected_rank_actions = _select_rank_actions(
        rank_actions,
        allocator=method_cfg.get("allocator", "greedy_activation"),
        budget_bytes=budget_bytes,
        family_rounds=int(method_cfg.get("family_rounds", 1)),
    )
    selected_layer_ranks = _collapse_selected_rank_actions(selected_rank_actions)
    print(
        f"[{config.run_id}] applying {len(selected_layer_ranks)} hybrid second-stage rank repairs "
        f"with total selected rank bytes {sum(action.byte_cost for action in selected_rank_actions)}"
    )
    _apply_targeted_rank_actions(
        fp_model=fp_model,
        quantized_model=quantized_model,
        layer_stats=layer_stats,
        selected_actions=selected_rank_actions,
        factor_dtype_bytes=method_cfg.get("factor_dtype_bytes", candidate_pool.get("factor_dtype_bytes", 2)),
    )

    if _should_sequential_offload(config):
        print(f"[{config.run_id}] offloading full-precision reference model to CPU before evaluation")
        fp_model.to("cpu")
        _clear_cuda_cache()
    quantized_model.to(runtime.device)
    quantized_model.eval()

    print(f"[{config.run_id}] loading evaluation split")
    eval_sequences = _load_eval_sequences(tokenizer, config)
    if base_method == "gptq":
        _validate_gptq_outputs(root, config, quantized_model, eval_sequences, runtime.device)
    print(f"[{config.run_id}] evaluating hybrid second-stage model on {len(eval_sequences)} sequences")
    eval_metrics = evaluate_perplexity(quantized_model, eval_sequences, runtime.device)

    print(f"[{config.run_id}] profiling hybrid second-stage model")
    merged_layer_metrics, final_residual_profiles, profiling_wall_time_s = _profile_model_pair(
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
        memory_total_bytes = base_total_bytes + prior_bit_action_bytes + repair_factor_bytes
    else:
        memory_total_bytes = sum(
            item.get("total_effective_bytes", item["total_quantized_bytes"]) for item in layer_stats.values()
        )

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
        "prior_run_id": method_cfg["prior_run_id"],
        "prior_bit_action_count": len(prior_bit_actions),
        "prior_bit_action_bytes": prior_bit_action_bytes,
        "hybrid_base_upgraded_layers": upgraded_layers,
        "selected_actions": _serialize_actions(selected_rank_actions),
        "selected_layer_ranks": selected_layer_ranks,
        "repair_factor_bytes": repair_factor_bytes,
        "perplexity": eval_metrics["perplexity"],
        "latency_ms_per_token": eval_metrics["latency_ms_per_token"],
        "evaluated_tokens": eval_metrics["evaluated_tokens"],
        "profiling_wall_time_s": profiling_wall_time_s,
    }
    _write_outputs(root, config, metrics, merged_layer_metrics, final_residual_profiles)
    _write_actions(root, config, prior_bit_actions + rank_actions, selected_rank_actions, base_actions=prior_bit_actions)
    _maybe_run_downstream(root, config, quantized_model, tokenizer, runtime.device, metrics)
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    start_time = time.perf_counter()
    profiling_cfg = config.raw["profiling"]
    layer_summaries = summarize_layer_errors(layer_stats)

    activation_errors = {}
    if profiling_cfg.get("layerwise_activation_error", False):
        sequential_offload = profiling_cfg.get("sequential_model_offload", False)
        if sequential_offload:
            fp_model.to("cpu")
            eval_model.to("cpu")
            _clear_cuda_cache()
        calibration_sequences = _load_profile_sequences(tokenizer, config)
        activation_errors = measure_activation_error(
            fp_model,
            eval_model,
            calibration_sequences,
            runtime_device,
            target_layers=target_layers,
            sequential_offload=sequential_offload,
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
    return merged_layer_metrics, residual_profiles, time.perf_counter() - start_time


def _resolve_selection_layer_error_map(
    root: Path,
    config: RunConfig,
    tokenizer,
    runtime_device,
    fp_model,
    eval_model,
    layer_stats: dict[str, dict[str, Any]],
    target_layers: list[str],
    fallback_source_path: str,
) -> tuple[dict[str, dict[str, Any]], float]:
    method_cfg = config.raw.get("method", {})
    if method_cfg.get("selection_profile_source", "candidate_pool") != "current_base_model":
        return _load_layer_error_map(root, fallback_source_path), 0.0

    start_time = time.perf_counter()
    layer_summaries = summarize_layer_errors(layer_stats)
    activation_errors = {}
    profiling_cfg = config.raw["profiling"]
    if profiling_cfg.get("layerwise_activation_error", False):
        sequential_offload = profiling_cfg.get("sequential_model_offload", False)
        if sequential_offload:
            fp_model.to("cpu")
            eval_model.to("cpu")
            _clear_cuda_cache()
        calibration_sequences = _load_profile_sequences(tokenizer, config)
        activation_errors = measure_activation_error(
            fp_model,
            eval_model,
            calibration_sequences,
            runtime_device,
            target_layers=target_layers,
            sequential_offload=sequential_offload,
        )

    merged_layer_metrics = merge_layer_metrics(layer_summaries, activation_errors)
    layer_error_map = {
        item["layer_name"]: item
        for item in merged_layer_metrics
        if item["layer_name"] in set(target_layers)
    }
    return layer_error_map, time.perf_counter() - start_time


def _should_sequential_offload(config: RunConfig) -> bool:
    return bool(config.raw.get("profiling", {}).get("sequential_model_offload", False))


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_model_to_runtime_device(model, runtime_device) -> None:
    if getattr(model, "hf_device_map", None):
        return
    model.to(runtime_device)


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

    quantized_model = quantize_model_gptq(
        source_model,
        config,
        tokenizer,
        model_name=config.model_name,
        runtime=_resolve_gptq_runtime(config),
    )
    method_cfg = config.raw["method"]
    layer_stats = estimate_gptq_layer_stats(
        model=fp_reference_model,
        bit_width=method_cfg.get("base_bit_width", method_cfg.get("bit_width", 4)),
        group_size=method_cfg.get("group_size", 128),
        symmetric=method_cfg.get("symmetric", True),
    )
    base_total_bytes = sum(item["total_quantized_bytes"] for item in layer_stats.values())
    return quantized_model, base_total_bytes, layer_stats


def _validate_gptq_outputs(
    root: Path,
    config: RunConfig,
    model,
    eval_sequences,
    runtime_device,
) -> None:
    method_cfg = config.raw.get("method", {})
    if not method_cfg.get("validate_outputs", True):
        return

    validation = validate_finite_outputs(
        model,
        eval_sequences,
        runtime_device,
        max_batches=method_cfg.get("non_finite_check_batches", 1),
    )
    validation_path = root / config.results_dir / "gptq_validation.json"
    write_json(validation_path, validation)
    if not validation.get("all_finite", False):
        raise RuntimeError(
            f"GPTQ validation failed for run {config.run_id}: non-finite logits or loss detected. "
            f"See {validation_path}."
        )


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


def _load_prior_selected_bit_actions(root: Path, run_id: str) -> list[ActionRecord]:
    candidate_paths = [
        root / f"results/modal/qwen3_1p7b_gptq_transfer/{run_id}/actions.json",
        root / f"results/modal/qwen3_8b_gptq_transfer/{run_id}/actions.json",
        root / f"results/modal/smollm3_3b_gptq_transfer/{run_id}/actions.json",
        root / f"results/qwen3_1p7b_gptq/{run_id.replace('_Q17B', '')}/actions.json",
        root / f"results/qwen3_8b_gptq/{run_id.replace('_Q8B', '')}/actions.json",
        root / f"results/smollm3_3b_gptq/{run_id.replace('_S3B', '')}/actions.json",
    ]

    payload = None
    for path in candidate_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        break

    if payload is None:
        for path in sorted(root.glob("results/**/actions.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    candidate = json.load(handle)
            except Exception:
                continue
            if candidate.get("run_id") == run_id:
                payload = candidate
                break

    if payload is None:
        raise FileNotFoundError(f"Could not locate selected bit actions for prior run_id='{run_id}'.")

    selected_payloads = payload.get("selected_actions", [])
    selected_bit_payloads = [item for item in selected_payloads if item.get("action_type") == "bit_upgrade"]
    if not selected_bit_payloads:
        raise ValueError(f"Run '{run_id}' does not contain any selected bit_upgrade actions.")
    return [ActionRecord(**item) for item in selected_bit_payloads]


def _candidate_layers_from_pool(candidate_pool: dict[str, Any]) -> list[str]:
    candidate_layers = list(candidate_pool.get("candidate_layers", []))
    candidate_layers.extend(candidate_pool.get("control_layers", []))
    return candidate_layers


def _load_target_memory_bytes(root: Path, run_id: str) -> int:
    candidate_paths = [
        root / f"results/phase1/{run_id}/metrics.json",
        root / f"results/phase2/{run_id}/metrics.json",
        root / f"results/qwen3_1p7b/{run_id.replace('_Q17B', '')}/metrics.json",
        root / f"results/smollm3_3b/{run_id.replace('_S3B', '')}/metrics.json",
        root / f"results/modal/qwen3_1p7b_baselines/{run_id}/metrics.json",
        root / f"results/modal/qwen3_1p7b_transfer/{run_id}/metrics.json",
        root / f"results/modal/qwen3_8b_gptq_baselines/{run_id}/metrics.json",
        root / f"results/modal/qwen3_8b_gptq_transfer/{run_id}/metrics.json",
        root / f"results/modal/qwen3_8b_baselines/{run_id}/metrics.json",
        root / f"results/modal/qwen3_8b_transfer/{run_id}/metrics.json",
        root / f"results/modal/smollm3_3b_baselines/{run_id}/metrics.json",
        root / f"results/modal/smollm3_3b_transfer/{run_id}/metrics.json",
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

    for metrics_path in sorted(root.glob("results/**/metrics.json")):
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if payload.get("run_id") != run_id:
            continue
        memory_total_bytes = payload.get("memory_total_bytes")
        if memory_total_bytes is not None:
            return int(memory_total_bytes)

    searched = "\n".join(path.as_posix() for path in candidate_paths)
    raise FileNotFoundError(
        f"Could not locate a metrics.json with memory_total_bytes for run_id='{run_id}'. "
        f"Searched:\n{searched}"
    )


def _resolve_budget_bytes(root: Path, method_cfg: dict[str, Any]) -> int:
    explicit_budget = method_cfg.get("budget_bytes")
    if explicit_budget is not None:
        return max(int(explicit_budget), 0)
    base_run_id = method_cfg["base_run_id"]
    budget_percent_of_base = method_cfg["budget_percent_of_base"]
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
    target_bit_widths: list[int],
    group_size: int,
    symmetric: bool,
    proxy_family: str,
    target_granularity: str = "matrix",
    row_block_size: int | None = None,
    column_block_size: int | None = None,
) -> list[ActionRecord]:
    fp_modules = dict(fp_model.named_modules())
    actions: list[ActionRecord] = []
    for layer_name in candidate_layers:
        module = fp_modules.get(layer_name)
        if not isinstance(module, nn.Linear):
            continue
        weight = module.weight.detach().to(torch.float32)
        proxy_score = _action_proxy_score(layer_error_map.get(layer_name, {}), proxy_family)
        for target_bit_width in target_bit_widths:
            if target_bit_width <= base_bit_width:
                continue
            if target_granularity == "row_block":
                if not row_block_size or row_block_size <= 0:
                    raise ValueError("row_block_size must be set when target_granularity='row_block'.")
                actions.extend(
                    _build_row_block_bit_actions(
                        layer_name=layer_name,
                        weight=weight,
                        base_bit_width=base_bit_width,
                        target_bit_width=target_bit_width,
                        group_size=group_size,
                        symmetric=symmetric,
                        proxy_family=proxy_family,
                        layer_proxy_score=proxy_score,
                        row_block_size=row_block_size,
                    )
                )
                continue
            if target_granularity == "column_block":
                if not column_block_size or column_block_size <= 0:
                    raise ValueError("column_block_size must be set when target_granularity='column_block'.")
                actions.extend(
                    _build_column_block_bit_actions(
                        layer_name=layer_name,
                        weight=weight,
                        base_bit_width=base_bit_width,
                        target_bit_width=target_bit_width,
                        group_size=group_size,
                        symmetric=symmetric,
                        proxy_family=proxy_family,
                        layer_proxy_score=proxy_score,
                        column_block_size=column_block_size,
                    )
                )
                continue

            _, base_stats = quantize_linear_weight(weight, bit_width=base_bit_width, group_size=group_size, symmetric=symmetric)
            _, target_stats = quantize_linear_weight(weight, bit_width=target_bit_width, group_size=group_size, symmetric=symmetric)
            extra_cost = target_stats["total_quantized_bytes"] - base_stats["total_quantized_bytes"]
            if extra_cost <= 0:
                continue
            actions.append(
                ActionRecord(
                    action_id=f"bit_{layer_name.replace('.', '_')}_{base_bit_width}to{target_bit_width}",
                    action_type="bit_upgrade",
                    target_granularity="matrix",
                    target_name=layer_name,
                    byte_cost=extra_cost,
                    proxy_family=proxy_family,
                    proxy_score=proxy_score,
                    predicted_gain_per_byte=proxy_score / max(extra_cost, 1),
                    bit_from=base_bit_width,
                    bit_to=target_bit_width,
                    metadata={
                        "group_size": group_size,
                        "symmetric": symmetric,
                        "quantized_weight_byte_delta": target_stats["quantized_weight_bytes"] - base_stats["quantized_weight_bytes"],
                        "metadata_byte_delta": target_stats["metadata_bytes"] - base_stats["metadata_bytes"],
                    },
                )
            )
    return actions


def _build_row_block_bit_actions(
    layer_name: str,
    weight: torch.Tensor,
    base_bit_width: int,
    target_bit_width: int,
    group_size: int,
    symmetric: bool,
    proxy_family: str,
    layer_proxy_score: float,
    row_block_size: int,
) -> list[ActionRecord]:
    actions: list[ActionRecord] = []
    rows, _ = weight.shape
    for row_start in range(0, rows, row_block_size):
        row_end = min(row_start + row_block_size, rows)
        block = weight[row_start:row_end, :]
        _, base_stats = quantize_linear_weight(block, bit_width=base_bit_width, group_size=group_size, symmetric=symmetric)
        _, target_stats = quantize_linear_weight(block, bit_width=target_bit_width, group_size=group_size, symmetric=symmetric)
        extra_cost = target_stats["total_quantized_bytes"] - base_stats["total_quantized_bytes"]
        if extra_cost <= 0:
            continue
        local_error_gain = max(base_stats["fro_error"] - target_stats["fro_error"], 0.0)
        proxy_score = layer_proxy_score * local_error_gain
        actions.append(
            ActionRecord(
                action_id=(
                    f"bitblock_{layer_name.replace('.', '_')}_{row_start}_{row_end}_"
                    f"{base_bit_width}to{target_bit_width}"
                ),
                action_type="bit_upgrade",
                target_granularity="row_block",
                target_name=layer_name,
                byte_cost=extra_cost,
                proxy_family=proxy_family,
                proxy_score=proxy_score,
                predicted_gain_per_byte=proxy_score / max(extra_cost, 1),
                bit_from=base_bit_width,
                bit_to=target_bit_width,
                metadata={
                    "group_size": group_size,
                    "symmetric": symmetric,
                    "row_start": row_start,
                    "row_end": row_end,
                    "row_block_size": row_block_size,
                    "base_block_fro_error": base_stats["fro_error"],
                    "target_block_fro_error": target_stats["fro_error"],
                    "local_error_gain": local_error_gain,
                    "quantized_weight_byte_delta": target_stats["quantized_weight_bytes"] - base_stats["quantized_weight_bytes"],
                    "metadata_byte_delta": target_stats["metadata_bytes"] - base_stats["metadata_bytes"],
                },
            )
        )
    return actions


def _build_column_block_bit_actions(
    layer_name: str,
    weight: torch.Tensor,
    base_bit_width: int,
    target_bit_width: int,
    group_size: int,
    symmetric: bool,
    proxy_family: str,
    layer_proxy_score: float,
    column_block_size: int,
) -> list[ActionRecord]:
    actions: list[ActionRecord] = []
    _, cols = weight.shape
    for col_start in range(0, cols, column_block_size):
        col_end = min(col_start + column_block_size, cols)
        block = weight[:, col_start:col_end]
        _, base_stats = quantize_linear_weight(block, bit_width=base_bit_width, group_size=group_size, symmetric=symmetric)
        _, target_stats = quantize_linear_weight(block, bit_width=target_bit_width, group_size=group_size, symmetric=symmetric)
        extra_cost = target_stats["total_quantized_bytes"] - base_stats["total_quantized_bytes"]
        if extra_cost <= 0:
            continue
        local_error_gain = max(base_stats["fro_error"] - target_stats["fro_error"], 0.0)
        proxy_score = layer_proxy_score * local_error_gain
        actions.append(
            ActionRecord(
                action_id=(
                    f"bitcol_{layer_name.replace('.', '_')}_{col_start}_{col_end}_"
                    f"{base_bit_width}to{target_bit_width}"
                ),
                action_type="bit_upgrade",
                target_granularity="column_block",
                target_name=layer_name,
                byte_cost=extra_cost,
                proxy_family=proxy_family,
                proxy_score=proxy_score,
                predicted_gain_per_byte=proxy_score / max(extra_cost, 1),
                bit_from=base_bit_width,
                bit_to=target_bit_width,
                metadata={
                    "group_size": group_size,
                    "symmetric": symmetric,
                    "col_start": col_start,
                    "col_end": col_end,
                    "column_block_size": column_block_size,
                    "base_block_fro_error": base_stats["fro_error"],
                    "target_block_fro_error": target_stats["fro_error"],
                    "local_error_gain": local_error_gain,
                    "quantized_weight_byte_delta": target_stats["quantized_weight_bytes"] - base_stats["quantized_weight_bytes"],
                    "metadata_byte_delta": target_stats["metadata_bytes"] - base_stats["metadata_bytes"],
                },
            )
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
) -> list[ActionRecord]:
    fp_modules = dict(fp_model.named_modules())
    profile_map = {item["layer_name"]: item for item in residual_profiles}
    actions: list[ActionRecord] = []
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
                ActionRecord(
                    action_id=f"rank_{layer_name.replace('.', '_')}_r{previous_rank}_to_r{rank}",
                    action_type="rank_repair",
                    target_granularity="matrix",
                    target_name=layer_name,
                    byte_cost=extra_cost,
                    proxy_family=proxy_family,
                    proxy_score=predicted_score,
                    predicted_gain_per_byte=predicted_score / max(extra_cost, 1),
                    rank=rank,
                    rank_from=previous_rank,
                    rank_to=rank,
                    rank_delta=delta_rank,
                    metadata={
                        "factor_dtype_bytes": factor_dtype_bytes,
                        "construction": "incremental_truncated_svd",
                    },
                )
            )
            previous_rank = rank
            previous_energy = energy_multiplier
    return actions


def _build_row_block_rank_actions(
    fp_model,
    eval_model,
    candidate_layers: list[str],
    layer_error_map: dict[str, dict[str, Any]],
    candidate_ranks: list[int],
    factor_dtype_bytes: int,
    proxy_family: str,
    row_block_size: int | None,
) -> list[ActionRecord]:
    if not row_block_size or row_block_size <= 0:
        raise ValueError("row_block_size must be set when target_granularity='row_block'.")

    fp_modules = dict(fp_model.named_modules())
    eval_modules = dict(eval_model.named_modules())
    actions: list[ActionRecord] = []

    for layer_name in candidate_layers:
        fp_module = fp_modules.get(layer_name)
        eval_module = eval_modules.get(layer_name)
        if not isinstance(fp_module, nn.Linear) or eval_module is None:
            continue

        fp_weight = fp_module.weight.detach().to(torch.float32)
        eval_weight = extract_aligned_module_weight(eval_module, fp_weight)
        if eval_weight is None:
            continue
        residual = fp_weight - eval_weight
        total_residual_energy = float(residual.pow(2).sum().item())
        if total_residual_energy <= 0:
            continue

        layer_proxy_score = _action_proxy_score(layer_error_map.get(layer_name, {}), proxy_family)
        rows, cols = fp_weight.shape
        for row_start in range(0, rows, row_block_size):
            row_end = min(row_start + row_block_size, rows)
            block_residual = residual[row_start:row_end, :]
            block_energy = float(block_residual.pow(2).sum().item())
            if block_energy <= 0:
                continue
            singular_values = torch.linalg.svdvals(block_residual)
            energy = singular_values.pow(2)
            cumulative = torch.cumsum(energy, dim=0)
            previous_rank = 0
            previous_energy_ratio = 0.0
            block_proxy_score = layer_proxy_score * (block_energy / max(total_residual_energy, 1e-12))
            for rank in sorted(candidate_ranks):
                effective_rank = min(rank, singular_values.numel())
                explained = float(cumulative[effective_rank - 1].item()) if effective_rank > 0 else 0.0
                energy_ratio = explained / max(block_energy, 1e-12)
                delta_rank = effective_rank - previous_rank
                delta_energy = max(energy_ratio - previous_energy_ratio, 0.0)
                extra_cost = ((row_end - row_start) * delta_rank + delta_rank * cols) * factor_dtype_bytes
                predicted_score = block_proxy_score * delta_energy
                if delta_rank <= 0 or extra_cost <= 0:
                    previous_rank = effective_rank
                    previous_energy_ratio = energy_ratio
                    continue
                actions.append(
                    ActionRecord(
                        action_id=(
                            f"rankblock_{layer_name.replace('.', '_')}_{row_start}_{row_end}_"
                            f"r{previous_rank}_to_r{effective_rank}"
                        ),
                        action_type="rank_repair",
                        target_granularity="row_block",
                        target_name=f"{layer_name}::rows[{row_start}:{row_end}]",
                        byte_cost=extra_cost,
                        proxy_family=proxy_family,
                        proxy_score=predicted_score,
                        predicted_gain_per_byte=predicted_score / max(extra_cost, 1),
                        rank=effective_rank,
                        rank_from=previous_rank,
                        rank_to=effective_rank,
                        rank_delta=delta_rank,
                        metadata={
                            "factor_dtype_bytes": factor_dtype_bytes,
                            "construction": "row_block_truncated_svd",
                            "layer_name": layer_name,
                            "row_start": row_start,
                            "row_end": row_end,
                            "row_block_size": row_block_size,
                            "block_residual_energy": block_energy,
                            "block_energy_ratio": block_energy / max(total_residual_energy, 1e-12),
                        },
                    )
                )
                previous_rank = effective_rank
                previous_energy_ratio = energy_ratio
    return actions


def _build_column_block_rank_actions(
    fp_model,
    eval_model,
    candidate_layers: list[str],
    layer_error_map: dict[str, dict[str, Any]],
    candidate_ranks: list[int],
    factor_dtype_bytes: int,
    proxy_family: str,
    column_block_size: int | None,
) -> list[ActionRecord]:
    if not column_block_size or column_block_size <= 0:
        raise ValueError("column_block_size must be set when target_granularity='column_block'.")

    fp_modules = dict(fp_model.named_modules())
    eval_modules = dict(eval_model.named_modules())
    actions: list[ActionRecord] = []

    for layer_name in candidate_layers:
        fp_module = fp_modules.get(layer_name)
        eval_module = eval_modules.get(layer_name)
        if not isinstance(fp_module, nn.Linear) or eval_module is None:
            continue

        fp_weight = fp_module.weight.detach().to(torch.float32)
        eval_weight = extract_aligned_module_weight(eval_module, fp_weight)
        if eval_weight is None:
            continue
        residual = fp_weight - eval_weight
        total_residual_energy = float(residual.pow(2).sum().item())
        if total_residual_energy <= 0:
            continue

        layer_proxy_score = _action_proxy_score(layer_error_map.get(layer_name, {}), proxy_family)
        rows, cols = fp_weight.shape
        for col_start in range(0, cols, column_block_size):
            col_end = min(col_start + column_block_size, cols)
            block_residual = residual[:, col_start:col_end]
            block_energy = float(block_residual.pow(2).sum().item())
            if block_energy <= 0:
                continue
            singular_values = torch.linalg.svdvals(block_residual)
            energy = singular_values.pow(2)
            cumulative = torch.cumsum(energy, dim=0)
            previous_rank = 0
            previous_energy_ratio = 0.0
            block_proxy_score = layer_proxy_score * (block_energy / max(total_residual_energy, 1e-12))
            for rank in sorted(candidate_ranks):
                effective_rank = min(rank, singular_values.numel())
                explained = float(cumulative[effective_rank - 1].item()) if effective_rank > 0 else 0.0
                energy_ratio = explained / max(block_energy, 1e-12)
                delta_rank = effective_rank - previous_rank
                delta_energy = max(energy_ratio - previous_energy_ratio, 0.0)
                extra_cost = (rows * delta_rank + delta_rank * (col_end - col_start)) * factor_dtype_bytes
                predicted_score = block_proxy_score * delta_energy
                if delta_rank <= 0 or extra_cost <= 0:
                    previous_rank = effective_rank
                    previous_energy_ratio = energy_ratio
                    continue
                actions.append(
                    ActionRecord(
                        action_id=(
                            f"rankcol_{layer_name.replace('.', '_')}_{col_start}_{col_end}_"
                            f"r{previous_rank}_to_r{effective_rank}"
                        ),
                        action_type="rank_repair",
                        target_granularity="column_block",
                        target_name=f"{layer_name}::cols[{col_start}:{col_end}]",
                        byte_cost=extra_cost,
                        proxy_family=proxy_family,
                        proxy_score=predicted_score,
                        predicted_gain_per_byte=predicted_score / max(extra_cost, 1),
                        rank=effective_rank,
                        rank_from=previous_rank,
                        rank_to=effective_rank,
                        rank_delta=delta_rank,
                        metadata={
                            "factor_dtype_bytes": factor_dtype_bytes,
                            "construction": "column_block_truncated_svd",
                            "layer_name": layer_name,
                            "col_start": col_start,
                            "col_end": col_end,
                            "column_block_size": column_block_size,
                            "block_residual_energy": block_energy,
                            "block_energy_ratio": block_energy / max(total_residual_energy, 1e-12),
                        },
                    )
                )
                previous_rank = effective_rank
                previous_energy_ratio = energy_ratio
    return actions


def _action_proxy_score(layer_error: dict[str, Any], proxy_family: str) -> float:
    if proxy_family == "activation":
        return float(layer_error.get("activation_relative_l2", layer_error.get("relative_fro_error", 0.0)))
    if proxy_family == "weight":
        return float(layer_error.get("relative_fro_error", 0.0))
    return float(layer_error.get("activation_relative_l2", layer_error.get("relative_fro_error", 0.0)))


def _select_bit_actions(actions: list[ActionRecord], allocator: str, budget_bytes: int) -> list[ActionRecord]:
    ordered = list(actions)
    if allocator == "greedy_activation":
        ordered.sort(key=lambda item: item.predicted_gain_per_byte or 0.0, reverse=True)
    selected = []
    used = 0
    seen_targets: set[tuple[Any, ...]] = set()
    for action in ordered:
        if action.target_granularity == "matrix":
            target_key = (action.target_name, action.target_granularity)
        elif action.target_granularity == "row_block":
            target_key = (
                action.target_name,
                action.target_granularity,
                int(action.metadata.get("row_start", 0)),
                int(action.metadata.get("row_end", 0)),
            )
        elif action.target_granularity == "column_block":
            target_key = (
                action.target_name,
                action.target_granularity,
                int(action.metadata.get("col_start", 0)),
                int(action.metadata.get("col_end", 0)),
            )
        else:
            target_key = (action.action_id,)
        if target_key in seen_targets:
            continue
        if used + action.byte_cost > budget_bytes:
            continue
        selected.append(_selected_action_payload(action, selection_order=len(selected) + 1, cumulative_budget=used + action.byte_cost))
        used += action.byte_cost
        seen_targets.add(target_key)
    return selected


def _select_rank_actions(
    actions: list[ActionRecord],
    allocator: str,
    budget_bytes: int,
    family_rounds: int = 1,
) -> list[ActionRecord]:
    if allocator == "uniform_rank":
        candidate_ranks = sorted({int(action.rank_to) for action in actions if action.rank_to is not None})
        best_selection: list[ActionRecord] = []
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
            used += action.byte_cost
            selected_payloads.append(_selected_action_payload(action, selection_order=index, cumulative_budget=used))
        return selected_payloads
    if allocator == "greedy_family_round_robin":
        return _select_rank_actions_family_round_robin(actions, budget_bytes, family_rounds=max(family_rounds, 1))

    return _select_rank_actions_incremental(actions, budget_bytes)


def _selected_action_payload(action: ActionRecord, selection_order: int, cumulative_budget: int) -> ActionRecord:
    return action.with_selection(selection_order=selection_order, cumulative_budget_bytes=cumulative_budget)


def _resolve_candidate_bit_widths(candidate_pool: dict[str, Any], method_cfg: dict[str, Any]) -> list[int]:
    explicit = method_cfg.get("candidate_bit_widths")
    if explicit:
        return sorted({int(bit) for bit in explicit})

    pool_cfg = candidate_pool.get("bit_actions", {})
    widths = list(pool_cfg.get("candidate_bit_widths", []))
    if method_cfg.get("include_future_candidate_bit_widths", False):
        widths.extend(pool_cfg.get("future_candidate_bit_widths", []))
    return sorted({int(bit) for bit in widths})


def _select_rank_actions_incremental(actions: list[ActionRecord], budget_bytes: int) -> list[ActionRecord]:
    actions_by_layer: dict[str, list[ActionRecord]] = {}
    for action in actions:
        actions_by_layer.setdefault(action.target_name, []).append(action)
    for layer_actions in actions_by_layer.values():
        layer_actions.sort(key=lambda item: int(item.rank_to or 0))

    current_rank_by_layer = {layer: 0 for layer in actions_by_layer}
    selected: list[ActionRecord] = []
    used = 0

    while True:
        available: list[ActionRecord] = []
        for layer_name, layer_actions in actions_by_layer.items():
            current_rank = current_rank_by_layer[layer_name]
            for action in layer_actions:
                if int(action.rank_from or 0) == current_rank:
                    available.append(action)
                    break

        fitting = [action for action in available if used + action.byte_cost <= budget_bytes]
        if not fitting:
            break

        best_action = max(fitting, key=lambda item: item.predicted_gain_per_byte or 0.0)
        used += best_action.byte_cost
        current_rank_by_layer[best_action.target_name] = int(best_action.rank_to or 0)
        selected.append(
            _selected_action_payload(
                best_action,
                selection_order=len(selected) + 1,
                cumulative_budget=used,
            )
        )

    return selected


def _select_rank_actions_family_round_robin(
    actions: list[ActionRecord],
    budget_bytes: int,
    family_rounds: int,
) -> list[ActionRecord]:
    actions_by_target: dict[str, list[ActionRecord]] = {}
    for action in actions:
        actions_by_target.setdefault(action.target_name, []).append(action)
    for target_actions in actions_by_target.values():
        target_actions.sort(key=lambda item: int(item.rank_to or 0))

    current_rank_by_target = {target: 0 for target in actions_by_target}
    selected: list[ActionRecord] = []
    used = 0
    rounds_completed = 0

    while True:
        available: list[ActionRecord] = []
        for target_name, target_actions in actions_by_target.items():
            current_rank = current_rank_by_target[target_name]
            for action in target_actions:
                if int(action.rank_from or 0) == current_rank:
                    available.append(action)
                    break

        fitting = [action for action in available if used + action.byte_cost <= budget_bytes]
        if not fitting:
            break

        if rounds_completed < family_rounds:
            best_by_family: dict[str, ActionRecord] = {}
            for action in fitting:
                family = _action_family_key(action)
                current = best_by_family.get(family)
                if current is None or (action.predicted_gain_per_byte or 0.0) > (current.predicted_gain_per_byte or 0.0):
                    best_by_family[family] = action
            round_actions = sorted(best_by_family.values(), key=lambda item: item.predicted_gain_per_byte or 0.0, reverse=True)
            chose_any = False
            for action in round_actions:
                if used + action.byte_cost > budget_bytes:
                    continue
                used += action.byte_cost
                current_rank_by_target[action.target_name] = int(action.rank_to or 0)
                selected.append(
                    _selected_action_payload(
                        action,
                        selection_order=len(selected) + 1,
                        cumulative_budget=used,
                    )
                )
                chose_any = True
            if not chose_any:
                break
            rounds_completed += 1
            continue

        best_action = max(fitting, key=lambda item: item.predicted_gain_per_byte or 0.0)
        used += best_action.byte_cost
        current_rank_by_target[best_action.target_name] = int(best_action.rank_to or 0)
        selected.append(
            _selected_action_payload(
                best_action,
                selection_order=len(selected) + 1,
                cumulative_budget=used,
            )
        )

    return selected


def _action_family_key(action: ActionRecord) -> str:
    layer_name = str(action.metadata.get("layer_name", "") or action.target_name.split("::", 1)[0])
    parts = layer_name.split(".")
    if len(parts) >= 5 and parts[0] == "model" and parts[1] == "layers":
        return ".".join(parts[3:])
    return layer_name


def _build_uniform_rank_sequences(actions: list[ActionRecord], target_rank: int) -> list[dict[str, Any]]:
    actions_by_layer: dict[str, list[ActionRecord]] = {}
    for action in actions:
        actions_by_layer.setdefault(action.target_name, []).append(action)

    sequences = []
    for layer_name, layer_actions in actions_by_layer.items():
        prefix = [action for action in sorted(layer_actions, key=lambda item: int(item.rank_to or 0)) if int(action.rank_to or 0) <= target_rank]
        if not prefix:
            continue
        if int(prefix[-1].rank_to or 0) != target_rank:
            continue
        sequences.append(
            {
                "layer_name": layer_name,
                "actions": prefix,
                "total_cost": sum(action.byte_cost for action in prefix),
                "total_score": sum(action.proxy_score for action in prefix),
            }
        )
    return sequences


def _collapse_selected_rank_actions(selected_actions: list[ActionRecord]) -> dict[str, int]:
    selected_layer_ranks: dict[str, int] = {}
    for action in selected_actions:
        selected_layer_ranks[action.target_name] = int(action.rank_to if action.rank_to is not None else action.rank or 0)
    return selected_layer_ranks


def _serialize_actions(actions: list[ActionRecord]) -> list[dict[str, Any]]:
    return [action.to_dict() for action in actions]


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
        fp_module = fp_modules[layer_name]
        quant_module = quant_modules[layer_name]
        if not isinstance(fp_module, nn.Linear):
            continue
        fp_weight = fp_module.weight.detach().to(torch.float32)
        quant_weight = extract_aligned_module_weight(quant_module, fp_weight)
        if quant_weight is None:
            continue
        repaired_weight, stats = compute_low_rank_repair(
            fp_weight=fp_weight,
            quant_weight=quant_weight,
            rank=rank,
            factor_dtype_bytes=factor_dtype_bytes,
        )
        target_weight = getattr(quant_module, "weight", None)
        if not torch.is_tensor(target_weight):
            replacement = _build_linear_replacement_like(
                fp_module=fp_module,
                reference_module=quant_module,
                weight=repaired_weight,
            )
            _replace_module_by_name(quantized_model, layer_name, replacement)
        elif tuple(target_weight.shape) == tuple(repaired_weight.shape):
            target_weight.data.copy_(repaired_weight.to(device=target_weight.device, dtype=target_weight.dtype))
        else:
            replacement_weight = repaired_weight
            if target_weight.ndim == 2 and tuple(target_weight.shape) == tuple(repaired_weight.t().shape):
                replacement_weight = repaired_weight.t().contiguous()
            replacement = _build_linear_replacement_like(
                fp_module=fp_module,
                reference_module=quant_module,
                weight=replacement_weight,
            )
            _replace_module_by_name(quantized_model, layer_name, replacement)
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


def _apply_targeted_rank_actions(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    layer_stats: dict[str, dict[str, Any]],
    selected_actions: list[ActionRecord],
    factor_dtype_bytes: int,
) -> None:
    if not selected_actions:
        for stats in layer_stats.values():
            if "total_effective_bytes" not in stats:
                stats["repair_factor_bytes"] = 0
                stats["repair_rank"] = 0
                stats["total_effective_bytes"] = stats["total_quantized_bytes"]
        return

    if all(action.target_granularity == "matrix" for action in selected_actions):
        selected_layer_ranks = _collapse_selected_rank_actions(selected_actions)
        _apply_targeted_svd_repairs(
            fp_model=fp_model,
            quantized_model=quantized_model,
            layer_stats=layer_stats,
            selected_layer_ranks=selected_layer_ranks,
            factor_dtype_bytes=factor_dtype_bytes,
        )
        return

    from llm_decomposition.quantization import compute_low_rank_repair

    fp_modules = dict(fp_model.named_modules())
    quant_modules = dict(quantized_model.named_modules())
    actions_by_target: dict[str, list[ActionRecord]] = {}
    for action in selected_actions:
        actions_by_target.setdefault(action.target_name, []).append(action)

    touched_layers: set[str] = set()
    for target_name, target_actions in actions_by_target.items():
        first = target_actions[0]
        if first.target_granularity not in {"row_block", "column_block"}:
            continue
        layer_name = str(first.metadata.get("layer_name", "") or target_name)
        final_rank = int(max(action.rank_to or action.rank or 0 for action in target_actions))

        fp_module = fp_modules.get(layer_name)
        quant_module = quant_modules.get(layer_name)
        if not isinstance(fp_module, nn.Linear) or quant_module is None:
            continue
        fp_weight = fp_module.weight.detach().to(torch.float32)
        quant_weight = extract_aligned_module_weight(quant_module, fp_weight)
        if quant_weight is None:
            continue

        patched_weight = quant_weight.clone()
        if first.target_granularity == "row_block":
            row_start = int(first.metadata["row_start"])
            row_end = int(first.metadata["row_end"])
            repaired_block, stats = compute_low_rank_repair(
                fp_weight=fp_weight[row_start:row_end, :],
                quant_weight=quant_weight[row_start:row_end, :],
                rank=final_rank,
                factor_dtype_bytes=factor_dtype_bytes,
            )
            patched_weight[row_start:row_end, :] = repaired_block
        else:
            col_start = int(first.metadata["col_start"])
            col_end = int(first.metadata["col_end"])
            repaired_block, stats = compute_low_rank_repair(
                fp_weight=fp_weight[:, col_start:col_end],
                quant_weight=quant_weight[:, col_start:col_end],
                rank=final_rank,
                factor_dtype_bytes=factor_dtype_bytes,
            )
            patched_weight[:, col_start:col_end] = repaired_block

        target_weight = getattr(quant_module, "weight", None)
        if not torch.is_tensor(target_weight):
            replacement = _build_linear_replacement_like(
                fp_module=fp_module,
                reference_module=quant_module,
                weight=patched_weight,
            )
            _replace_module_by_name(quantized_model, layer_name, replacement)
        elif tuple(target_weight.shape) == tuple(patched_weight.shape):
            target_weight.data.copy_(patched_weight.to(device=target_weight.device, dtype=target_weight.dtype))
        else:
            replacement_weight = patched_weight
            if target_weight.ndim == 2 and tuple(target_weight.shape) == tuple(patched_weight.t().shape):
                replacement_weight = patched_weight.t().contiguous()
            replacement = _build_linear_replacement_like(
                fp_module=fp_module,
                reference_module=quant_module,
                weight=replacement_weight,
            )
            _replace_module_by_name(quantized_model, layer_name, replacement)

        base_stats = dict(layer_stats[layer_name])
        updated_stats = dict(base_stats)
        updated_stats["repair_factor_bytes"] = base_stats.get("repair_factor_bytes", 0) + stats["repair_factor_bytes"]
        updated_stats["repair_rank"] = max(base_stats.get("repair_rank", 0), final_rank)
        remaining = fp_weight - patched_weight
        remaining_sq_error = float(remaining.pow(2).sum().item())
        weight_sq_norm = float(fp_weight.pow(2).sum().item())
        updated_stats["post_repair_fro_error"] = remaining_sq_error ** 0.5
        updated_stats["post_repair_relative_fro_error"] = (remaining_sq_error / max(weight_sq_norm, 1e-12)) ** 0.5
        updated_stats["total_effective_bytes"] = base_stats["total_quantized_bytes"] + updated_stats["repair_factor_bytes"]
        layer_stats[layer_name] = updated_stats
        touched_layers.add(layer_name)

    for layer_name, stats in layer_stats.items():
        if layer_name not in touched_layers and "total_effective_bytes" not in stats:
            stats["repair_factor_bytes"] = stats.get("repair_factor_bytes", 0)
            stats["repair_rank"] = stats.get("repair_rank", 0)
            stats["total_effective_bytes"] = stats["total_quantized_bytes"] + stats.get("repair_factor_bytes", 0)


def _apply_targeted_bit_actions(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    layer_stats: dict[str, dict[str, Any]],
    selected_actions: list[ActionRecord],
    target_bit_width: int,
    group_size: int,
    symmetric: bool,
) -> list[str]:
    fp_modules = dict(fp_model.named_modules())
    quant_modules = dict(quantized_model.named_modules())
    actions_by_layer: dict[str, list[ActionRecord]] = {}
    for action in selected_actions:
        actions_by_layer.setdefault(action.target_name, []).append(action)

    upgraded_layers: list[str] = []
    for layer_name, layer_actions in actions_by_layer.items():
        fp_module = fp_modules.get(layer_name)
        quant_module = quant_modules.get(layer_name)
        if not isinstance(fp_module, nn.Linear) or quant_module is None:
            continue

        fp_weight = fp_module.weight.detach().to(torch.float32)
        quant_weight = extract_aligned_module_weight(quant_module, fp_weight)
        if quant_weight is None:
            continue
        patched_weight = quant_weight.clone()
        base_stats = dict(layer_stats[layer_name])
        quant_delta = 0
        metadata_delta = 0

        ordered_actions = sorted(
            layer_actions,
            key=lambda action: (
                0 if action.target_granularity == "matrix" else 1,
                int(action.metadata.get("row_start", 0)),
                int(action.metadata.get("col_start", 0)),
                int(action.selection_order or 0),
            ),
        )
        for action in ordered_actions:
            quant_delta += int(action.metadata.get("quantized_weight_byte_delta", 0))
            metadata_delta += int(action.metadata.get("metadata_byte_delta", 0))
            if action.target_granularity == "matrix":
                upgraded_weight, _ = quantize_linear_weight(
                    fp_weight,
                    bit_width=action.bit_to or target_bit_width,
                    group_size=group_size,
                    symmetric=symmetric,
                )
                patched_weight = upgraded_weight
                break
            if action.target_granularity == "row_block":
                row_start = int(action.metadata["row_start"])
                row_end = int(action.metadata["row_end"])
                upgraded_block, _ = quantize_linear_weight(
                    fp_weight[row_start:row_end, :],
                    bit_width=action.bit_to or target_bit_width,
                    group_size=group_size,
                    symmetric=symmetric,
                )
                patched_weight[row_start:row_end, :] = upgraded_block
                continue
            if action.target_granularity == "column_block":
                col_start = int(action.metadata["col_start"])
                col_end = int(action.metadata["col_end"])
                upgraded_block, _ = quantize_linear_weight(
                    fp_weight[:, col_start:col_end],
                    bit_width=action.bit_to or target_bit_width,
                    group_size=group_size,
                    symmetric=symmetric,
                )
                patched_weight[:, col_start:col_end] = upgraded_block

        target_weight = getattr(quant_module, "weight", None)
        if not torch.is_tensor(target_weight):
            replacement = _build_linear_replacement_like(
                fp_module=fp_module,
                reference_module=quant_module,
                weight=patched_weight,
            )
            _replace_module_by_name(quantized_model, layer_name, replacement)
        elif tuple(target_weight.shape) == tuple(patched_weight.shape):
            target_weight.data.copy_(patched_weight.to(device=target_weight.device, dtype=target_weight.dtype))
        else:
            replacement_weight = patched_weight
            if target_weight.ndim == 2 and tuple(target_weight.shape) == tuple(patched_weight.t().shape):
                replacement_weight = patched_weight.t().contiguous()
            replacement = _build_linear_replacement_like(
                fp_module=fp_module,
                reference_module=quant_module,
                weight=replacement_weight,
            )
            _replace_module_by_name(quantized_model, layer_name, replacement)

        remaining = fp_weight - patched_weight
        remaining_sq_error = float(remaining.pow(2).sum().item())
        weight_sq_norm = float(fp_weight.pow(2).sum().item())
        updated_stats = dict(base_stats)
        updated_stats["quantized_weight_bytes"] = base_stats["quantized_weight_bytes"] + quant_delta
        updated_stats["metadata_bytes"] = base_stats["metadata_bytes"] + metadata_delta
        updated_stats["total_quantized_bytes"] = updated_stats["quantized_weight_bytes"] + updated_stats["metadata_bytes"]
        updated_stats["fro_error"] = remaining_sq_error ** 0.5
        updated_stats["relative_fro_error"] = (remaining_sq_error / max(weight_sq_norm, 1e-12)) ** 0.5
        updated_stats["repair_factor_bytes"] = 0
        updated_stats["repair_rank"] = 0
        updated_stats["total_effective_bytes"] = updated_stats["total_quantized_bytes"]
        layer_stats[layer_name] = updated_stats
        upgraded_layers.append(layer_name)

    for stats in layer_stats.values():
        if "total_effective_bytes" not in stats:
            stats["repair_factor_bytes"] = 0
            stats["repair_rank"] = 0
            stats["total_effective_bytes"] = stats["total_quantized_bytes"]

    return sorted(set(upgraded_layers))


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
    actions: list[ActionRecord],
    selected_actions: list[ActionRecord],
    base_actions: list[ActionRecord] | None = None,
) -> None:
    run_dir = root / config.raw["outputs"]["results_dir"]
    write_json(
        run_dir / "actions.json",
        {
            "run_id": config.run_id,
            "status": "completed",
            "actions": _serialize_actions(actions),
            "base_actions": _serialize_actions(base_actions or []),
            "selected_actions": _serialize_actions(selected_actions),
        },
    )


def _replace_module_by_name(root_module: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent = root_module
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _build_linear_replacement_like(
    fp_module: nn.Linear,
    reference_module: nn.Module,
    weight: torch.Tensor,
) -> nn.Linear:
    target_device = _module_device(reference_module)
    target_dtype = _module_dtype(reference_module)
    if not torch.empty((), dtype=target_dtype).is_floating_point():
        target_dtype = fp_module.weight.dtype if fp_module.weight.dtype.is_floating_point else torch.float16
    replacement = nn.Linear(
        in_features=fp_module.in_features,
        out_features=fp_module.out_features,
        bias=fp_module.bias is not None,
        device=target_device,
        dtype=target_dtype,
    )
    replacement.weight.data.copy_(weight.to(device=target_device, dtype=target_dtype))
    if fp_module.bias is not None:
        replacement.bias.data.copy_(fp_module.bias.detach().to(device=target_device, dtype=target_dtype))
    return replacement


def _module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.device
    return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.dtype
    return torch.float16


def _full_precision_memory_bytes(model) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _maybe_run_downstream(
    root: Path,
    config: RunConfig,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    metrics: dict[str, Any],
) -> None:
    """Run downstream zero-shot evaluation if ``downstream.enabled`` is set."""
    downstream_cfg = config.raw.get("downstream")
    if not downstream_cfg or not downstream_cfg.get("enabled", False):
        return

    from llm_decomposition.downstream_eval import (
        evaluate_downstream,
        write_downstream_metrics,
    )

    print(f"[{config.run_id}] running downstream evaluation")
    results = evaluate_downstream(
        model=model,
        tokenizer=tokenizer,
        tasks=downstream_cfg.get("tasks"),
        num_fewshot=downstream_cfg.get("num_fewshot"),
        batch_size=downstream_cfg.get("batch_size", 4),
        device=device,
    )
    run_dir = root / config.raw["outputs"]["results_dir"]
    output_file = downstream_cfg.get("output_file", "downstream_metrics.json")
    write_downstream_metrics(run_dir, config.run_id, results, output_file)
    metrics["downstream"] = results.get("results", {})
    print(f"[{config.run_id}] downstream evaluation complete")
