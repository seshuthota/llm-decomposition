from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


def quantize_model_rtn(
    model: nn.Module,
    bit_width: int,
    group_size: int,
    symmetric: bool,
) -> tuple[nn.Module, dict[str, dict[str, Any]]]:
    return quantize_model_mixed_precision(
        model=model,
        default_bit_width=bit_width,
        layer_bit_overrides={},
        group_size=group_size,
        symmetric=symmetric,
    )


def quantize_model_mixed_precision(
    model: nn.Module,
    default_bit_width: int,
    layer_bit_overrides: dict[str, int],
    group_size: int,
    symmetric: bool,
) -> tuple[nn.Module, dict[str, dict[str, Any]]]:
    quantized_model = copy.deepcopy(model)
    layer_stats: dict[str, dict[str, Any]] = {}
    source_modules = dict(model.named_modules())

    for name, module in quantized_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        target_bit_width = layer_bit_overrides.get(name, default_bit_width)
        original_weight = source_modules[name].weight.detach().to(torch.float32)
        quantized_weight, stats = quantize_linear_weight(
            original_weight,
            bit_width=target_bit_width,
            group_size=group_size,
            symmetric=symmetric,
        )
        module.weight.data.copy_(quantized_weight.to(module.weight.dtype))
        layer_stats[name] = stats

    return quantized_model, layer_stats


def quantize_linear_weight(
    weight: torch.Tensor,
    bit_width: int,
    group_size: int,
    symmetric: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if weight.dim() != 2:
        raise ValueError("RTN quantization expects a 2D linear weight matrix.")

    rows, cols = weight.shape
    q_weight = torch.empty_like(weight, dtype=torch.float32)
    qmin, qmax = _quant_bounds(bit_width, symmetric)

    original_sq_error = 0.0
    weight_sq_norm = float(weight.pow(2).sum().item())
    num_groups = 0

    for row_index in range(rows):
        row = weight[row_index]
        chunks = torch.split(row, group_size)
        quantized_chunks: list[torch.Tensor] = []
        for chunk in chunks:
            dequantized = _quantize_chunk(chunk, qmin=qmin, qmax=qmax, symmetric=symmetric)
            quantized_chunks.append(dequantized)
            original_sq_error += float((chunk - dequantized).pow(2).sum().item())
            num_groups += 1
        q_weight[row_index] = torch.cat(quantized_chunks, dim=0)

    metadata_bytes = num_groups * 2
    if not symmetric:
        metadata_bytes += num_groups

    numel = weight.numel()
    quantized_weight_bytes = (numel * bit_width + 7) // 8
    total_quantized_bytes = quantized_weight_bytes + metadata_bytes

    stats = {
        "shape": [rows, cols],
        "num_parameters": numel,
        "bit_width": bit_width,
        "group_size": group_size,
        "symmetric": symmetric,
        "num_groups": num_groups,
        "quantized_weight_bytes": quantized_weight_bytes,
        "metadata_bytes": metadata_bytes,
        "total_quantized_bytes": total_quantized_bytes,
        "fp32_weight_bytes": numel * 4,
        "fro_error": original_sq_error ** 0.5,
        "relative_fro_error": (original_sq_error / max(weight_sq_norm, 1e-12)) ** 0.5,
    }
    return q_weight, stats


def apply_uniform_svd_repair(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    layer_stats: dict[str, dict[str, Any]],
    layer_names: list[str],
    rank: int,
    factor_dtype_bytes: int,
) -> dict[str, dict[str, Any]]:
    fp_modules = dict(fp_model.named_modules())
    quant_modules = dict(quantized_model.named_modules())
    repair_stats: dict[str, dict[str, Any]] = {}

    for layer_name in layer_names:
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
        repair_stats[layer_name] = updated_stats
        layer_stats[layer_name] = updated_stats

    for layer_name, stats in layer_stats.items():
        if "total_effective_bytes" not in stats:
            stats["repair_factor_bytes"] = 0
            stats["repair_rank"] = 0
            stats["total_effective_bytes"] = stats["total_quantized_bytes"]

    return repair_stats


def compute_low_rank_repair(
    fp_weight: torch.Tensor,
    quant_weight: torch.Tensor,
    rank: int,
    factor_dtype_bytes: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    residual = fp_weight - quant_weight
    max_rank = min(residual.shape)
    effective_rank = min(rank, max_rank)
    if effective_rank <= 0:
        fro_error = float(residual.pow(2).sum().sqrt().item())
        return quant_weight.clone(), {
            "repair_rank": 0,
            "repair_factor_bytes": 0,
            "post_repair_fro_error": fro_error,
            "post_repair_relative_fro_error": 1.0,
        }

    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    u_r = u[:, :effective_rank]
    s_r = s[:effective_rank]
    vh_r = vh[:effective_rank, :]
    low_rank_residual = (u_r * s_r.unsqueeze(0)) @ vh_r
    repaired_weight = quant_weight + low_rank_residual

    remaining = fp_weight - repaired_weight
    remaining_sq_error = float(remaining.pow(2).sum().item())
    weight_sq_norm = float(fp_weight.pow(2).sum().item())
    repair_factor_bytes = (
        fp_weight.shape[0] * effective_rank + effective_rank * fp_weight.shape[1]
    ) * factor_dtype_bytes

    return repaired_weight, {
        "repair_rank": effective_rank,
        "repair_factor_bytes": repair_factor_bytes,
        "post_repair_fro_error": remaining_sq_error ** 0.5,
        "post_repair_relative_fro_error": (remaining_sq_error / max(weight_sq_norm, 1e-12)) ** 0.5,
    }


def _quant_bounds(bit_width: int, symmetric: bool) -> tuple[int, int]:
    if symmetric:
        qmax = 2 ** (bit_width - 1) - 1
        qmin = -2 ** (bit_width - 1)
        return qmin, qmax
    qmin = 0
    qmax = 2**bit_width - 1
    return qmin, qmax


def _quantize_chunk(chunk: torch.Tensor, qmin: int, qmax: int, symmetric: bool) -> torch.Tensor:
    if symmetric:
        max_abs = float(chunk.abs().max().item())
        if max_abs == 0.0:
            return torch.zeros_like(chunk, dtype=torch.float32)
        scale = max_abs / max(qmax, 1)
        q = torch.clamp(torch.round(chunk / scale), qmin, qmax)
        return q * scale

    min_value = float(chunk.min().item())
    max_value = float(chunk.max().item())
    if max_value == min_value:
        return torch.full_like(chunk, fill_value=min_value, dtype=torch.float32)
    scale = (max_value - min_value) / max(qmax - qmin, 1)
    zero_point = round(qmin - min_value / scale)
    q = torch.clamp(torch.round(chunk / scale + zero_point), qmin, qmax)
    return (q - zero_point) * scale
