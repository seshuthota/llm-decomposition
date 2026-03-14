from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def summarize_layer_errors(layer_stats: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for name, stats in layer_stats.items():
        summaries.append(
            {
                "layer_name": name,
                "fro_error": stats.get("post_repair_fro_error", stats["fro_error"]),
                "relative_fro_error": stats.get(
                    "post_repair_relative_fro_error",
                    stats["relative_fro_error"],
                ),
                "quantized_weight_bytes": stats["quantized_weight_bytes"],
                "metadata_bytes": stats["metadata_bytes"],
                "total_quantized_bytes": stats["total_quantized_bytes"],
                "repair_factor_bytes": stats.get("repair_factor_bytes", 0),
                "repair_rank": stats.get("repair_rank", 0),
                "total_effective_bytes": stats.get(
                    "total_effective_bytes",
                    stats["total_quantized_bytes"],
                ),
            }
        )
    summaries.sort(key=lambda item: item["relative_fro_error"], reverse=True)
    return summaries


def measure_activation_error(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    sequences: list[torch.Tensor],
    device: torch.device,
    target_layers: list[str] | None = None,
    sequential_offload: bool = False,
) -> dict[str, dict[str, float]]:
    fp_modules = {
        name: module for name, module in fp_model.named_modules() if isinstance(module, nn.Linear)
    }
    quant_named_modules = dict(quantized_model.named_modules())
    layer_names = sorted(target_layers or fp_modules.keys())

    fp_outputs: dict[str, torch.Tensor] = {}
    quant_outputs: dict[str, torch.Tensor] = {}
    accumulators = {
        name: {"sum_sq_error": 0.0, "sum_sq_ref": 0.0, "num_batches": 0}
        for name in layer_names
        if name in fp_modules and name in quant_named_modules
    }

    fp_hooks = [
        fp_modules[name].register_forward_hook(_capture_output_hook(fp_outputs, name))
        for name in accumulators
    ]
    quant_hooks = [
        quant_named_modules[name].register_forward_hook(_capture_output_hook(quant_outputs, name))
        for name in accumulators
    ]

    try:
        with torch.no_grad():
            for sequence in sequences:
                fp_outputs.clear()
                quant_outputs.clear()
                inputs = sequence.unsqueeze(0).to(device)
                if sequential_offload:
                    fp_model.to(device)
                    fp_model(input_ids=inputs)
                    fp_model.to("cpu")
                    _clear_cuda_cache(device)

                    quantized_model.to(device)
                    quantized_model(input_ids=inputs)
                    quantized_model.to("cpu")
                    _clear_cuda_cache(device)
                else:
                    fp_model(input_ids=inputs)
                    quantized_model(input_ids=inputs)
                for name, stats in accumulators.items():
                    if name not in fp_outputs or name not in quant_outputs:
                        continue
                    ref = fp_outputs[name]
                    test = quant_outputs[name]
                    diff = (ref - test).to(torch.float32)
                    stats["sum_sq_error"] += float(diff.pow(2).sum().item())
                    stats["sum_sq_ref"] += float(ref.to(torch.float32).pow(2).sum().item())
                    stats["num_batches"] += 1
    finally:
        for hook in fp_hooks + quant_hooks:
            hook.remove()

    results: dict[str, dict[str, float]] = {}
    for name, stats in accumulators.items():
        mse = stats["sum_sq_error"] / max(stats["num_batches"], 1)
        relative_l2 = (stats["sum_sq_error"] / max(stats["sum_sq_ref"], 1e-12)) ** 0.5
        results[name] = {
            "activation_mse": mse,
            "activation_relative_l2": relative_l2,
            "num_batches": float(stats["num_batches"]),
        }
    return results


def merge_layer_metrics(
    layer_summaries: list[dict[str, Any]],
    activation_errors: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    merged = []
    for item in layer_summaries:
        layer_name = item["layer_name"]
        merged_item = dict(item)
        merged_item.update(activation_errors.get(layer_name, {}))
        merged.append(merged_item)
    merged.sort(
        key=lambda entry: entry.get("activation_relative_l2", entry["relative_fro_error"]),
        reverse=True,
    )
    return merged


def profile_residual_svd(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    candidate_layers: list[str],
    candidate_ranks: list[int],
) -> list[dict[str, Any]]:
    fp_modules = {
        name: module for name, module in fp_model.named_modules() if isinstance(module, nn.Linear)
    }
    quant_modules = dict(quantized_model.named_modules())
    profiles = []
    for layer_name in candidate_layers:
        fp_module = fp_modules.get(layer_name)
        quant_module = quant_modules.get(layer_name)
        if fp_module is None or quant_module is None:
            continue
        fp_weight = fp_module.weight.detach().to(torch.float32)
        quant_weight = extract_aligned_module_weight(quant_module, fp_weight)
        if quant_weight is None:
            continue
        residual = fp_weight - quant_weight
        singular_values = torch.linalg.svdvals(residual)
        energy = singular_values.pow(2)
        total_energy = float(energy.sum().item())
        rank_energy = {}
        cumulative = torch.cumsum(energy, dim=0)
        for rank in candidate_ranks:
            effective_rank = min(rank, singular_values.numel())
            explained = float(cumulative[effective_rank - 1].item()) if effective_rank > 0 else 0.0
            rank_energy[str(rank)] = explained / max(total_energy, 1e-12)
        profiles.append(
            {
                "layer_name": layer_name,
                "shape": list(residual.shape),
                "total_energy": total_energy,
                "rank_energy": rank_energy,
            }
        )
    return profiles


def _capture_output_hook(store: dict[str, torch.Tensor], name: str):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        store[name] = output.detach().cpu()

    return hook


def _clear_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def extract_aligned_module_weight(
    module: nn.Module,
    reference_weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    weight = _extract_module_weight(module)
    if weight is None:
        return None
    if reference_weight is None:
        return weight
    if tuple(weight.shape) == tuple(reference_weight.shape):
        return weight
    if weight.ndim == 2 and tuple(weight.t().shape) == tuple(reference_weight.shape):
        return weight.t().contiguous()
    return None


def _extract_module_weight(module: nn.Module) -> torch.Tensor | None:
    for attr_name in ("dequantize_weight", "dequantize", "unpack"):
        method = getattr(module, attr_name, None)
        if not callable(method):
            continue
        try:
            candidate = method()
        except TypeError:
            continue
        except Exception:
            continue
        if torch.is_tensor(candidate):
            return candidate.detach().to(torch.float32)

    weight = getattr(module, "weight", None)
    if torch.is_tensor(weight):
        return weight.detach().to(torch.float32)
    return None
