from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.nn as nn
from optimum.gptq import GPTQQuantizer

from llm_decomposition.hf_utils import load_text_split
from llm_decomposition.quantization import quantize_linear_weight


def build_gptq_calibration_texts(config) -> list[str]:
    calibration_cfg = config.raw["calibration"]
    dataset = load_text_split(
        calibration_cfg["dataset"],
        calibration_cfg["subset"],
        calibration_cfg["split"],
    )
    if calibration_cfg.get("sampling") == "seeded_shuffle":
        dataset = dataset.shuffle(seed=calibration_cfg.get("seed", 42))

    text_field = _detect_text_field(dataset.column_names)
    text_limit = calibration_cfg.get("num_text_samples", calibration_cfg.get("num_sequences", 128))
    texts: list[str] = []
    for text in dataset[text_field]:
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        texts.append(stripped)
        if len(texts) >= text_limit:
            break
    return texts


def quantize_model_gptq(model, config, tokenizer) -> tuple[nn.Module, int]:
    method_cfg = config.raw["method"]
    calibration_cfg = config.raw["calibration"]
    if not hasattr(model, "hf_device_map"):
        try:
            first_param = next(model.parameters())
            device = first_param.device
            device_label = device.type
            if device.index is not None:
                device_label = f"{device.type}:{device.index}"
            model.hf_device_map = {"": device_label}
        except StopIteration:
            model.hf_device_map = {"": "cpu"}
    quantizer = GPTQQuantizer(
        bits=method_cfg.get("bit_width", 4),
        dataset=build_gptq_calibration_texts(config),
        group_size=method_cfg.get("group_size", 128),
        damp_percent=method_cfg.get("damp_percent", 0.1),
        desc_act=method_cfg.get("desc_act", False),
        act_group_aware=method_cfg.get("act_group_aware", True),
        sym=method_cfg.get("symmetric", True),
        true_sequential=method_cfg.get("true_sequential", True),
        batch_size=method_cfg.get("batch_size", 1),
        model_seqlen=calibration_cfg.get("sequence_length"),
        backend=method_cfg.get("backend"),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    quantized_model = quantizer.quantize_model(model, tokenizer)
    with TemporaryDirectory(prefix="llm_decomp_gptq_") as tmp_dir:
        quantizer.save(quantized_model, tmp_dir, safe_serialization=True)
        total_bytes = compute_saved_artifact_bytes(Path(tmp_dir))
    return quantized_model, total_bytes


def estimate_gptq_layer_stats(
    model: nn.Module,
    bit_width: int,
    group_size: int,
    symmetric: bool,
) -> dict[str, dict[str, Any]]:
    layer_stats: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        _, stats = quantize_linear_weight(
            module.weight.detach().to(torch.float32),
            bit_width=bit_width,
            group_size=group_size,
            symmetric=symmetric,
        )
        layer_stats[name] = stats
    return layer_stats


def apply_targeted_bit_upgrades(
    fp_model: nn.Module,
    quantized_model: nn.Module,
    layer_stats: dict[str, dict[str, Any]],
    target_layers: list[str],
    target_bit_width: int,
    group_size: int,
    symmetric: bool,
) -> list[str]:
    fp_modules = dict(fp_model.named_modules())
    quant_modules = dict(quantized_model.named_modules())
    upgraded: list[str] = []

    for layer_name in target_layers:
        fp_module = fp_modules.get(layer_name)
        quant_module = quant_modules.get(layer_name)
        if not isinstance(fp_module, nn.Linear) or not isinstance(quant_module, nn.Linear):
            continue
        upgraded_weight, upgraded_stats = quantize_linear_weight(
            fp_module.weight.detach().to(torch.float32),
            bit_width=target_bit_width,
            group_size=group_size,
            symmetric=symmetric,
        )
        quant_module.weight.data.copy_(upgraded_weight.to(quant_module.weight.dtype))
        layer_stats[layer_name] = upgraded_stats
        upgraded.append(layer_name)
    return upgraded


def compute_saved_artifact_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _detect_text_field(column_names: list[str]) -> str:
    for candidate in ("text", "content", "sentence"):
        if candidate in column_names:
            return candidate
    raise ValueError(f"Could not detect a text column from columns: {column_names}")
