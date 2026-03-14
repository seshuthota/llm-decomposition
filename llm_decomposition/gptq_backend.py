from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
import torch.nn as nn
from accelerate.hooks import AlignDevicesHook
from transformers import AutoModelForCausalLM, GPTQConfig
from optimum.gptq import GPTQQuantizer

from llm_decomposition.hf_utils import _hf_token, load_text_split
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


def build_gptq_config(config, tokenizer) -> GPTQConfig:
    return GPTQConfig(tokenizer=tokenizer, **_gptq_common_kwargs(config))


def _gptq_common_kwargs(config) -> dict[str, Any]:
    method_cfg = config.raw["method"]
    calibration_cfg = config.raw["calibration"]
    return {
        "bits": method_cfg.get("bit_width", 4),
        "dataset": build_gptq_calibration_texts(config),
        "group_size": method_cfg.get("group_size", 128),
        "damp_percent": method_cfg.get("damp_percent", 0.1),
        "desc_act": method_cfg.get("desc_act", False),
        "act_group_aware": method_cfg.get("act_group_aware", True),
        "sym": method_cfg.get("symmetric", True),
        "true_sequential": method_cfg.get("true_sequential", True),
        "batch_size": method_cfg.get("batch_size", 1),
        "model_seqlen": calibration_cfg.get("sequence_length"),
        "backend": method_cfg.get("backend"),
    }


def quantize_model_gptq(model, config, tokenizer, *, model_name: str | None = None, runtime=None) -> tuple[nn.Module, int]:
    method_cfg = config.raw["method"]
    implementation = method_cfg.get("implementation", "transformers_gptq_config")
    if implementation == "transformers_gptq_config":
        resolved_model_name = model_name or getattr(getattr(model, "config", None), "_name_or_path", None)
        if resolved_model_name is None:
            raise ValueError("GPTQ transformers implementation requires a model_name.")
        quantized_model = _quantize_model_gptq_transformers(
            model_name=resolved_model_name,
            config=config,
            tokenizer=tokenizer,
            runtime=runtime,
        )
    elif implementation == "optimum_quantizer":
        quantized_model = _quantize_model_gptq_optimum(
            model=model,
            config=config,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(f"Unsupported GPTQ implementation '{implementation}'.")

    with TemporaryDirectory(prefix="llm_decomp_gptq_") as tmp_dir:
        quantized_model.save_pretrained(tmp_dir, safe_serialization=True)
        total_bytes = compute_saved_artifact_bytes(Path(tmp_dir))
    return quantized_model, total_bytes


def _quantize_model_gptq_transformers(model_name: str, config, tokenizer, runtime):
    method_cfg = config.raw["method"]
    quantization_config = build_gptq_config(config, tokenizer)
    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "quantization_config": quantization_config,
    }
    if runtime is not None and runtime.device.type == "cuda":
        kwargs["torch_dtype"] = runtime.dtype
        device_map_mode = method_cfg.get("device_map", "auto")
        if device_map_mode == "auto":
            kwargs["device_map"] = "auto"
        elif device_map_mode in {"single", "cuda"}:
            kwargs["device_map"] = {"": _runtime_device_label(runtime)}
        elif device_map_mode in {None, "none"}:
            pass
        else:
            kwargs["device_map"] = device_map_mode
    token = _hf_token()
    if token is not None:
        kwargs["token"] = token
    with _patched_optimum_quantize_model(_runtime_device_label(runtime)), _patched_accelerate_paramless_pre_forward():
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()
    return model


def _quantize_model_gptq_optimum(model, config, tokenizer):
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
    quantizer = GPTQQuantizer(**_gptq_common_kwargs(config))
    if hasattr(model, "config"):
        model.config.use_cache = False
    quantized_model = quantizer.quantize_model(model, tokenizer)
    return quantized_model


@contextmanager
def _patched_optimum_quantize_model(device_label: str):
    original = GPTQQuantizer.quantize_model

    def wrapped(self, model, tokenizer):
        if not hasattr(model, "hf_device_map"):
            model.hf_device_map = {"": device_label}
        return original(self, model, tokenizer)

    GPTQQuantizer.quantize_model = wrapped
    try:
        yield
    finally:
        GPTQQuantizer.quantize_model = original


def _runtime_device_label(runtime) -> str:
    if runtime is None:
        return "cpu"
    device = runtime.device
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


@contextmanager
def _patched_accelerate_paramless_pre_forward():
    original = AlignDevicesHook.pre_forward

    def wrapped(self, module, *args, **kwargs):
        try:
            return original(self, module, *args, **kwargs)
        except StopIteration:
            has_local_state = any(True for _ in module.parameters(recurse=False)) or any(
                True for _ in module.buffers(recurse=False)
            )
            if has_local_state:
                raise
            return args, kwargs

    AlignDevicesHook.pre_forward = wrapped
    try:
        yield
    finally:
        AlignDevicesHook.pre_forward = original


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
        if not isinstance(fp_module, nn.Linear) or quant_module is None:
            continue
        upgraded_weight, upgraded_stats = quantize_linear_weight(
            fp_module.weight.detach().to(torch.float32),
            bit_width=target_bit_width,
            group_size=group_size,
            symmetric=symmetric,
        )
        replacement = _build_linear_replacement(
            source_module=fp_module,
            weight=upgraded_weight,
            target_dtype=_module_dtype(quant_module),
            target_device=_module_device(quant_module),
        )
        _replace_module_by_name(quantized_model, layer_name, replacement)
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


def _replace_module_by_name(root_module: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent = root_module
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.device
    return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        return tensor.dtype
    return torch.float16


def _build_linear_replacement(
    source_module: nn.Linear,
    weight: torch.Tensor,
    target_dtype: torch.dtype,
    target_device: torch.device,
) -> nn.Linear:
    if not torch.empty((), dtype=target_dtype).is_floating_point():
        target_dtype = source_module.weight.dtype if source_module.weight.dtype.is_floating_point else torch.float16
    replacement = nn.Linear(
        in_features=source_module.in_features,
        out_features=source_module.out_features,
        bias=source_module.bias is not None,
        device=target_device,
        dtype=target_dtype,
    )
    replacement.weight.data.copy_(weight.to(device=target_device, dtype=target_dtype))
    if source_module.bias is not None:
        replacement.bias.data.copy_(source_module.bias.detach().to(device=target_device, dtype=target_dtype))
    return replacement
