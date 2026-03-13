from __future__ import annotations

import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    name: str
    required_modules: tuple[str, ...]
    description: str


METHOD_SPECS = {
    "full_precision": MethodSpec(
        name="full_precision",
        required_modules=("torch", "transformers", "datasets"),
        description="Full-precision Hugging Face model evaluation.",
    ),
    "rtn": MethodSpec(
        name="rtn",
        required_modules=("torch", "transformers", "datasets"),
        description="Round-to-nearest post-training quantization baseline.",
    ),
    "gptq": MethodSpec(
        name="gptq",
        required_modules=("torch", "transformers", "datasets", "optimum", "gptqmodel"),
        description="GPTQ post-training quantization baseline.",
    ),
    "uniform_svd_repair": MethodSpec(
        name="uniform_svd_repair",
        required_modules=("torch", "transformers", "datasets"),
        description="RTN baseline with uniform low-rank residual repair.",
    ),
    "mixed_precision_budget_match": MethodSpec(
        name="mixed_precision_budget_match",
        required_modules=("torch", "transformers", "datasets"),
        description="Bits-only baseline matched to a repair memory budget.",
    ),
    "targeted_mixed_precision": MethodSpec(
        name="targeted_mixed_precision",
        required_modules=("torch", "transformers", "datasets"),
        description="Phase-2 targeted mixed-precision allocator over a restricted action pool.",
    ),
    "targeted_svd_rank": MethodSpec(
        name="targeted_svd_rank",
        required_modules=("torch", "transformers", "datasets"),
        description="Phase-2 targeted low-rank allocator over a restricted action pool.",
    ),
    "hybrid_second_stage": MethodSpec(
        name="hybrid_second_stage",
        required_modules=("torch", "transformers", "datasets"),
        description="Phase-2 hybrid second-stage allocator on top of a bits-only base point.",
    ),
}


def method_spec(name: str) -> MethodSpec:
    if name not in METHOD_SPECS:
        available = ", ".join(sorted(METHOD_SPECS))
        raise ValueError(f"Unknown method '{name}'. Available methods: {available}")
    return METHOD_SPECS[name]


def missing_modules(modules: tuple[str, ...]) -> list[str]:
    return [module for module in modules if importlib.util.find_spec(module) is None]
