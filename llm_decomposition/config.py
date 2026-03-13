from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "run_id",
    "phase",
    "description",
    "model",
    "method",
    "calibration",
    "evaluation",
    "profiling",
    "outputs",
}


@dataclass(frozen=True)
class RunConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def run_id(self) -> str:
        return self.raw["run_id"]

    @property
    def results_dir(self) -> str:
        return self.raw["outputs"]["results_dir"]

    @property
    def model_name(self) -> str:
        return self.raw["model"]["name"]

    @property
    def method_name(self) -> str:
        return self.raw["method"]["name"]

    @property
    def bit_width(self) -> int | None:
        return self.raw["method"].get("bit_width")


@dataclass(frozen=True)
class Manifest:
    path: Path
    raw: dict[str, Any]
    run_configs: list[RunConfig]

    @property
    def phase(self) -> str:
        return self.raw["phase"]

    @property
    def description(self) -> str:
        return self.raw.get("description", "")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_run_config(config_path: Path, config: dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL_KEYS - set(config)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{config_path}: missing required keys: {missing_list}")

    outputs = config["outputs"]
    if "results_dir" not in outputs:
        raise ValueError(f"{config_path}: outputs.results_dir is required")

    if "name" not in config["model"]:
        raise ValueError(f"{config_path}: model.name is required")

    if "name" not in config["method"]:
        raise ValueError(f"{config_path}: method.name is required")


def load_manifest(root: Path, manifest_rel_path: str) -> Manifest:
    manifest_path = (root / manifest_rel_path).resolve()
    manifest_raw = load_json(manifest_path)
    run_configs: list[RunConfig] = []

    for run_rel_path in manifest_raw["runs"]:
        run_path = (root / run_rel_path).resolve()
        run_raw = load_json(run_path)
        validate_run_config(run_path, run_raw)
        run_configs.append(RunConfig(path=run_path, raw=run_raw))

    return Manifest(path=manifest_path, raw=manifest_raw, run_configs=run_configs)
