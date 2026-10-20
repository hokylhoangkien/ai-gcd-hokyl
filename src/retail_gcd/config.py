from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    pass


REQUIRED_LOSS_WEIGHTS = (
    "lambda_supervised",
    "lambda_consistency",
    "lambda_distillation",
    "lambda_instance_contrastive",
    "lambda_supervised_contrastive",
    "lambda_dual_branch",
    "lambda_cico",
    "lambda_sdl",
    "lambda_adl",
    "lambda_hierarchy",
    "lambda_entropy",
    "lambda_attribute",
    "lambda_conflict",
    "lambda_calibration",
)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "train", "output_dir"):
        if section not in config:
            raise ConfigurationError(f"Missing config section/key: {section}")
    data = config["data"]
    model = config["model"]
    train = config["train"]
    for key in ("manifest", "image_root", "attributes", "batch_size", "num_workers"):
        if key not in data:
            raise ConfigurationError(f"Missing data.{key}")
    if len(data["attributes"]) != len(set(data["attributes"])):
        raise ConfigurationError("data.attributes contains duplicates")
    validation_splits = data.get("validation_splits", ["val"])
    if not validation_splits:
        raise ConfigurationError("data.validation_splits must contain at least one validation split")
    if "test" in validation_splits:
        raise ConfigurationError(
            "data.validation_splits must not contain 'test'; select checkpoints on validation data "
            "and evaluate test only with retail-gcd-eval"
        )
    if model.get("fusion_mode") not in {"visual", "attributes", "simple", "oracle", "proposed"}:
        raise ConfigurationError("model.fusion_mode is invalid")
    if train.get("epochs", 0) < 1 or data.get("batch_size", 0) < 1:
        raise ConfigurationError("epochs and batch_size must be positive")
    for key in REQUIRED_LOSS_WEIGHTS:
        if key not in train or train[key] < 0:
            raise ConfigurationError(f"train.{key} must be present and non-negative")
    for key in ("pseudo_label_threshold", "attribute_confidence_threshold", "debiased_pseudo_threshold"):
        if not 0.0 <= train[key] <= 1.0:
            raise ConfigurationError(f"train.{key} must be in [0, 1]")
    scheduled_components = ("dual_branch", "cico", "sdl", "adl", "hierarchy", "conflict", "calibration")
    for component in scheduled_components:
        start = train.get(f"{component}_start_epoch", 0)
        warmup = train.get(f"{component}_warmup_epochs", 0)
        if not isinstance(start, int) or start < 0:
            raise ConfigurationError(f"train.{component}_start_epoch must be a non-negative integer")
        if not isinstance(warmup, int) or warmup < 0:
            raise ConfigurationError(f"train.{component}_warmup_epochs must be a non-negative integer")
    if model["fusion_mode"] == "oracle" and train.get("lambda_calibration", 0) != 0:
        raise ConfigurationError("Oracle fusion must set lambda_calibration=0")

