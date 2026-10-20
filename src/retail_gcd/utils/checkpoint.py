from __future__ import annotations

import json
from pathlib import Path

import torch


def atomic_torch_save(state: dict, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(destination)


def save_run_metadata(output_dir: str | Path, config: dict, class_mapping: dict[str, int]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    serializable_config = {key: value for key, value in config.items() if not key.startswith("_")}
    (output / "resolved_config.json").write_text(json.dumps(serializable_config, indent=2), encoding="utf-8")
    (output / "class_mapping.json").write_text(
        json.dumps(class_mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"model", "epoch", "class_mapping", "config"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    return checkpoint

