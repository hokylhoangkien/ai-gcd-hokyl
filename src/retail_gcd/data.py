from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import Dataset

from .manifest import DEFAULT_ATTRIBUTES


class RetailGCDDataset(Dataset):
    """Manifest-backed dataset that never exposes novel labels to the loss.

    `target` is retained for evaluation. Training code must gate supervised
    losses with `is_labeled`.
    """

    def __init__(
        self,
        manifest: str | Path,
        image_root: str | Path,
        transform: Callable | None = None,
        splits: Sequence[str] = ("labeled", "unlabeled"),
        attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
        sku_to_index: dict[str, int] | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.transform = transform
        self.attributes = tuple(attributes)
        with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
            allowed_splits = {split.strip().lower() for split in splits}
            self.rows = []
            for row in csv.DictReader(handle):
                row["split"] = row["split"].strip().lower()
                row["image_path"] = row["image_path"].strip()
                row["sku_id"] = row["sku_id"].strip()
                if row["split"] in allowed_splits:
                    self.rows.append(row)
        sku_names = sorted({row["sku_id"] for row in self.rows})
        if sku_to_index is None:
            known_names = sorted({row["sku_id"] for row in self.rows if bool(int(row["is_known"]))})
            novel_names = sorted(set(sku_names) - set(known_names))
            # DebGCD-style OVA heads require known targets to occupy [0, K_known).
            ordered_names = known_names + novel_names
            self.sku_to_index = {name: index for index, name in enumerate(ordered_names)}
        else:
            self.sku_to_index = sku_to_index
        missing = set(sku_names) - set(self.sku_to_index)
        if missing:
            raise ValueError(f"Evaluation manifest contains unmapped SKUs: {sorted(missing)[:10]}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with Image.open(self.image_root / row["image_path"]) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        values = [row.get(name, "").strip() for name in self.attributes]
        confidences = [float(row.get(f"{name}_conf", "0") or 0.0) for name in self.attributes]
        return {
            "image": image,
            "target": self.sku_to_index[row["sku_id"]],
            "sku_id": row["sku_id"],
            "is_known": bool(int(row["is_known"])),
            "is_labeled": bool(int(row["is_labeled"])),
            "attribute_values": values,
            "attribute_confidences": confidences,
            "index": index,
        }


def collate_retail(batch: list[dict]) -> dict:
    import torch

    first_image = batch[0]["image"]
    if isinstance(first_image, (tuple, list)):
        images = [torch.stack([sample["image"][view] for sample in batch]) for view in range(len(first_image))]
    else:
        images = torch.stack([sample["image"] for sample in batch])
    return {
        "image": images,
        "target": torch.tensor([sample["target"] for sample in batch], dtype=torch.long),
        "is_known": torch.tensor([sample["is_known"] for sample in batch], dtype=torch.bool),
        "is_labeled": torch.tensor([sample["is_labeled"] for sample in batch], dtype=torch.bool),
        "attribute_values": [sample["attribute_values"] for sample in batch],
        "attribute_confidences": torch.tensor(
            [sample["attribute_confidences"] for sample in batch], dtype=torch.float32
        ),
        "index": torch.tensor([sample["index"] for sample in batch]),
        "sku_id": [sample["sku_id"] for sample in batch],
    }

