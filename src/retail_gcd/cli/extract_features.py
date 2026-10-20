from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import RetailGCDDataset, collate_retail
from ..model import create_model
from ..train import build_transforms
from ..utils.checkpoint import load_checkpoint


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Export fused features from a trained checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["labeled", "unlabeled"])
    parser.add_argument("--output", default="features.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    data_config = config["data"]
    mapping = checkpoint["class_mapping"]
    dataset = RetailGCDDataset(
        data_config["manifest"], data_config["image_root"], build_transforms(False),
        splits=tuple(args.splits), attributes=data_config["attributes"], sku_to_index=mapping,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=data_config["num_workers"],
        pin_memory=device.type == "cuda", collate_fn=collate_retail,
    )
    num_known = config["model"].get("num_known_classes") or len(
        {row["sku_id"] for row in dataset.rows if bool(int(row["is_known"]))}
    )
    model = create_model(config["model"], len(mapping), device, data_config["attributes"], num_known)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    features, targets, labeled, known, indices = [], [], [], [], []
    for batch in tqdm(loader, desc="extract"):
        images = batch["image"].to(device, non_blocking=True)
        confidences = batch["attribute_confidences"].to(device, non_blocking=True)
        output = model(images, batch["attribute_values"], confidences)
        features.append(output.fused_features.cpu().numpy())
        targets.append(batch["target"].numpy())
        labeled.append(batch["is_labeled"].numpy())
        known.append(batch["is_known"].numpy())
        indices.append(batch["index"].numpy())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.concatenate(features), targets=np.concatenate(targets),
        labeled_mask=np.concatenate(labeled), known_mask=np.concatenate(known),
        indices=np.concatenate(indices),
    )
    print(f"saved {sum(len(part) for part in features)} feature vectors to {args.output}")


if __name__ == "__main__":
    main()

