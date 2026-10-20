from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import RetailGCDDataset, collate_retail
from ..evaluation.report import build_evaluation_report
from ..model import create_model
from ..train import build_transforms
from ..utils.checkpoint import load_checkpoint


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retail GCD checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--head", choices=("primary", "debiased", "visual", "attribute"), default="primary"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    config, mapping = checkpoint["config"], checkpoint["class_mapping"]
    data_config = config["data"]
    dataset = RetailGCDDataset(
        data_config["manifest"], data_config["image_root"], build_transforms(False),
        tuple(args.splits), data_config["attributes"], mapping,
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
    predictions, targets, known, siblings = [], [], [], []
    sibling_index = data_config["attributes"].index("product_line") if "product_line" in data_config["attributes"] else None
    for batch in tqdm(loader, desc="evaluate"):
        output = model(
            batch["image"].to(device), batch["attribute_values"],
            batch["attribute_confidences"].to(device), config["train"]["temperature"],
        )
        selected_logits = {
            "primary": output.logits,
            "debiased": output.debiased_logits,
            "visual": output.visual_logits,
            "attribute": output.attribute_logits,
        }[args.head]
        predictions.append(selected_logits.argmax(dim=-1).cpu().numpy())
        targets.append(batch["target"].numpy())
        known.append(batch["is_known"].numpy())
        if sibling_index is not None:
            siblings.extend(values[sibling_index] for values in batch["attribute_values"])
    report = build_evaluation_report(
        np.concatenate(targets), np.concatenate(predictions), np.concatenate(known),
        np.asarray(siblings) if sibling_index is not None else None,
    )
    rendered = json.dumps(report, indent=2, allow_nan=True)
    output_path = Path(args.output) if args.output else Path(args.checkpoint).with_name(f"evaluation_{args.head}.json")
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

