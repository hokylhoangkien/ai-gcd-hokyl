from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..evaluation.class_count import estimate_class_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate total GCD class count from an exported feature NPZ")
    parser.add_argument("--features", required=True, help="NPZ with features, targets, and labeled_mask arrays")
    parser.add_argument("--min-classes", type=int, required=True)
    parser.add_argument("--max-classes", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--silhouette-weight", type=float, default=0.5)
    parser.add_argument("--output", default="class_count_estimate.json")
    args = parser.parse_args()

    archive = np.load(args.features)
    estimated, candidates = estimate_class_count(
        archive["features"], archive["targets"], archive["labeled_mask"].astype(bool),
        args.min_classes, args.max_classes, args.step, args.seed, args.silhouette_weight,
    )
    result = {"estimated_total_classes": estimated, "candidates": [candidate.__dict__ for candidate in candidates]}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

