from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


ATTRIBUTES = ("brand", "product_line", "variant", "volume", "pack_count", "packaging")
REQUIRED = {"image_path", "sku_id", "source_split"}


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "known"}


def prepare(args: argparse.Namespace) -> None:
    input_path = Path(args.annotations)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Input annotations missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Input annotation file is empty")

    rng = random.Random(args.seed)
    classes = sorted({row["sku_id"].strip() for row in rows})
    explicit_known = "is_known" in rows[0] and any(row.get("is_known", "").strip() for row in rows)
    if explicit_known:
        status_by_class: dict[str, set[bool]] = defaultdict(set)
        for row in rows:
            status_by_class[row["sku_id"].strip()].add(parse_bool(row["is_known"]))
        inconsistent = [sku for sku, statuses in status_by_class.items() if len(statuses) != 1]
        if inconsistent:
            raise ValueError(f"Classes marked both known and novel: {inconsistent[:10]}")
        known_classes = {sku for sku, statuses in status_by_class.items() if next(iter(statuses))}
    else:
        shuffled = classes.copy()
        rng.shuffle(shuffled)
        known_classes = set(shuffled[: round(len(shuffled) * args.known_ratio)])

    # Select whole capture sessions for supervision when session_id exists.
    groups_by_class: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["source_split"].strip().lower() == "train" and row["sku_id"].strip() in known_classes:
            group = row.get("session_id", "").strip() or row["image_path"].strip()
            if group not in groups_by_class[row["sku_id"].strip()]:
                groups_by_class[row["sku_id"].strip()].append(group)
    labeled_groups: set[tuple[str, str]] = set()
    for sku, groups in groups_by_class.items():
        rng.shuffle(groups)
        count = max(1, round(len(groups) * args.labeled_ratio))
        labeled_groups.update((sku, group) for group in groups[:count])

    output_rows: list[dict[str, object]] = []
    for row in rows:
        sku = row["sku_id"].strip()
        source_split = row["source_split"].strip().lower()
        if source_split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid source_split {source_split!r}; expected train/val/test")
        known = sku in known_classes
        group = row.get("session_id", "").strip() or row["image_path"].strip()
        labeled = source_split == "train" and known and (sku, group) in labeled_groups
        split = ("labeled" if labeled else "unlabeled") if source_split == "train" else source_split
        output: dict[str, object] = {
            "image_path": row["image_path"].strip(),
            "sku_id": sku,
            "split": split,
            "is_known": int(known),
            "is_labeled": int(labeled),
        }
        for attribute in ATTRIBUTES:
            value = row.get(attribute, "").strip()
            confidence = row.get(f"{attribute}_conf", "").strip()
            output[attribute] = value
            # Missing confidence means unknown reliability, not oracle certainty.
            output[f"{attribute}_conf"] = float(confidence) if confidence else 0.0
        output["source"] = row.get("source", "planogram").strip() or "planogram"
        output["session_id"] = row.get("session_id", "").strip()
        output_rows.append(output)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_path", "sku_id", "split", "is_known", "is_labeled",
        *[field for name in ATTRIBUTES for field in (name, f"{name}_conf")],
        "source", "session_id",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"wrote {len(output_rows)} rows; {len(known_classes)} known / "
        f"{len(classes) - len(known_classes)} novel classes to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert crop annotations to the retail GCD manifest")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", default="data/processed/planogram_manifest.csv")
    parser.add_argument("--known-ratio", type=float, default=0.5)
    parser.add_argument("--labeled-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()

