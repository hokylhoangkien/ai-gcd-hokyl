from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

DEFAULT_ATTRIBUTES = (
    "brand",
    "product_line",
    "variant",
    "volume",
    "pack_count",
    "packaging",
)
REQUIRED_COLUMNS = {"image_path", "sku_id", "split", "is_known", "is_labeled"}
VALID_SPLITS = {"labeled", "unlabeled", "val", "test"}


def _normalized_path(value: str) -> str:
    """Normalize manifest paths without requiring the target to exist."""
    return Path(value.strip().replace("\\", "/")).as_posix().casefold()


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        return list(reader)


def validate_manifest(
    manifest_path: str | Path,
    image_root: str | Path,
    attributes: tuple[str, ...] = DEFAULT_ATTRIBUTES,
    check_images: bool = True,
    check_duplicates: bool = True,
) -> dict[str, object]:
    rows = read_manifest(manifest_path)
    if not rows:
        raise ValueError("Manifest has no samples")

    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    sku_by_split: dict[str, set[str]] = {name: set() for name in VALID_SPLITS}
    known_skus: set[str] = set()
    novel_skus: set[str] = set()
    root = Path(image_root)
    resolved_root = root.resolve()
    path_rows: dict[str, list[tuple[int, str]]] = {}
    session_splits: dict[str, set[str]] = {}
    image_records: list[tuple[int, str, Path]] = []
    leaked_attributes: dict[str, list[int]] = {name: [] for name in attributes}

    for index, row in enumerate(rows, start=2):
        split = row["split"].strip().lower()
        sku = row["sku_id"].strip()
        if split not in VALID_SPLITS:
            errors.append(f"row {index}: invalid split {split!r}")
            continue
        if not sku:
            errors.append(f"row {index}: empty sku_id")
        split_counts[split] += 1
        sku_by_split[split].add(sku)

        image_value = row["image_path"].strip()
        normalized = _normalized_path(image_value)
        path_rows.setdefault(normalized, []).append((index, split))
        session = row.get("session_id", "").strip()
        if session:
            # labeled/unlabeled are both partitions of the source train split.
            source_split = "train" if split in {"labeled", "unlabeled"} else split
            session_splits.setdefault(session.casefold(), set()).add(source_split)

        try:
            known = int(row["is_known"])
            labeled = int(row["is_labeled"])
            if known not in (0, 1) or labeled not in (0, 1):
                raise ValueError
        except ValueError:
            errors.append(f"row {index}: is_known/is_labeled must be 0 or 1")
            continue

        if labeled != (split == "labeled"):
            errors.append(f"row {index}: is_labeled must be 1 exactly when split='labeled'")
        if labeled and not known:
            errors.append(f"row {index}: labeled samples must be known")
        (known_skus if known else novel_skus).add(sku)

        for attribute in attributes:
            confidence_name = f"{attribute}_conf"
            if attribute not in row or confidence_name not in row:
                errors.append(f"row {index}: missing {attribute}/{confidence_name}")
                continue
            try:
                confidence = float(row[confidence_name] or 0.0)
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError
            except ValueError:
                errors.append(f"row {index}: {confidence_name} must be in [0, 1]")
            value = row.get(attribute, "").strip()
            if value and value.casefold() == sku.casefold():
                leaked_attributes[attribute].append(index)

        if check_images:
            image_path = (root / image_value).resolve()
            if not image_path.is_relative_to(resolved_root):
                errors.append(f"row {index}: image_path escapes image_root: {image_value!r}")
                continue
            if not image_path.is_file():
                errors.append(f"row {index}: image not found: {image_path}")
            else:
                image_records.append((index, split, image_path))

    duplicate_paths = {path: records for path, records in path_rows.items() if len(records) > 1}
    for path, records in list(duplicate_paths.items())[:20]:
        errors.append(f"duplicate image_path {path!r} at rows {[row for row, _ in records]}")

    leaking_sessions = {session: splits for session, splits in session_splits.items() if len(splits) > 1}
    for session, splits in list(leaking_sessions.items())[:20]:
        errors.append(f"session {session!r} appears across source splits: {sorted(splits)}")

    for attribute, row_numbers in leaked_attributes.items():
        if row_numbers:
            errors.append(
                f"label leakage: {attribute!r} equals sku_id in {len(row_numbers)} rows "
                f"(examples: {row_numbers[:10]})"
            )

    duplicate_hash_groups = 0
    if check_images and check_duplicates:
        by_hash: dict[str, list[tuple[int, str, Path]]] = {}
        # Files with unique byte sizes cannot be exact duplicates. Hash only
        # size collisions, which makes large dataset preflight substantially
        # faster while preserving exact detection.
        by_size: dict[int, list[tuple[int, str, Path]]] = {}
        for record in image_records:
            by_size.setdefault(record[2].stat().st_size, []).append(record)
        for same_size in by_size.values():
            if len(same_size) < 2:
                continue
            for record in same_size:
                by_hash.setdefault(_sha256(record[2]), []).append(record)
        for records in by_hash.values():
            if len(records) < 2:
                continue
            duplicate_hash_groups += 1
            locations = [f"row {row} ({split}: {path})" for row, split, path in records]
            errors.append("duplicate image content: " + "; ".join(locations[:5]))

    overlap = known_skus & novel_skus
    if overlap:
        errors.append(f"SKUs marked both known and novel: {sorted(overlap)[:10]}")
    if not split_counts["labeled"] or not split_counts["unlabeled"]:
        errors.append("Both labeled and unlabeled samples are required")

    summary = {
        "samples": len(rows),
        "split_counts": dict(split_counts),
        "known_skus": len(known_skus),
        "novel_skus": len(novel_skus),
        "duplicate_hash_groups": duplicate_hash_groups,
        "errors": errors,
    }
    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors[:50]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a retail GCD CSV manifest")
    parser.add_argument("manifest")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-duplicates", action="store_true")
    args = parser.parse_args()
    summary = validate_manifest(
        args.manifest, args.image_root,
        check_images=not args.skip_images,
        check_duplicates=not args.skip_duplicates,
    )
    print(summary)


if __name__ == "__main__":
    main()

