import csv
from pathlib import Path

import numpy as np
import json
from pathlib import Path

from retail_gcd.manifest import validate_manifest
from retail_gcd.metrics import gcd_metrics
from retail_gcd.losses import (
    consistency_loss,
    distribution_certainty,
    hierarchical_contrastive_loss,
    semantic_distribution_loss,
)
from retail_gcd.config import ConfigurationError, load_config, validate_config
from retail_gcd.evaluation.class_count import estimate_class_count
from retail_gcd.evaluation.hierarchy import build_hierarchy
import torch


def test_global_assignment_metrics() -> None:
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([1, 1, 0, 0])
    metrics = gcd_metrics(targets, predictions, np.array([True, True, False, False]))
    assert metrics["all_acc"] == 1.0
    assert metrics["old_acc"] == 1.0
    assert metrics["new_acc"] == 1.0


def test_example_manifest_without_image_check() -> None:
    summary = validate_manifest("data/manifest.example.csv", "data/images", check_images=False)
    assert summary["samples"] == 3
    assert not summary["errors"]


def test_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fields = [
        "image_path", "sku_id", "split", "is_known", "is_labeled",
        *[field for name in ("brand", "product_line", "variant", "volume", "pack_count", "packaging")
          for field in (name, f"{name}_conf")],
    ]
    base = {field: "" for field in fields}
    for name in ("brand", "product_line", "variant", "volume", "pack_count", "packaging"):
        base[f"{name}_conf"] = "0"
    rows = [
        {**base, "image_path": "same.jpg", "sku_id": "known", "split": "labeled", "is_known": "1", "is_labeled": "1"},
        {**base, "image_path": ".\\same.jpg", "sku_id": "novel", "split": "unlabeled", "is_known": "0", "is_labeled": "0"},
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        validate_manifest(manifest, tmp_path, check_images=False)
    except ValueError as error:
        assert "duplicate image_path" in str(error)
    else:
        raise AssertionError("normalized duplicate paths must be rejected")


def test_manifest_rejects_duplicate_content_across_splits(tmp_path: Path) -> None:
    (tmp_path / "first.jpg").write_bytes(b"identical image payload")
    (tmp_path / "second.jpg").write_bytes(b"identical image payload")
    manifest = tmp_path / "manifest.csv"
    fields = [
        "image_path", "sku_id", "split", "is_known", "is_labeled",
        *[field for name in ("brand", "product_line", "variant", "volume", "pack_count", "packaging")
          for field in (name, f"{name}_conf")],
    ]
    base = {field: "" for field in fields}
    for name in ("brand", "product_line", "variant", "volume", "pack_count", "packaging"):
        base[f"{name}_conf"] = "0"
    rows = [
        {**base, "image_path": "first.jpg", "sku_id": "known", "split": "labeled", "is_known": "1", "is_labeled": "1"},
        {**base, "image_path": "second.jpg", "sku_id": "novel", "split": "unlabeled", "is_known": "0", "is_labeled": "0"},
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        validate_manifest(manifest, tmp_path)
    except ValueError as error:
        assert "duplicate image content" in str(error)
    else:
        raise AssertionError("byte-identical images must be rejected")


def test_consistency_threshold_rejects_uncertain_pseudo_labels() -> None:
    logits_a = torch.zeros(2, 3, requires_grad=True)
    logits_b = torch.zeros(2, 3, requires_grad=True)
    loss = consistency_loss(logits_a, logits_b, confidence_threshold=0.8)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits_a.grad is not None


def test_planogram_config_is_valid() -> None:
    config = load_config("configs/planogram_proposed.json")
    assert config["model"]["fusion_mode"] == "proposed"
    grocery = load_config("configs/grocery_store_proposed.json")
    preflight = load_config("configs/grocery_store_preflight.json")
    assert grocery["data"]["manifest"].endswith("grocery_store_manifest.csv")
    assert preflight["train"]["max_train_batches"] == 1
    for path in Path("configs/generated").glob("*.json"):
        load_config(path)


def test_checkpoint_selection_rejects_test_split() -> None:
    config = load_config("configs/planogram_proposed.json")
    config["data"]["validation_splits"] = ["test"]
    try:
        validate_config(config)
    except ConfigurationError as error:
        assert "must not contain 'test'" in str(error)
    else:
        raise AssertionError("test split must never be accepted for checkpoint selection")


def test_training_augmentation_keeps_product_text_direction() -> None:
    config = load_config("configs/planogram_proposed.json")
    assert config["data"]["horizontal_flip"] is False


def test_class_count_estimation_returns_candidate_table() -> None:
    generator = np.random.default_rng(7)
    features = np.concatenate(
        [generator.normal(loc=center, scale=0.05, size=(12, 2)) for center in (-2.0, 0.0, 2.0)]
    )
    labels = np.repeat(np.arange(3), 12)
    labeled_mask = np.zeros(36, dtype=bool)
    labeled_mask[:6] = True
    labeled_mask[12:18] = True
    estimate, candidates = estimate_class_count(features, labels, labeled_mask, 2, 4, seed=3)
    assert 2 <= estimate <= 4
    assert [candidate.total_classes for candidate in candidates] == [2, 3, 4]


def test_debgcd_losses_accept_all_labeled_batch() -> None:
    logits_a = torch.randn(4, 6, requires_grad=True)
    logits_b = torch.randn(4, 6, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0])
    labeled = torch.ones(4, dtype=torch.bool)
    certainty = distribution_certainty(logits_a[~labeled])
    assert certainty.shape == (0,)
    loss = semantic_distribution_loss(logits_a, logits_b, targets, labeled)
    loss.backward()
    assert torch.isfinite(loss)


def test_selex_hierarchy_and_loss_are_finite() -> None:
    generator = np.random.default_rng(9)
    features = np.concatenate(
        [generator.normal(loc=center, scale=0.03, size=(8, 4)) for center in (-2.0, 0.0, 2.0, 4.0)]
    ).astype(np.float32)
    targets = np.repeat(np.arange(4), 8)
    labeled = np.zeros(32, dtype=bool)
    labeled[:4] = True
    labeled[8:12] = True
    hierarchy = build_hierarchy(features, targets, labeled, total_classes=4, levels=2, seed=2)
    assert len(hierarchy.assignments) == 2
    first = torch.tensor(features, requires_grad=True)
    second = first + 0.01 * torch.randn_like(first)
    labels = [torch.tensor(assignments) for assignments in hierarchy.assignments]
    loss = hierarchical_contrastive_loss(first, second, labels)
    assert torch.isfinite(loss)
    loss.backward()
