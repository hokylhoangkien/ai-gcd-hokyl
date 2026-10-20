from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from .data import RetailGCDDataset, collate_retail
from .manifest import validate_manifest
from .config import load_config
from .losses import (
    conflict_aware_pair_loss,
    consistency_loss,
    cross_modal_anchor_consistency,
    cross_view_distillation_loss,
    auxiliary_debiased_loss,
    distribution_certainty,
    hierarchical_contrastive_loss,
    instance_contrastive_loss,
    mean_entropy_regularizer,
    pairwise_attribute_masks,
    reliability_calibration_loss,
    semantic_distribution_loss,
    supervised_contrastive_loss,
    supervised_loss,
)
from .metrics import gcd_metrics
from .model import create_model
from .evaluation.hierarchy import HierarchyState, build_hierarchy
from .utils.checkpoint import atomic_torch_save, save_run_metadata
from .utils.seed import make_worker_init_fn, seed_everything


class TwoViews:
    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, image):
        return self.transform(image), self.transform(image)


def build_transforms(train: bool, horizontal_flip: bool = False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if train:
        operations = [transforms.RandomResizedCrop(224, scale=(0.6, 1.0))]
        if horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip())
        operations.extend([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.ToTensor(), normalize])
        base = transforms.Compose(operations)
        return TwoViews(base)
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])


def move_batch(batch: dict, device: torch.device) -> dict:
    for key in ("target", "is_known", "is_labeled", "attribute_confidences", "index"):
        batch[key] = batch[key].to(device, non_blocking=True)
    batch["image"] = [image.to(device, non_blocking=True) for image in batch["image"]]
    return batch


def scheduled_weight(config: dict, component: str, epoch: int) -> float:
    base = config[f"lambda_{component}"]
    start = config.get(f"{component}_start_epoch", 0)
    warmup = config.get(f"{component}_warmup_epochs", 0)
    if epoch < start:
        return 0.0
    if warmup <= 0:
        return base
    return base * min((epoch - start + 1) / warmup, 1.0)


def train_epoch(
    model, loader, optimizer, scaler, config, device, epoch: int,
    hierarchy_state: HierarchyState | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    threshold = config["attribute_confidence_threshold"]
    batches_seen = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            # Both augmented views have identical text. Encode it once so the
            # frozen CLIP text tower does half the work and uses less memory.
            encoded_attributes = model.encode_attributes(batch["attribute_values"], device)
            first = model(
                batch["image"][0], batch["attribute_values"], batch["attribute_confidences"],
                config["temperature"], encoded_attributes,
            )
            second = model(
                batch["image"][1], batch["attribute_values"], batch["attribute_confidences"],
                config["temperature"], encoded_attributes,
            )
            supervised = supervised_loss(first.logits, batch["target"], batch["is_labeled"])
            warmup_epochs = max(config.get("pseudo_label_warmup_epochs", 0), 1)
            threshold = config["pseudo_label_threshold"] * min(epoch / warmup_epochs, 1.0)
            consistency = consistency_loss(first.logits, second.logits, threshold)
            distillation = cross_view_distillation_loss(
                first.logits * config["temperature"], second.logits * config["temperature"],
                epoch, config["teacher_temperature_warmup_epochs"],
                config["warmup_teacher_temperature"], config["teacher_temperature"],
                config["student_temperature"],
            )
            instance_contrastive = instance_contrastive_loss(
                first.fused_features, second.fused_features, config["contrastive_temperature"]
            )
            supervised_contrastive = supervised_contrastive_loss(
                first.fused_features, second.fused_features, batch["target"], batch["is_labeled"],
                config["contrastive_temperature"],
            )
            dual_branch = 0.25 * (
                supervised_loss(first.visual_logits, batch["target"], batch["is_labeled"])
                + supervised_loss(first.attribute_logits, batch["target"], batch["is_labeled"])
                + consistency_loss(first.visual_logits, second.visual_logits, threshold)
                + consistency_loss(first.attribute_logits, second.attribute_logits, threshold)
            )
            cico = cross_modal_anchor_consistency(
                first.image_features, first.attribute_summary, batch["target"], batch["is_labeled"]
            )
            semantic_distribution = semantic_distribution_loss(
                first.distribution_logits, second.distribution_logits,
                batch["target"], batch["is_labeled"],
            )
            certainty = distribution_certainty(first.distribution_logits[~batch["is_labeled"]])
            debiased = auxiliary_debiased_loss(
                first.debiased_logits, first.logits, batch["target"], batch["is_labeled"], certainty,
                config["debiased_pseudo_threshold"],
            )
            if hierarchy_state is not None:
                hierarchy_labels = [
                    torch.as_tensor(assignments[batch["index"].cpu().numpy()], device=device, dtype=torch.long)
                    for assignments in hierarchy_state.assignments
                ]
                hierarchy = hierarchical_contrastive_loss(
                    first.fused_features, second.fused_features, hierarchy_labels,
                    config["contrastive_temperature"],
                )
            else:
                hierarchy = first.fused_features.sum() * 0.0
            entropy = mean_entropy_regularizer(first.logits)
            calibration = reliability_calibration_loss(
                first.reliabilities,
                first.image_features,
                first.attribute_features,
                batch["attribute_confidences"],
            )
            positive, conflict = pairwise_attribute_masks(
                batch["attribute_values"], first.reliabilities, threshold
            )
            attribute = conflict_aware_pair_loss(first.fused_features, positive, torch.zeros_like(conflict))
            conflict_loss = conflict_aware_pair_loss(
                first.fused_features, torch.zeros_like(positive), conflict
            )
            loss = (
                config["lambda_supervised"] * supervised
                + config["lambda_consistency"] * consistency
                + config["lambda_distillation"] * distillation
                + config["lambda_instance_contrastive"] * instance_contrastive
                + config["lambda_supervised_contrastive"] * supervised_contrastive
                + scheduled_weight(config, "dual_branch", epoch) * dual_branch
                + scheduled_weight(config, "cico", epoch) * cico
                + scheduled_weight(config, "sdl", epoch) * semantic_distribution
                + scheduled_weight(config, "adl", epoch) * debiased
                + scheduled_weight(config, "hierarchy", epoch) * hierarchy
                + config["lambda_entropy"] * entropy
                + config["lambda_attribute"] * attribute
                + scheduled_weight(config, "conflict", epoch) * conflict_loss
                + scheduled_weight(config, "calibration", epoch) * calibration
            )
        scaler.scale(loss).backward()
        if config.get("gradient_clip_norm"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
        scaler.step(optimizer)
        scaler.update()
        values = {
            "loss": loss,
            "supervised": supervised,
            "consistency": consistency,
            "distillation": distillation,
            "instance_contrastive": instance_contrastive,
            "supervised_contrastive": supervised_contrastive,
            "dual_branch": dual_branch,
            "cico": cico,
            "semantic_distribution": semantic_distribution,
            "debiased": debiased,
            "hierarchy": hierarchy,
            "attribute": attribute,
            "conflict": conflict_loss,
            "calibration": calibration,
        }
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
        batches_seen += 1
        if config.get("max_train_batches") and batches_seen >= config["max_train_batches"]:
            break
    return {name: value / max(batches_seen, 1) for name, value in totals.items()}


@torch.no_grad()
def refresh_hierarchy(model, loader, total_classes: int, config: dict, device: torch.device) -> HierarchyState:
    model.eval()
    feature_parts, target_parts, labeled_parts, index_parts = [], [], [], []
    for batch in tqdm(loader, desc="hierarchy", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        confidences = batch["attribute_confidences"].to(device, non_blocking=True)
        output = model(images, batch["attribute_values"], confidences, config["temperature"])
        feature_parts.append(output.fused_features.cpu().numpy())
        target_parts.append(batch["target"].numpy())
        labeled_parts.append(batch["is_labeled"].numpy())
        index_parts.append(batch["index"].numpy())
    features = np.concatenate(feature_parts)
    targets = np.concatenate(target_parts)
    labeled = np.concatenate(labeled_parts).astype(bool)
    indices = np.concatenate(index_parts)
    order = np.argsort(indices)
    return build_hierarchy(
        features[order], targets[order], labeled[order], total_classes,
        levels=config["hierarchy_levels"], iterations=config["hierarchy_kmeans_iterations"],
        seed=config.get("seed", 1),
    )


@torch.no_grad()
def evaluate(
    model, loader, temperature: float, device: torch.device, max_batches: int | None = None
) -> dict[str, float]:
    model.eval()
    predictions, targets, known = [], [], []
    for batch_index, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        confidences = batch["attribute_confidences"].to(device, non_blocking=True)
        output = model(images, batch["attribute_values"], confidences, temperature)
        predictions.append(output.logits.argmax(dim=-1).cpu().numpy())
        targets.append(batch["target"].numpy())
        known.append(batch["is_known"].numpy())
        if max_batches and batch_index + 1 >= max_batches:
            break
    if not targets:
        raise RuntimeError("Validation loader is empty; provide a non-empty val split for checkpoint selection")
    return gcd_metrics(np.concatenate(targets), np.concatenate(predictions), np.concatenate(known))


def run(config_path: str) -> None:
    config = load_config(config_path)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config, model_config, train_config = config["data"], config["model"], config["train"]

    # Fail before model/download/GPU allocation when the manifest leaks or
    # contains exact duplicate images.
    manifest_summary = validate_manifest(
        data_config["manifest"], data_config["image_root"],
        attributes=tuple(data_config["attributes"]),
    )
    print(json.dumps({"manifest": manifest_summary}))

    train_dataset = RetailGCDDataset(
        data_config["manifest"], data_config["image_root"],
        build_transforms(True, data_config.get("horizontal_flip", False)),
        splits=("labeled", "unlabeled"), attributes=data_config["attributes"]
    )
    eval_dataset = RetailGCDDataset(
        data_config["manifest"], data_config["image_root"], build_transforms(False),
        splits=tuple(data_config.get("validation_splits", ["val"])), attributes=data_config["attributes"],
        sku_to_index=train_dataset.sku_to_index,
    )
    if len(eval_dataset) == 0:
        raise ValueError(
            "No samples were found in data.validation_splits. Add a validation split; "
            "do not use the test split for checkpoint selection."
        )
    hierarchy_dataset = RetailGCDDataset(
        data_config["manifest"], data_config["image_root"], build_transforms(False),
        splits=("labeled", "unlabeled"), attributes=data_config["attributes"],
        sku_to_index=train_dataset.sku_to_index,
    )
    # Oracle class-count protocol. Estimated-K is intentionally a separate
    # experiment so papers cannot silently mix the two settings.
    num_clusters = len(train_dataset.sku_to_index)
    num_known_classes = len({row["sku_id"] for row in train_dataset.rows if bool(int(row["is_known"]))})
    model_config["num_known_classes"] = num_known_classes
    model = create_model(
        model_config, num_clusters, device, data_config["attributes"], num_known_classes
    )
    # Keep temporarily frozen visual parameters in optimizer groups so they
    # start updating when the configured warm-up ends.
    optimizer = AdamW(
        model.parameters(), lr=train_config["learning_rate"], weight_decay=train_config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_config["epochs"], eta_min=train_config.get("min_learning_rate", 1e-6)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_config["amp"] and device.type == "cuda")
    train_sampler = None
    if data_config.get("balance_labeled_unlabeled", True):
        labeled_count = sum(row["is_labeled"] == "1" for row in train_dataset.rows)
        unlabeled_count = len(train_dataset) - labeled_count
        unlabeled_weight = labeled_count / max(unlabeled_count, 1)
        sample_weights = [1.0 if row["is_labeled"] == "1" else unlabeled_weight for row in train_dataset.rows]
        generator = torch.Generator().manual_seed(config["seed"])
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=generator)
    train_loader = DataLoader(
        train_dataset, batch_size=data_config["batch_size"], shuffle=train_sampler is None,
        sampler=train_sampler, drop_last=True,
        num_workers=data_config["num_workers"], pin_memory=device.type == "cuda", collate_fn=collate_retail,
        worker_init_fn=make_worker_init_fn(config["seed"]),
        persistent_workers=data_config["num_workers"] > 0,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=data_config.get("eval_batch_size", data_config["batch_size"]), shuffle=False,
        num_workers=data_config["num_workers"], pin_memory=device.type == "cuda", collate_fn=collate_retail,
        worker_init_fn=make_worker_init_fn(config["seed"]),
        persistent_workers=data_config["num_workers"] > 0,
    )
    hierarchy_loader = DataLoader(
        hierarchy_dataset, batch_size=data_config.get("eval_batch_size", data_config["batch_size"]),
        shuffle=False, num_workers=data_config["num_workers"], pin_memory=device.type == "cuda",
        collate_fn=collate_retail, worker_init_fn=make_worker_init_fn(config["seed"]),
        persistent_workers=data_config["num_workers"] > 0,
    )

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_run_metadata(output_dir, config, train_dataset.sku_to_index)
    history = []
    start_epoch = 0
    best_harmonic = float("-inf")
    resume_path = config.get("resume")
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_harmonic = checkpoint.get("best_harmonic", best_harmonic)
    hierarchy_state = None
    for epoch in range(start_epoch, train_config["epochs"]):
        model.set_visual_trainable(epoch >= model_config.get("freeze_visual_until_epoch", 0))
        if epoch >= train_config.get("hierarchy_start_epoch", 0) and train_config["lambda_hierarchy"] > 0 and (
            hierarchy_state is None or epoch % train_config["hierarchy_refresh_interval"] == 0
        ):
            hierarchy_config = {**train_config, "seed": config["seed"] + epoch}
            hierarchy_state = refresh_hierarchy(model, hierarchy_loader, num_clusters, hierarchy_config, device)
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, train_config, device, epoch, hierarchy_state
        )
        eval_metrics = evaluate(
            model, eval_loader, train_config["temperature"], device,
            max_batches=train_config.get("max_eval_batches"),
        )
        record = {"epoch": epoch + 1, "train": train_metrics, "eval": eval_metrics}
        history.append(record)
        print(json.dumps(record))
        scheduler.step()
        harmonic = eval_metrics.get("harmonic", float("nan"))
        improved = np.isfinite(harmonic) and harmonic > best_harmonic
        if improved:
            best_harmonic = harmonic
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
            "best_harmonic": best_harmonic, "class_mapping": train_dataset.sku_to_index,
            "config": config,
        }
        atomic_torch_save(state, output_dir / "last.pt")
        if improved:
            atomic_torch_save(state, output_dir / "best.pt")
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/planogram_proposed.json")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

