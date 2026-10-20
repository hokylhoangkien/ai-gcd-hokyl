from __future__ import annotations

import torch
from torch.nn import functional as F


def supervised_loss(logits: torch.Tensor, targets: torch.Tensor, labeled: torch.Tensor) -> torch.Tensor:
    if not labeled.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[labeled], targets[labeled])


def consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    confidence_threshold: float = 0.0,
) -> torch.Tensor:
    target_a = F.softmax(logits_a.detach(), dim=-1)
    target_b = F.softmax(logits_b.detach(), dim=-1)
    confidence_a = target_a.max(dim=-1).values
    confidence_b = target_b.max(dim=-1).values
    accepted = (confidence_a >= confidence_threshold) & (confidence_b >= confidence_threshold)
    if not accepted.any():
        return (logits_a.sum() + logits_b.sum()) * 0.0
    loss_a = F.kl_div(F.log_softmax(logits_a[accepted], dim=-1), target_b[accepted], reduction="batchmean")
    loss_b = F.kl_div(F.log_softmax(logits_b[accepted], dim=-1), target_a[accepted], reduction="batchmean")
    return 0.5 * (loss_a + loss_b)


def cross_view_distillation_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
    warmup_teacher_temperature: float = 0.07,
    teacher_temperature: float = 0.04,
    student_temperature: float = 0.1,
) -> torch.Tensor:
    """DINO/SimGCD-style cross-view self-distillation with teacher sharpening."""
    progress = min(epoch / max(warmup_epochs, 1), 1.0)
    temperature = warmup_teacher_temperature + progress * (
        teacher_temperature - warmup_teacher_temperature
    )
    first_teacher = F.softmax(first_logits.detach() / temperature, dim=-1)
    second_teacher = F.softmax(second_logits.detach() / temperature, dim=-1)
    first_student = F.log_softmax(first_logits / student_temperature, dim=-1)
    second_student = F.log_softmax(second_logits / student_temperature, dim=-1)
    return -0.5 * (
        (second_teacher * first_student).sum(dim=-1).mean()
        + (first_teacher * second_student).sum(dim=-1).mean()
    )


def mean_entropy_regularizer(logits: torch.Tensor) -> torch.Tensor:
    mean_probability = F.softmax(logits, dim=-1).mean(dim=0)
    return (mean_probability * mean_probability.clamp_min(1e-8).log()).sum()


def instance_contrastive_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """NT-Xent over two augmented views without allocating label leakage."""
    batch_size = features_a.shape[0]
    if batch_size < 2:
        return (features_a.sum() + features_b.sum()) * 0.0
    features = F.normalize(torch.cat([features_a, features_b], dim=0), dim=-1)
    logits = features @ features.T / temperature
    diagonal = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
    logits = logits.masked_fill(diagonal, -torch.inf)
    targets = torch.arange(2 * batch_size, device=features.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)


def supervised_contrastive_loss(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    targets: torch.Tensor,
    labeled: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive objective restricted to genuinely labeled rows."""
    if labeled.sum() < 1:
        return (features_a.sum() + features_b.sum()) * 0.0
    features = F.normalize(torch.cat([features_a[labeled], features_b[labeled]], dim=0), dim=-1)
    labels = torch.cat([targets[labeled], targets[labeled]], dim=0)
    logits = features @ features.T / temperature
    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    valid = positive_mask.any(dim=1)
    mean_positive_log_prob = (log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
                              / positive_mask.sum(dim=1).clamp_min(1))
    return -mean_positive_log_prob[valid].mean()


def reliability_calibration_loss(
    reliabilities: torch.Tensor,
    image_features: torch.Tensor,
    attribute_features: torch.Tensor,
    ocr_confidences: torch.Tensor,
) -> torch.Tensor:
    agreement = F.cosine_similarity(image_features[:, None, :], attribute_features, dim=-1)
    agreement = ((agreement.detach() + 1.0) / 2.0).clamp(0.0, 1.0)
    # A conservative soft target: both OCR evidence and cross-modal agreement
    # must be high. This can later be replaced by labeled OCR correctness.
    target = torch.sqrt(ocr_confidences.clamp(0.0, 1.0) * agreement)
    return F.binary_cross_entropy(reliabilities.clamp(1e-6, 1 - 1e-6), target)


def pairwise_attribute_masks(
    values: list[list[str]],
    reliabilities: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build reliable must-link and cannot-link masks without SKU labels."""
    batch_size = len(values)
    positive = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=reliabilities.device)
    conflict = torch.zeros_like(positive)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            matches = 0
            conflicts = 0
            for k, (left, right) in enumerate(zip(values[i], values[j])):
                reliable = bool(
                    reliabilities[i, k].detach() >= threshold
                    and reliabilities[j, k].detach() >= threshold
                )
                if reliable and left and right:
                    matches += int(left.casefold() == right.casefold())
                    conflicts += int(left.casefold() != right.casefold())
            positive[i, j] = positive[j, i] = matches >= 2 and conflicts == 0
            conflict[i, j] = conflict[j, i] = conflicts > 0
    return positive, conflict


def conflict_aware_pair_loss(
    features: torch.Tensor,
    positive: torch.Tensor,
    conflict: torch.Tensor,
    positive_margin: float = 0.7,
    negative_margin: float = 0.3,
) -> torch.Tensor:
    similarity = features @ features.T
    losses = []
    if positive.any():
        losses.append(F.relu(positive_margin - similarity[positive]).mean())
    if conflict.any():
        losses.append(F.relu(similarity[conflict] - negative_margin).mean())
    return sum(losses, features.sum() * 0.0)


def cross_modal_anchor_consistency(
    image_features: torch.Tensor,
    attribute_features: torch.Tensor,
    targets: torch.Tensor,
    labeled: torch.Tensor,
    temperature: float = 0.4,
) -> torch.Tensor:
    """GET-style CICO adapted from pseudo text to observed retail attributes."""
    if not labeled.any():
        return (image_features.sum() + attribute_features.sum()) * 0.0
    classes = torch.unique(targets[labeled])
    image_anchors = torch.stack([image_features[labeled & targets.eq(label)].mean(0) for label in classes]).detach()
    attribute_anchors = torch.stack(
        [attribute_features[labeled & targets.eq(label)].mean(0) for label in classes]
    ).detach()
    image_distribution = F.softmax(
        F.normalize(image_features, dim=-1) @ F.normalize(image_anchors, dim=-1).T / temperature, dim=-1
    )
    attribute_distribution = F.softmax(
        F.normalize(attribute_features, dim=-1) @ F.normalize(attribute_anchors, dim=-1).T / temperature, dim=-1
    )
    image_to_attribute = -(attribute_distribution.detach() * image_distribution.clamp_min(1e-8).log()).sum(-1)
    attribute_to_image = -(image_distribution.detach() * attribute_distribution.clamp_min(1e-8).log()).sum(-1)
    return (image_to_attribute + attribute_to_image).mean()


def ova_supervised_loss(distribution_logits: torch.Tensor, known_targets: torch.Tensor) -> torch.Tensor:
    """DebGCD/OVA positive-class plus hardest-negative objective."""
    if known_targets.numel() == 0:
        return distribution_logits.sum() * 0.0
    known_count = distribution_logits.shape[-1] // 2
    probabilities = F.softmax(distribution_logits.reshape(distribution_logits.shape[0], 2, known_count), dim=1)
    positive_mask = F.one_hot(known_targets, num_classes=probabilities.shape[-1]).float()
    negative_mask = 1.0 - positive_mask
    positive = -(probabilities[:, 1].clamp_min(1e-8).log() * positive_mask).sum(dim=1)
    hardest_negative = -(probabilities[:, 0].clamp_min(1e-8).log() * negative_mask).max(dim=1).values
    return (positive + hardest_negative).mean()


def ova_entropy(distribution_logits: torch.Tensor) -> torch.Tensor:
    known_count = distribution_logits.shape[-1] // 2
    probabilities = F.softmax(distribution_logits.reshape(distribution_logits.shape[0], 2, known_count), dim=1)
    return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1).mean()


def semantic_distribution_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    targets: torch.Tensor,
    labeled: torch.Tensor,
    entropy_weight: float = 0.1,
) -> torch.Tensor:
    supervised = ova_supervised_loss(first_logits[labeled], targets[labeled])
    if (~labeled).any():
        first_unlabeled = first_logits[~labeled]
        second_unlabeled = second_logits[~labeled]
        entropy = 0.5 * (ova_entropy(first_unlabeled) + ova_entropy(second_unlabeled))
        known_count = first_unlabeled.shape[-1] // 2
        first_probability = F.softmax(first_unlabeled.reshape(first_unlabeled.shape[0], 2, known_count), dim=1)
        second_probability = F.softmax(second_unlabeled.reshape(second_unlabeled.shape[0], 2, known_count), dim=1)
        consistency = (first_probability - second_probability).square().sum(dim=(1, 2)).mean()
    else:
        entropy = first_logits.sum() * 0.0
        consistency = first_logits.sum() * 0.0
    return supervised + entropy_weight * entropy + consistency


def distribution_certainty(distribution_logits: torch.Tensor) -> torch.Tensor:
    if distribution_logits.shape[0] == 0:
        return distribution_logits.new_empty((0,))
    known_count = distribution_logits.shape[-1] // 2
    probabilities = F.softmax(distribution_logits.reshape(distribution_logits.shape[0], 2, known_count), dim=1)
    predicted_known_class = probabilities[:, 1].argmax(dim=-1)
    row = torch.arange(len(probabilities), device=probabilities.device)
    unknown_probability = probabilities[row, 0, predicted_known_class]
    return (2.0 * unknown_probability - 1.0).abs().detach()


def auxiliary_debiased_loss(
    debiased_logits: torch.Tensor,
    base_logits: torch.Tensor,
    targets: torch.Tensor,
    labeled: torch.Tensor,
    certainty: torch.Tensor,
    confidence_threshold: float = 0.95,
) -> torch.Tensor:
    supervised = supervised_loss(debiased_logits, targets, labeled)
    if not (~labeled).any():
        return supervised
    teacher_probability = F.softmax(base_logits[~labeled].detach(), dim=-1)
    confidence, pseudo_target = teacher_probability.max(dim=-1)
    accepted = confidence.ge(confidence_threshold)
    if not accepted.any():
        return supervised
    per_sample = F.cross_entropy(debiased_logits[~labeled][accepted], pseudo_target[accepted], reduction="none")
    return supervised + (per_sample * certainty[accepted]).mean()


def hierarchical_contrastive_loss(
    first_features: torch.Tensor,
    second_features: torch.Tensor,
    hierarchy_labels: list[torch.Tensor],
    temperature: float = 0.1,
) -> torch.Tensor:
    """SelEx-style self-expertise over progressively coarser pseudo-labels."""
    total = (first_features.sum() + second_features.sum()) * 0.0
    for level, labels in enumerate(hierarchy_labels):
        all_valid = torch.ones_like(labels, dtype=torch.bool)
        total = total + supervised_contrastive_loss(
            first_features, second_features, labels, all_valid, temperature
        ) / (2 ** (level + 1))
    return total

