from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..metrics import gcd_metrics, optimal_assignment


def _macro_accuracy(targets: np.ndarray, mapped_predictions: np.ndarray) -> float:
    scores = []
    for label in np.unique(targets):
        mask = targets == label
        scores.append(float((mapped_predictions[mask] == targets[mask]).mean()))
    return float(np.mean(scores)) if scores else float("nan")


def build_evaluation_report(
    targets: np.ndarray,
    predictions: np.ndarray,
    known_mask: np.ndarray,
    sibling_groups: np.ndarray | None = None,
) -> dict[str, float]:
    report = gcd_metrics(targets, predictions, known_mask)
    assignment = optimal_assignment(targets, predictions)
    mapped = np.asarray([assignment.get(int(prediction), -1) for prediction in predictions])
    report["macro_accuracy"] = _macro_accuracy(targets, mapped)

    frequencies = Counter(int(target) for target in targets)
    ordered = sorted(frequencies, key=frequencies.get, reverse=True)
    thirds = np.array_split(np.asarray(ordered), 3)
    for name, labels in zip(("head", "medium", "tail"), thirds):
        mask = np.isin(targets, labels)
        report[f"{name}_accuracy"] = float((mapped[mask] == targets[mask]).mean()) if mask.any() else float("nan")

    if sibling_groups is not None:
        group_to_classes: dict[object, set[int]] = defaultdict(set)
        for target, group in zip(targets, sibling_groups):
            if group not in (None, ""):
                group_to_classes[group].add(int(target))
        sibling_group_names = {group for group, classes in group_to_classes.items() if len(classes) > 1}
        mask = np.asarray([group in sibling_group_names for group in sibling_groups])
        report["sibling_accuracy"] = float((mapped[mask] == targets[mask]).mean()) if mask.any() else float("nan")
        report["sibling_samples"] = int(mask.sum())
    return report

