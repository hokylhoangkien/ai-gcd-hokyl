from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def optimal_assignment(targets: np.ndarray, predictions: np.ndarray) -> dict[int, int]:
    size = int(max(targets.max(), predictions.max()) + 1)
    matrix = np.zeros((size, size), dtype=np.int64)
    for prediction, target in zip(predictions, targets):
        matrix[prediction, target] += 1
    rows, columns = linear_sum_assignment(matrix.max() - matrix)
    return {int(row): int(column) for row, column in zip(rows, columns)}


def cluster_accuracy(
    targets: np.ndarray,
    predictions: np.ndarray,
    assignment: dict[int, int] | None = None,
) -> float:
    assignment = assignment or optimal_assignment(targets, predictions)
    mapped = np.asarray([assignment.get(int(prediction), -1) for prediction in predictions])
    return float((mapped == targets).mean())


def gcd_metrics(targets: np.ndarray, predictions: np.ndarray, known_mask: np.ndarray) -> dict[str, float]:
    # Fit one global Hungarian assignment. Re-fitting independently on Old and
    # New can map one predicted cluster to different targets and inflate scores.
    assignment = optimal_assignment(targets, predictions)

    def safe_acc(mask: np.ndarray) -> float:
        return cluster_accuracy(targets[mask], predictions[mask], assignment) if mask.any() else float("nan")

    old = safe_acc(known_mask)
    new = safe_acc(~known_mask)
    harmonic = 2 * old * new / (old + new) if old + new > 0 else 0.0
    return {
        "all_acc": cluster_accuracy(targets, predictions, assignment),
        "old_acc": old,
        "new_acc": new,
        "harmonic": harmonic,
        "nmi": float(normalized_mutual_info_score(targets, predictions)),
        "ari": float(adjusted_rand_score(targets, predictions)),
    }

