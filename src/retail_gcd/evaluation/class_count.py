from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ..metrics import cluster_accuracy


@dataclass(frozen=True)
class ClassCountCandidate:
    total_classes: int
    labeled_accuracy: float
    silhouette: float
    combined_score: float


def _subsample(features: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(features) <= maximum:
        return features
    generator = np.random.default_rng(seed)
    return features[generator.choice(len(features), maximum, replace=False)]


def estimate_class_count(
    features: np.ndarray,
    labels: np.ndarray,
    labeled_mask: np.ndarray,
    minimum_classes: int,
    maximum_classes: int,
    step: int = 1,
    seed: int = 1,
    silhouette_weight: float = 0.5,
    silhouette_samples: int = 5000,
) -> tuple[int, list[ClassCountCandidate]]:
    """Estimate total K without using labels of unlabeled samples.

    Candidate K is scored by clustering agreement on labeled samples plus an
    intrinsic silhouette term over all features. This is an explicit heuristic,
    not an oracle estimator, and must be reported separately from oracle-K.
    """
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(labeled_mask):
        raise ValueError("features, labels and labeled_mask shapes are inconsistent")
    known_classes = len(np.unique(labels[labeled_mask]))
    minimum_classes = max(minimum_classes, known_classes)
    if maximum_classes < minimum_classes:
        raise ValueError("maximum_classes must be >= minimum_classes and known class count")

    candidates: list[ClassCountCandidate] = []
    for total_classes in range(minimum_classes, maximum_classes + 1, step):
        kmeans = KMeans(n_clusters=total_classes, n_init=10, random_state=seed)
        assignments = kmeans.fit_predict(features)
        labeled_accuracy = cluster_accuracy(labels[labeled_mask], assignments[labeled_mask])
        sample = _subsample(features, silhouette_samples, seed)
        if len(sample) <= total_classes:
            silhouette = -1.0
        else:
            sample_assignments = kmeans.predict(sample)
            silhouette = float(silhouette_score(sample, sample_assignments, metric="cosine"))
        normalized_silhouette = (silhouette + 1.0) / 2.0
        combined = (1.0 - silhouette_weight) * labeled_accuracy + silhouette_weight * normalized_silhouette
        candidates.append(ClassCountCandidate(total_classes, labeled_accuracy, silhouette, combined))
    best = max(candidates, key=lambda candidate: candidate.combined_score)
    return best.total_classes, candidates

