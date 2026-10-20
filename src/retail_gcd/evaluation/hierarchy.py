from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class HierarchyState:
    assignments: list[np.ndarray]
    prototypes: list[np.ndarray]
    radii: list[np.ndarray]


def _squared_distances(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)


def semi_supervised_kmeans(
    features: np.ndarray,
    targets: np.ndarray,
    labeled_mask: np.ndarray,
    total_classes: int,
    iterations: int = 20,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-means with labeled assignments fixed to their known class indices."""
    known_classes = np.unique(targets[labeled_mask]).astype(int)
    if not np.array_equal(known_classes, np.arange(len(known_classes))):
        raise ValueError("Known class indices must be contiguous from zero")
    known_count = len(known_classes)
    novel_count = total_classes - known_count
    if novel_count < 1:
        raise ValueError("GCD hierarchy requires at least one novel cluster")
    centers = [features[labeled_mask & (targets == label)].mean(axis=0) for label in known_classes]
    unlabeled_features = features[~labeled_mask]
    novel_initializer = KMeans(n_clusters=novel_count, n_init=10, random_state=seed).fit(unlabeled_features)
    centers = np.concatenate([np.stack(centers), novel_initializer.cluster_centers_], axis=0)
    assignments = _squared_distances(features, centers).argmin(axis=1)
    assignments[labeled_mask] = targets[labeled_mask]
    for _ in range(iterations):
        previous = assignments.copy()
        distances = _squared_distances(features, centers)
        assignments = distances.argmin(axis=1)
        assignments[labeled_mask] = targets[labeled_mask]
        for cluster in range(total_classes):
            members = features[assignments == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
        if np.array_equal(previous, assignments):
            break
    radii = np.asarray(
        [np.sqrt(_squared_distances(features[assignments == cluster], centers[cluster:cluster + 1])).mean()
         if np.any(assignments == cluster) else 0.0 for cluster in range(total_classes)]
    )
    return assignments, centers, radii


def build_hierarchy(
    features: np.ndarray,
    targets: np.ndarray,
    labeled_mask: np.ndarray,
    total_classes: int,
    levels: int = 3,
    iterations: int = 20,
    seed: int = 1,
) -> HierarchyState:
    assignments, centers, radii = semi_supervised_kmeans(
        features, targets, labeled_mask, total_classes, iterations, seed
    )
    assignment_levels, prototype_levels, radius_levels = [assignments], [centers], [radii]
    known_count = len(np.unique(targets[labeled_mask]))
    current_known, current_novel = known_count, total_classes - known_count
    for level in range(1, levels):
        next_known = max(current_known // 2, 1)
        next_novel = max(current_novel // 2, 1)
        known_map = KMeans(n_clusters=next_known, n_init=10, random_state=seed + level).fit_predict(
            centers[:current_known]
        )
        novel_map = KMeans(n_clusters=next_novel, n_init=10, random_state=seed + level).fit_predict(
            centers[current_known:]
        ) + next_known
        cluster_map = np.concatenate([known_map, novel_map])
        assignments = cluster_map[assignments]
        next_total = next_known + next_novel
        centers = np.stack([features[assignments == cluster].mean(axis=0) for cluster in range(next_total)])
        radii = np.asarray([
            np.sqrt(_squared_distances(features[assignments == cluster], centers[cluster:cluster + 1])).mean()
            for cluster in range(next_total)
        ])
        assignment_levels.append(assignments.copy())
        prototype_levels.append(centers.copy())
        radius_levels.append(radii.copy())
        current_known, current_novel = next_known, next_novel
    return HierarchyState(assignment_levels, prototype_levels, radius_levels)

