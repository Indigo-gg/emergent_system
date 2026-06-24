"""
Novelty filter: k-NN novelty scoring + adaptive threshold + dead universe detection.

Novelty is measured as the average distance to the k nearest neighbors
in the Novelty Archive. Higher scores = more novel.
"""

import numpy as np
from typing import Optional


def novelty_score(behavior_vec: np.ndarray, archive_vectors: np.ndarray,
                  k: int = 15) -> float:
    """
    Compute novelty score as average distance to k nearest neighbors.

    Args:
        behavior_vec: 12D behavior vector
        archive_vectors: (N, 12) array of archived behavior vectors
        k: number of nearest neighbors

    Returns:
        novelty score (higher = more novel). Returns inf if archive < k.
    """
    if len(archive_vectors) < k:
        return float('inf')

    distances = np.linalg.norm(archive_vectors - behavior_vec, axis=1)
    k_nearest = np.sort(distances)[:k]
    return float(np.mean(k_nearest))


class AdaptiveThreshold:
    """Self-adjusting novelty threshold based on archive history."""

    def __init__(self, cfg: dict):
        self.threshold = 0.5  # initial threshold
        self.stale_limit = cfg['novelty'].get('stale_generations', 10)
        self.stale_count = 0  # consecutive generations without novelty
        self.history = []  # past novelty scores

    def update(self, was_novel: bool, score: float = 0.0):
        """Update threshold based on whether novelty was found."""
        if score > 0 and np.isfinite(score):
            self.history.append(score)

        if was_novel:
            self.stale_count = 0
            # Raise threshold toward median of recent scores
            if len(self.history) >= 10:
                median = np.median(self.history[-100:])
                self.threshold = max(self.threshold, median * 0.8)
        else:
            self.stale_count += 1
            if self.stale_count >= self.stale_limit:
                # Lower threshold to encourage exploration
                self.threshold *= 0.8
                self.stale_count = 0

    def is_novel(self, score: float) -> bool:
        """Check if a novelty score exceeds the threshold."""
        return score > self.threshold


class DeadUniverseFilter:
    """Filters out dead/uninteresting simulation results."""

    def __init__(self, cfg: dict):
        self.min_survival_rate = cfg['novelty'].get('min_survival_rate', 0.1)
        self.min_speed_variance = cfg['novelty'].get('min_speed_variance', 0.001)
        self.max_entropy_ratio = cfg['novelty'].get('max_entropy_ratio', 0.95)

    def is_dead(self, features_12d: np.ndarray) -> bool:
        """
        Check if a simulation result is "dead" (uninteresting).

        features_12d layout:
            [0] spatial_entropy_mean
            [4] speed_variance_mean
            [10] survival_rate

        Returns True if the result should be filtered out.
        """
        if len(features_12d) < 11:
            return True

        survival_rate = features_12d[10]
        entropy_mean = features_12d[0]
        speed_var_mean = features_12d[4]

        # Dead: too few survivors
        if survival_rate < self.min_survival_rate:
            return True

        # Dead: completely uniform distribution
        if entropy_mean > self.max_entropy_ratio:
            return True

        # Dead: all particles static
        if speed_var_mean < self.min_speed_variance:
            return True

        return False


def compute_novelty_archive_stats(archive: list) -> dict:
    """Compute statistics about the novelty archive."""
    if not archive:
        return {'size': 0, 'mean_score': 0.0, 'median_score': 0.0}

    scores = [e['novelty_score'] for e in archive if np.isfinite(e.get('novelty_score', 0))]
    return {
        'size': len(archive),
        'mean_score': float(np.mean(scores)) if scores else 0.0,
        'median_score': float(np.median(scores)) if scores else 0.0,
    }
