"""
Novelty filter: k-NN novelty scoring + adaptive threshold + dead universe detection.

Novelty is measured as the average distance to the k nearest neighbors
in the Novelty Archive. Higher scores = more novel.
"""

import numpy as np
from typing import Optional

# Feature normalization bounds (15D): (min, max) for each feature dimension.
# Used to normalize features before distance computation so no single feature
# dominates the Euclidean distance due to scale differences.
FEATURE_BOUNDS = np.array([
    (0.0, 1.0),      # [0]  spatial_entropy_mean (already normalized)
    (0.0, 0.5),      # [1]  spatial_entropy_var
    (0.0, 100.0),    # [2]  islands_mean (raw count)
    (0.0, 5000.0),   # [3]  islands_var
    (0.0, 10.0),     # [4]  speed_variance_mean
    (0.0, 5.0),      # [5]  fft_amp_1
    (0.0, 5.0),      # [6]  fft_amp_2
    (0.0, 5.0),      # [7]  fft_amp_3
    (-1000.0, 1000.0),  # [8]  angular_momentum_skew
    (0.0, 1.0),      # [9]  density_laplacian_var_mean
    (0.0, 1.0),      # [10] survival_rate (already normalized)
    (-1.0, 1.0),     # [11] autocorr_lag10 (already normalized)
    (0.0, 5.0),      # [12] nutrient_consume_mean
    (0.0, 5.0),      # [13] waste_peak_mean
    (-10.0, 10.0),   # [14] energy_skew
], dtype=np.float32)


def _normalize_features(vec: np.ndarray) -> np.ndarray:
    """Normalize feature vector to [0, 1] using fixed bounds."""
    v = np.asarray(vec, dtype=np.float32)
    mins = FEATURE_BOUNDS[:len(v), 0]
    maxs = FEATURE_BOUNDS[:len(v), 1]
    return (v - mins) / (maxs - mins + 1e-8)


def novelty_score(behavior_vec: np.ndarray, archive_vectors: np.ndarray,
                  k: int = 15) -> float:
    """
    Compute novelty score as average distance to k nearest neighbors.

    Args:
        behavior_vec: 12D behavior vector
        archive_vectors: (N, 12) array of archived behavior vectors
        k: number of nearest neighbors

    Returns:
        novelty score (higher = more novel). Returns large finite value if archive < k.
    """
    if len(archive_vectors) < k:
        # Bootstrap phase: return large but finite value
        return 1000.0

    # Normalize both query and archive vectors to [0, 1] per feature dimension
    norm_query = _normalize_features(behavior_vec)
    norm_archive = np.apply_along_axis(_normalize_features, 1, archive_vectors)

    distances = np.linalg.norm(norm_archive - norm_query, axis=1)
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
        # Only track finite scores for threshold adjustment
        if score > 0 and np.isfinite(score) and score < 100.0:
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
        if not np.isfinite(score):
            return False  # never auto-admit inf scores
        # Bootstrap phase: large finite score always counts as novel
        if score >= 100.0:
            return True
        return score > self.threshold


class DeadUniverseFilter:
    """Filters out dead/uninteresting simulation results (v6: 15D features)."""

    def __init__(self, cfg: dict):
        self.min_survival_rate = cfg['novelty'].get('min_survival_rate', 0.1)
        self.min_speed_variance = cfg['novelty'].get('min_speed_variance', 0.001)
        self.max_entropy_ratio = cfg['novelty'].get('max_entropy_ratio', 0.95)
        self.min_nutrient_consume = cfg['novelty'].get('min_nutrient_consume', 0.001)
        self.max_energy_skew = cfg['novelty'].get('max_energy_skew', 3.0)

    def is_dead(self, features: np.ndarray) -> bool:
        """
        Check if a simulation result is "dead" (uninteresting).

        Features layout (15D):
            [0]  spatial_entropy_mean
            [4]  speed_variance_mean
            [10] survival_rate
            [12] nutrient_consume_mean  (v6)
            [14] energy_skew            (v6)

        Returns True if the result should be filtered out.
        """
        if len(features) < 11:
            return True

        survival_rate = features[10]
        entropy_mean = features[0]
        speed_var_mean = features[4]

        # Dead: too few survivors
        if survival_rate < self.min_survival_rate:
            return True

        # Dead: completely uniform distribution
        if entropy_mean > self.max_entropy_ratio:
            return True

        # Dead: all particles static
        if speed_var_mean < self.min_speed_variance:
            return True

        # v6: Dead: particles not eating
        if len(features) > 12:
            nutrient_consume = features[12]
            if nutrient_consume < self.min_nutrient_consume:
                return True

        # v6: Dead: extreme energy monopoly (one particle hoards all energy)
        if len(features) > 14:
            energy_skew = features[14]
            if energy_skew > self.max_energy_skew:
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
