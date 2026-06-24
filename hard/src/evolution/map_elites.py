"""
MAP-Elites grid + Novelty Archive: hybrid feature archive system.

3D MAP-Elites grid: dense competition in (entropy, islands, fft_amp) space.
Novelty Archive: open-ended archiving of novel behavior patterns.
"""

import json
import random
import numpy as np
from typing import Optional

from src.novelty.filter import novelty_score, AdaptiveThreshold, DeadUniverseFilter


class MAPElites:
    """
    3D MAP-Elites grid for dense competition.

    Grid dimensions: entropy_mean × islands_mean × fft_amp_1
    Resolution: 15 per axis → 3,375 cells
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.resolution = cfg['map_elites'].get('resolution_per_dim', 15)
        self.grid = {}  # (i, j, k) → cell data

        # Feature ranges for discretization (will be adaptive)
        self._ranges = {
            'entropy': (0.0, 1.0),
            'islands': (0.0, 50.0),
            'fft_amp': (0.0, 1.0),
        }

    def try_archive(self, genome, features_3d: tuple, fitness: float,
                    features_12d: np.ndarray = None, seed: int = 0) -> bool:
        """
        Try to archive a genome in the grid.
        Returns True if the genome was archived (or replaced an inferior one).
        """
        key = self._discretize(features_3d)
        formula = genome.to_formula() if hasattr(genome, 'to_formula') else ''

        cell_data = {
            'genome': genome,
            'fitness': fitness,
            'features_3d': features_3d,
            'features_12d': features_12d.tolist() if features_12d is not None else [],
            'formula': formula,
            'seed': seed,
            'generation': genome.generation if hasattr(genome, 'generation') else 0,
        }

        if key not in self.grid or fitness > self.grid[key]['fitness']:
            self.grid[key] = cell_data
            return True
        return False

    def random_non_empty_cell(self, rng: random.Random = None) -> Optional[dict]:
        """Randomly select a non-empty grid cell (weighted by fitness)."""
        if not self.grid:
            return None
        if rng is None:
            rng = random.Random()

        entries = list(self.grid.values())
        weights = [max(e['fitness'], 0.001) for e in entries]
        total = sum(weights)
        weights = [w / total for w in weights]
        return rng.choices(entries, weights=weights, k=1)[0]

    def get_filled_count(self) -> int:
        """Number of occupied grid cells."""
        return len(self.grid)

    def get_total_cells(self) -> int:
        """Total number of grid cells."""
        return self.resolution ** 3

    def get_fill_ratio(self) -> float:
        """Fraction of occupied cells."""
        total = self.get_total_cells()
        return len(self.grid) / total if total > 0 else 0.0

    def get_all_formulas(self) -> list:
        """Get all archived formulas with their features."""
        results = []
        for key, data in self.grid.items():
            results.append({
                'key': key,
                'formula': data.get('formula', ''),
                'fitness': data['fitness'],
                'features_3d': data['features_3d'],
            })
        return results

    def export_summary(self) -> str:
        """Export a human-readable summary of the grid."""
        lines = [f"MAP-Elites Grid: {len(self.grid)}/{self.get_total_cells()} cells filled"]
        lines.append(f"Fill ratio: {self.get_fill_ratio():.1%}")
        lines.append("")

        # Sort by fitness
        sorted_cells = sorted(self.grid.items(), key=lambda x: x[1]['fitness'], reverse=True)
        for i, (key, data) in enumerate(sorted_cells[:10]):
            lines.append(f"  #{i+1} [{key}] fit={data['fitness']:.4f} "
                        f"formula={data.get('formula', '')[:60]}")

        return '\n'.join(lines)

    def _discretize(self, features_3d: tuple) -> tuple:
        """Convert continuous 3D features to grid cell index."""
        entropy, islands, fft_amp = features_3d

        def _clamp_and_bin(val, range_tuple, n_bins):
            low, high = range_tuple
            val = max(low, min(high, val))
            if high == low:
                return 0
            bin_idx = int((val - low) / (high - low) * n_bins)
            return min(bin_idx, n_bins - 1)

        i = _clamp_and_bin(entropy, self._ranges['entropy'], self.resolution)
        j = _clamp_and_bin(islands, self._ranges['islands'], self.resolution)
        k = _clamp_and_bin(fft_amp, self._ranges['fft_amp'], self.resolution)
        return (i, j, k)


class NoveltyArchive:
    """
    Open-ended archive of novel behavior patterns.

    No grid limit — stores all behaviors judged as "novel" by k-NN distance.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.archive = []  # list of entry dicts
        self.k = cfg['novelty'].get('k_neighbors', 15)
        self.threshold_manager = AdaptiveThreshold(cfg)
        self.dead_filter = DeadUniverseFilter(cfg)

    def try_add(self, genome, behavior_vector: np.ndarray,
                fitness: float = 0.0, seed: int = 0) -> bool:
        """
        Try to add a genome to the novelty archive.
        Returns True if added (was novel enough and not dead).
        """
        # Dead universe filter
        if self.dead_filter.is_dead(behavior_vector):
            return False

        # Compute novelty score
        archive_vectors = self._get_vectors()
        score = novelty_score(behavior_vector, archive_vectors, self.k)

        # Check against adaptive threshold
        is_novel = self.threshold_manager.is_novel(score)
        self.threshold_manager.update(is_novel, score)

        if is_novel:
            formula = genome.to_formula() if hasattr(genome, 'to_formula') else ''
            self.archive.append({
                'genome': genome,
                'behavior_vector': behavior_vector.tolist(),
                'novelty_score': score,
                'fitness': fitness,
                'seed': seed,
                'generation': genome.generation if hasattr(genome, 'generation') else 0,
                'formula': formula,
            })
            return True

        return False

    def random_entry(self, rng: random.Random = None) -> Optional[dict]:
        """Select a random entry from the archive."""
        if not self.archive:
            return None
        if rng is None:
            rng = random.Random()
        return rng.choice(self.archive)

    def size(self) -> int:
        return len(self.archive)

    def get_all_formulas(self) -> list:
        """Get all archived formulas."""
        return [{
            'formula': e.get('formula', ''),
            'novelty_score': e.get('novelty_score', 0),
            'fitness': e.get('fitness', 0),
            'generation': e.get('generation', 0),
        } for e in self.archive]

    def get_stats(self) -> dict:
        """Get archive statistics."""
        if not self.archive:
            return {'size': 0, 'mean_score': 0.0, 'threshold': self.threshold_manager.threshold}
        scores = [e['novelty_score'] for e in self.archive if np.isfinite(e.get('novelty_score', 0))]
        return {
            'size': len(self.archive),
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'median_score': float(np.median(scores)) if scores else 0.0,
            'threshold': self.threshold_manager.threshold,
        }

    def _get_vectors(self) -> np.ndarray:
        """Get all behavior vectors as a numpy array."""
        if not self.archive:
            return np.zeros((0, 12), dtype=np.float32)
        vectors = [e['behavior_vector'] for e in self.archive]
        return np.array(vectors, dtype=np.float32)
