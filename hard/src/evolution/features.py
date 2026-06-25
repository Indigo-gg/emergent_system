"""
15-dimensional time-invariant feature extraction (v6: +3 environment features).

Captures dynamic emergent behavior via statistical moments and frequency
domain features that are phase-invariant.

Feature vector (15D):
  [0]  spatial_entropy_mean     — order vs chaos
  [1]  spatial_entropy_var      — stable vs fluctuating
  [2]  islands_mean             — aggregated vs dispersed
  [3]  islands_var              — stable vs split-merge
  [4]  speed_variance_mean      — activity level
  [5]  fft_amp_1                — oscillation mode 1
  [6]  fft_amp_2                — oscillation mode 2
  [7]  fft_amp_3                — oscillation mode 3
  [8]  angular_momentum_skew    — rotation vs translation
  [9]  density_laplacian_var_mean — uniform vs textured
  [10] survival_rate            — self-sustaining vs decaying
  [11] autocorr_lag10           — periodic vs random
  [12] nutrient_consume_mean    — average nutrient absorbed per step (v6)
  [13] waste_peak_mean          — average waste field maximum per step (v6)
  [14] energy_skew              — skewness of particle energy distribution (v6)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SampleSnapshot:
    """One sample's worth of metrics from the simulation."""
    spatial_entropy: float = 0.0
    island_count: int = 0
    speed_variance: float = 0.0
    angular_momentum: float = 0.0
    density_laplacian_var: float = 0.0
    alive_count: int = 0
    total_count: int = 0
    # v6: environment features
    nutrient_consumed: float = 0.0
    waste_peak: float = 0.0
    energy_skew: float = 0.0


class FeatureExtractor:
    """Extracts 15D time-invariant features from simulation samples (v6)."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.sample_interval = cfg['novelty'].get('sample_interval', 500)
        self.samples: list[SampleSnapshot] = []

    def reset(self):
        """Clear collected samples."""
        self.samples = []

    def sample(self, particles, cfg: dict, environment=None):
        """
        Collect one sample snapshot from the current simulation state.
        Should be called every sample_interval steps.

        Args:
            particles: ParticleSystem instance
            cfg: config dict
            environment: EnvironmentLayer instance (optional, for v6 env features)
        """
        pos_x = particles.pos_x.to_numpy()
        pos_y = particles.pos_y.to_numpy()
        vel_x = particles.vel_x.to_numpy()
        vel_y = particles.vel_y.to_numpy()
        alive = particles.alive.to_numpy()
        energy = particles.energy.to_numpy()

        n_alive = np.sum(alive)
        n_total = len(alive)

        if n_alive == 0:
            self.samples.append(SampleSnapshot(
                spatial_entropy=0.0, island_count=0,
                speed_variance=0.0, angular_momentum=0.0,
                density_laplacian_var=0.0,
                alive_count=0, total_count=n_total,
                nutrient_consumed=0.0, waste_peak=0.0, energy_skew=0.0
            ))
            return

        # Filter to alive particles
        px = pos_x[alive == 1]
        py = pos_y[alive == 1]
        vx = vel_x[alive == 1]
        vy = vel_y[alive == 1]
        energy_alive = energy[alive == 1]

        w = cfg['world']['width']
        h = cfg['world']['height']

        # Spatial entropy (only active particles — those with energy > 0)
        active_mask = energy_alive > 0
        if np.sum(active_mask) < 10:
            # Too few active particles — treat as dead
            self.samples.append(SampleSnapshot(
                spatial_entropy=0.0, island_count=0,
                speed_variance=0.0, angular_momentum=0.0,
                density_laplacian_var=0.0,
                alive_count=int(n_alive), total_count=n_total,
                nutrient_consumed=0.0, waste_peak=0.0, energy_skew=0.0
            ))
            return

        px_active = px[active_mask]
        py_active = py[active_mask]
        vx_active = vx[active_mask]
        vy_active = vy[active_mask]

        # Spatial entropy
        entropy = self._spatial_entropy(px_active, py_active, w, h)

        # Island count
        islands = self._island_count(px_active, py_active, cfg)

        # Speed variance (active particles only)
        speeds = np.sqrt(vx_active**2 + vy_active**2)
        speed_var = float(np.var(speeds))

        # Angular momentum
        cx, cy = np.mean(px_active), np.mean(py_active)
        angular_mom = np.sum((px_active - cx) * vy_active - (py_active - cy) * vx_active)

        # Density Laplacian variance
        density_lap_var = self._density_laplacian_var(px_active, py_active, w, h)

        # v6: Environment features
        nutrient_consumed = 0.0
        waste_peak = 0.0
        energy_skew_val = 0.0

        if environment is not None:
            nutrient_consumed = environment.get_avg_nutrient()
            waste_peak = environment.get_avg_waste()

        # Energy skewness (distribution of energy across alive particles)
        if len(energy_alive) >= 3:
            mean_e = np.mean(energy_alive)
            std_e = np.std(energy_alive)
            if std_e > 1e-10:
                energy_skew_val = float(np.mean(((energy_alive - mean_e) / std_e) ** 3))

        self.samples.append(SampleSnapshot(
            spatial_entropy=entropy,
            island_count=islands,
            speed_variance=speed_var,
            angular_momentum=float(angular_mom),
            density_laplacian_var=density_lap_var,
            alive_count=int(n_alive),
            total_count=n_total,
            nutrient_consumed=nutrient_consumed,
            waste_peak=waste_peak,
            energy_skew=energy_skew_val
        ))

    def compute_features(self) -> np.ndarray:
        """
        Compute 15D feature vector from collected samples (v6).
        Returns zeros if no samples collected.
        """
        if not self.samples:
            return np.zeros(15, dtype=np.float32)

        entropies = np.array([s.spatial_entropy for s in self.samples])
        islands = np.array([s.island_count for s in self.samples], dtype=float)
        speed_vars = np.array([s.speed_variance for s in self.samples])
        angular_moms = np.array([s.angular_momentum for s in self.samples])
        density_lap_vars = np.array([s.density_laplacian_var for s in self.samples])
        alive_counts = np.array([s.alive_count for s in self.samples])
        total_counts = np.array([s.total_count for s in self.samples])
        nutrient_consumed = np.array([s.nutrient_consumed for s in self.samples])
        waste_peaks = np.array([s.waste_peak for s in self.samples])
        energy_skews = np.array([s.energy_skew for s in self.samples])

        features = np.zeros(15, dtype=np.float32)

        # [0] Spatial entropy mean
        features[0] = float(np.mean(entropies))
        # [1] Spatial entropy variance
        features[1] = float(np.var(entropies))
        # [2] Islands mean
        features[2] = float(np.mean(islands))
        # [3] Islands variance
        features[3] = float(np.var(islands))
        # [4] Speed variance mean
        features[4] = float(np.mean(speed_vars))
        # [5-7] FFT of speed variance time series (top 3 amplitudes)
        features[5], features[6], features[7] = self._fft_top3(speed_vars)
        # [8] Angular momentum skewness
        features[8] = self._skewness(angular_moms)
        # [9] Density Laplacian variance mean
        features[9] = float(np.mean(density_lap_vars))
        # [10] Survival rate (final sample)
        final_total = total_counts[-1] if total_counts[-1] > 0 else 1
        features[10] = float(alive_counts[-1]) / float(final_total)
        # [11] Autocorrelation lag-10 of speed variance
        features[11] = self._autocorrelation(speed_vars, lag=10)
        # [12] Nutrient consume mean (v6)
        features[12] = float(np.mean(nutrient_consumed))
        # [13] Waste peak mean (v6)
        features[13] = float(np.mean(waste_peaks))
        # [14] Energy skewness (v6)
        features[14] = float(np.mean(energy_skews))

        return features

    def get_3d_features(self) -> tuple:
        """Extract the 3 features used for MAP-Elites grid indexing."""
        features = self.compute_features()
        return (float(features[0]), float(features[2]), float(features[5]))

    # ── Internal computation methods ──

    def _spatial_entropy(self, px, py, w, h, grid_size=16):
        """Compute spatial entropy of particle distribution."""
        if len(px) == 0:
            return 0.0
        hist, _, _ = np.histogram2d(px, py, bins=grid_size, range=[[0, w], [0, h]])
        probs = hist.flatten() / max(len(px), 1)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(grid_size * grid_size)
        return float(entropy / max(max_entropy, 1e-10))

    def _island_count(self, px, py, cfg):
        """Count connected components via grid-based flood fill."""
        if len(px) == 0:
            return 0
        cell_size = cfg['world']['cell_size'] * 3
        w = cfg['world']['width']
        h = cfg['world']['height']
        grid_w = int(w / cell_size) + 1
        grid_h = int(h / cell_size) + 1

        occupancy = np.zeros((grid_h, grid_w), dtype=bool)
        for i in range(len(px)):
            col = min(int(px[i] / cell_size), grid_w - 1)
            row = min(int(py[i] / cell_size), grid_h - 1)
            occupancy[row, col] = True

        return self._count_islands(occupancy)

    def _count_islands(self, grid):
        """Count connected components in boolean grid."""
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] and not visited[r, c]:
                    count += 1
                    self._flood_fill(grid, visited, r, c)
        return count

    def _flood_fill(self, grid, visited, r, c):
        rows, cols = grid.shape
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                continue
            if visited[cr, cc] or not grid[cr, cc]:
                continue
            visited[cr, cc] = True
            stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])

    def _density_laplacian_var(self, px, py, w, h, grid_size=16):
        """Compute variance of the Laplacian of density field."""
        if len(px) < 2:
            return 0.0
        hist, _, _ = np.histogram2d(px, py, bins=grid_size, range=[[0, w], [0, h]])
        density = hist / max(len(px), 1)

        # Discrete Laplacian
        lap = np.zeros_like(density)
        lap[1:-1, 1:-1] = (
            density[2:, 1:-1] + density[:-2, 1:-1] +
            density[1:-1, 2:] + density[1:-1, :-2] -
            4 * density[1:-1, 1:-1]
        )
        return float(np.var(lap))

    def _fft_top3(self, time_series):
        """Get top 3 FFT amplitudes of a time series."""
        n = len(time_series)
        if n < 4:
            return (0.0, 0.0, 0.0)

        # Remove DC component
        ts = time_series - np.mean(time_series)
        fft = np.fft.rfft(ts)
        amplitudes = np.abs(fft)[1:]  # skip DC

        if len(amplitudes) == 0:
            return (0.0, 0.0, 0.0)

        # Top 3 - normalize by sqrt(n) for proper amplitude scaling
        sorted_amps = np.sort(amplitudes)[::-1]
        top3 = sorted_amps[:3] / np.sqrt(max(n, 1))
        result = [float(a) for a in top3]
        while len(result) < 3:
            result.append(0.0)
        return tuple(result)

    def _skewness(self, data):
        """Compute skewness of a distribution."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))

    def _autocorrelation(self, time_series, lag=10):
        """Compute lag-N autocorrelation of a time series."""
        n = len(time_series)
        if n <= lag:
            return 0.0
        mean = np.mean(time_series)
        var = np.var(time_series)
        if var < 1e-10:
            return 0.0
        # lag-10 autocorrelation
        corr = np.mean((time_series[:n-lag] - mean) * (time_series[lag:] - mean)) / var
        return float(np.clip(corr, -1.0, 1.0))
