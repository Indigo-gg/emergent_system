"""
Lennard-Jones potential: continuous attraction-repulsion force.

U(r) = 4ε [(σ/r)^12 - (σ/r)^6]
F(r) = -dU/dr = 24ε [2(σ/r)^12 - (σ/r)^6] / r

At r = σ:   F = 0  (equilibrium)
At r < σ:   F > 0  (repulsion)
At r > σ:   F < 0  (attraction, up to cutoff)
At r > cutoff: F = 0

This replaces the hard collision threshold with a continuous force field,
enabling soft "membrane" structures to emerge from particle interactions.
"""

import taichi as ti


@ti.data_oriented
class LennardJonesPotential:
    """Lennard-Jones force computation on GPU."""

    def __init__(self, cfg: dict):
        lj_cfg = cfg.get('lj', {})
        self.sigma = lj_cfg.get('sigma', 0.5)
        self.epsilon = lj_cfg.get('epsilon', 1.0)
        self.cutoff = lj_cfg.get('cutoff', 2.5)
        self.enabled = lj_cfg.get('enabled', True)

        # Precompute sigma^6 for GPU kernel (avoid repeated pow)
        self.sigma6 = self.sigma ** 6
        self.sigma12 = self.sigma ** 12
        self.cutoff2 = self.cutoff ** 2

    @ti.func
    def compute_force(self, dist: ti.f32) -> ti.f32:
        """Compute L-J force magnitude at distance dist.

        Returns positive for repulsion, negative for attraction.
        Returns 0 if dist > cutoff or dist ≈ 0.
        """
        f_mag = 0.0
        if dist > 1e-6 and dist < self.cutoff:
            inv_r = 1.0 / dist
            sr2 = self.sigma * self.sigma * inv_r * inv_r
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            # F = 24ε [2(σ/r)^12 - (σ/r)^6] / r
            f_mag = 24.0 * self.epsilon * (2.0 * sr12 - sr6) * inv_r
        return f_mag

    @ti.func
    def compute_energy(self, dist: ti.f32) -> ti.f32:
        """Compute L-J potential energy at distance dist."""
        energy = 0.0
        if dist > 1e-6 and dist < self.cutoff:
            inv_r = 1.0 / dist
            sr2 = self.sigma * self.sigma * inv_r * inv_r
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6
            energy = 4.0 * self.epsilon * (sr12 - sr6)
        return energy
