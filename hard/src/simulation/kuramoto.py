"""
Kuramoto phase synchronization model.

Each particle has an internal oscillation phase θ and natural frequency ω.
Phase update: dθ_i/dt = ω_i + (K/N_neighbors) Σ_j sin(θ_j - θ_i)

When K is large enough, particles spontaneously synchronize their phases,
creating collective "heartbeat" or "breathing" patterns.

Combined with L-J forces, this produces pulsating soft-matter structures.
"""

import taichi as ti


@ti.data_oriented
class KuramotoSync:
    """Kuramoto phase synchronization on GPU via spatial hash."""

    def __init__(self, cfg: dict):
        k_cfg = cfg.get('kuramoto', {})
        self.K = k_cfg.get('coupling_strength', 2.0)
        self.enabled = k_cfg.get('enabled', False)
        self.n = cfg['simulation']['num_particles']

    @ti.kernel
    def step(self,
             phase: ti.template(),
             dphase: ti.template(),
             alive: ti.template(),
             pos_x: ti.template(),
             pos_y: ti.template(),
             spatial_hash: ti.template(),
             sense_radius: ti.f32):
        """Update phases via Kuramoto coupling using spatial hash neighbors.

        For each alive particle i:
          coupling_i = (1/count) Σ_j sin(θ_j - θ_i)   over neighbors j
          θ_i += ω_i + K * coupling_i
          θ_i = wrap to [0, 2π]
        """
        cell_size = spatial_hash.cell_size
        radius = sense_radius
        r2 = radius * radius

        for i in range(self.n):
            if alive[i] == 0:
                continue

            qx = pos_x[i]
            qy = pos_y[i]

            # Find neighbor cells
            min_col = ti.math.clamp(int((qx - radius) / cell_size), 0, spatial_hash.cols - 1)
            max_col = ti.math.clamp(int((qx + radius) / cell_size), 0, spatial_hash.cols - 1)
            min_row = ti.math.clamp(int((qy - radius) / cell_size), 0, spatial_hash.rows - 1)
            max_row = ti.math.clamp(int((qy + radius) / cell_size), 0, spatial_hash.rows - 1)

            coupling = 0.0
            count = 0

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell = row * spatial_hash.cols + col
                    idx = spatial_hash.cell_head[cell]
                    iter_count = 0
                    while idx != -1 and iter_count < 128:
                        iter_count += 1
                        if idx != i and alive[idx] == 1:
                            dx = qx - pos_x[idx]
                            dy = qy - pos_y[idx]
                            d2 = dx * dx + dy * dy
                            if d2 < r2 and d2 > 0.0:
                                coupling += ti.sin(phase[idx] - phase[i])
                                count += 1
                        idx = spatial_hash.particle_next[idx]

            # Phase update
            if count > 0:
                phase[i] += dphase[i] + self.K * coupling / float(count)
            else:
                phase[i] += dphase[i]

            # Wrap to [0, 2π]
            phase[i] = phase[i] - ti.floor(phase[i] / 6.28318) * 6.28318
