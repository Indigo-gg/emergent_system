"""
Environment layer: nutrient field + waste field with diffusion, decay, and injection.

v6 key addition: particles compete for resources and avoid toxins.
This creates natural selection pressure — the "amplifier of emergence".

Fields are stored on the same grid as spatial hash (same cell_size, cols, rows).
"""

import taichi as ti
import numpy as np
import math


@ti.data_oriented
class EnvironmentLayer:
    """Manages nutrient and waste scalar fields on GPU."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        env_cfg = cfg['environment']
        world_cfg = cfg['world']

        # Grid dimensions (same as spatial hash)
        self.cell_size = world_cfg['cell_size']
        self.world_w = world_cfg['width']
        self.world_h = world_cfg['height']
        self.cols = int(self.world_w / self.cell_size) + 1
        self.rows = int(self.world_h / self.cell_size) + 1
        self.num_cells = self.cols * self.rows

        # Diffusion/decay parameters
        self.nutrient_diffuse = env_cfg['nutrient_diffuse_rate']
        self.nutrient_decay = env_cfg['nutrient_decay_rate']
        self.nutrient_inject_interval = env_cfg['nutrient_inject_interval']
        self.nutrient_patch_count = env_cfg['nutrient_patch_count']
        self.nutrient_patch_amount = env_cfg['nutrient_patch_amount']
        self.nutrient_drift_speed = env_cfg['nutrient_drift_speed']

        self.waste_diffuse = env_cfg['waste_diffuse_rate']
        self.waste_decay = env_cfg['waste_decay_rate']
        self.waste_production_rate = env_cfg['waste_production_rate']

        self.base_metabolism = env_cfg['base_metabolism']
        self.move_cost = env_cfg['move_cost']
        self.absorb_rate = env_cfg['absorb_rate']
        self.dormant_metabolism = env_cfg['dormant_metabolism']
        self.max_dormant_ticks = env_cfg['max_dormant_ticks']
        self.waste_metabolism_factor = env_cfg['waste_metabolism_factor']

        # Scalar fields
        self.nutrient_field = ti.field(dtype=ti.f32, shape=(self.rows, self.cols))
        self.waste_field = ti.field(dtype=ti.f32, shape=(self.rows, self.cols))

        # Gradient fields (precomputed each step)
        self.grad_nut_x = ti.field(dtype=ti.f32, shape=(self.rows, self.cols))
        self.grad_nut_y = ti.field(dtype=ti.f32, shape=(self.rows, self.cols))
        self.grad_waste_x = ti.field(dtype=ti.f32, shape=(self.rows, self.cols))
        self.grad_waste_y = ti.field(dtype=ti.f32, shape=(self.rows, self.cols))

        # Nutrient hotspot patches (drifting injection points)
        # Stored as flat arrays for GPU access
        self.max_patches = 16
        self.patch_cx = ti.field(dtype=ti.f32, shape=self.max_patches)
        self.patch_cy = ti.field(dtype=ti.f32, shape=self.max_patches)
        self.patch_angle = ti.field(dtype=ti.f32, shape=self.max_patches)
        self.n_patches = self.nutrient_patch_count

        # Step counter for injection timing
        self.step_count = 0

    def initialize(self, seed: int = 42):
        """Initialize fields to zero and place nutrient hotspots."""
        # Zero all fields using a kernel
        self._zero_fields()

        # Place initial nutrient hotspots along a circle (CPU-side)
        center_x = self.world_w / 2.0
        center_y = self.world_h / 2.0
        radius = min(self.world_w, self.world_h) * 0.3

        patch_cx_np = np.zeros(self.max_patches, dtype=np.float32)
        patch_cy_np = np.zeros(self.max_patches, dtype=np.float32)
        patch_angle_np = np.zeros(self.max_patches, dtype=np.float32)

        for p in range(self.max_patches):
            if p < self.n_patches:
                angle = 2.0 * math.pi * p / self.n_patches
                patch_cx_np[p] = center_x + radius * math.cos(angle)
                patch_cy_np[p] = center_y + radius * math.sin(angle)
                patch_angle_np[p] = angle
            else:
                patch_cx_np[p] = -1.0
                patch_cy_np[p] = -1.0
                patch_angle_np[p] = 0.0

        self.patch_cx.from_numpy(patch_cx_np)
        self.patch_cy.from_numpy(patch_cy_np)
        self.patch_angle.from_numpy(patch_angle_np)
        self.step_count = 0

    @ti.kernel
    def _zero_fields(self):
        """Zero all environment fields."""
        for i, j in self.nutrient_field:
            self.nutrient_field[i, j] = 0.0
            self.waste_field[i, j] = 0.0

    @ti.kernel
    def environment_step(self):
        """Run one step of environment dynamics: diffuse, decay, inject."""
        cs = self.cell_size
        ww = self.world_w
        wh = self.world_h

        # 1. Nutrient diffusion + decay (5-point Laplacian on toroidal grid)
        for i, j in self.nutrient_field:
            # Neighbor indices with periodic boundary
            ip = (i + 1) % self.rows
            im = (i - 1 + self.rows) % self.rows
            jp = (j + 1) % self.cols
            jm = (j - 1 + self.cols) % self.cols

            laplacian = (self.nutrient_field[ip, j] + self.nutrient_field[im, j] +
                         self.nutrient_field[i, jp] + self.nutrient_field[i, jm] -
                         4.0 * self.nutrient_field[i, j])
            val = self.nutrient_field[i, j] + self.nutrient_diffuse * laplacian - self.nutrient_decay * self.nutrient_field[i, j]
            self.nutrient_field[i, j] = ti.max(val, 0.0)

        # 2. Waste diffusion + decay
        for i, j in self.waste_field:
            ip = (i + 1) % self.rows
            im = (i - 1 + self.rows) % self.rows
            jp = (j + 1) % self.cols
            jm = (j - 1 + self.cols) % self.cols

            laplacian = (self.waste_field[ip, j] + self.waste_field[im, j] +
                         self.waste_field[i, jp] + self.waste_field[i, jm] -
                         4.0 * self.waste_field[i, j])
            val = self.waste_field[i, j] + self.waste_diffuse * laplacian - self.waste_decay * self.waste_field[i, j]
            self.waste_field[i, j] = ti.max(val, 0.0)

        # 3. Nutrient injection at drifting hotspots
        for p in range(self.n_patches):
            if self.patch_cx[p] < 0.0:
                continue

            cx = self.patch_cx[p]
            cy = self.patch_cy[p]
            amount = self.nutrient_patch_amount
            inject_radius = cs * 3.0  # inject into a 3-cell radius

            # Gaussian injection around hotspot center
            min_r = ti.max(int((cx - inject_radius) / cs), 0)
            max_r = ti.min(int((cx + inject_radius) / cs), self.cols - 1)
            min_c = ti.max(int((cy - inject_radius) / cs), 0)
            max_c = ti.min(int((cy + inject_radius) / cs), self.rows - 1)

            for row in range(min_c, max_c + 1):
                for col in range(min_r, max_r + 1):
                    dx = col * cs - cx
                    dy = row * cs - cy
                    dist_sq = dx * dx + dy * dy
                    sigma_sq = inject_radius * inject_radius
                    weight = ti.exp(-dist_sq / (2.0 * sigma_sq))
                    self.nutrient_field[row, col] += amount * weight * 0.01

            # Drift the hotspot (spiral path)
            self.patch_angle[p] += self.nutrient_drift_speed
            drift_r = 5.0 + 2.0 * ti.sin(self.patch_angle[p] * 0.3)
            self.patch_cx[p] = cx + drift_r * ti.cos(self.patch_angle[p]) * cs
            self.patch_cy[p] = cy + drift_r * ti.sin(self.patch_angle[p]) * cs

            # Wrap to world boundaries
            self.patch_cx[p] = self.patch_cx[p] - ti.floor(self.patch_cx[p] / ww) * ww
            self.patch_cy[p] = self.patch_cy[p] - ti.floor(self.patch_cy[p] / wh) * wh

    @ti.kernel
    def compute_gradients(self):
        """Precompute spatial gradients of nutrient and waste fields (central difference)."""
        cs = self.cell_size
        inv_2cs = 1.0 / (2.0 * cs)

        for i, j in self.nutrient_field:
            ip = (i + 1) % self.rows
            im = (i - 1 + self.rows) % self.rows
            jp = (j + 1) % self.cols
            jm = (j - 1 + self.cols) % self.cols

            # ∇nutrient: x = dN/d(col), y = dN/d(row)
            self.grad_nut_x[i, j] = (self.nutrient_field[i, jp] - self.nutrient_field[i, jm]) * inv_2cs
            self.grad_nut_y[i, j] = (self.nutrient_field[ip, j] - self.nutrient_field[im, j]) * inv_2cs

            # ∇waste
            self.grad_waste_x[i, j] = (self.waste_field[i, jp] - self.waste_field[i, jm]) * inv_2cs
            self.grad_waste_y[i, j] = (self.waste_field[ip, j] - self.waste_field[im, j]) * inv_2cs

    @ti.kernel
    def particle_environment_interaction(self,
                                         pos_x: ti.template(),
                                         pos_y: ti.template(),
                                         vel_x: ti.template(),
                                         vel_y: ti.template(),
                                         energy: ti.template(),
                                         dormant_ticks: ti.template(),
                                         alive: ti.template(),
                                         total_nutrient_absorbed: ti.f32) -> ti.f32:
        """
        Environment-particle interaction:
        1. Absorb nutrient from current position → energy gain
        2. Produce waste proportional to nutrient absorbed
        3. Metabolism: energy cost = base + move_cost * speed
        4. Waste penalty: high waste → increased metabolism
        5. Dormancy: energy <= 0 → dormant (low metabolism, no movement)
        6. Wake up: dormant particle near nutrient → resume
        7. Death: dormant too long → permanent death

        Returns total nutrient absorbed (for fitness tracking).
        """
        cs = self.cell_size
        n = pos_x.shape[0]
        absorbed_total = 0.0

        for i in range(n):
            if alive[i] == 0:
                continue

            # Get grid cell for this particle's position
            col = ti.math.clamp(int(pos_x[i] / cs), 0, self.cols - 1)
            row = ti.math.clamp(int(pos_y[i] / cs), 0, self.rows - 1)

            nut_val = self.nutrient_field[row, col]
            waste_val = self.waste_field[row, col]
            spd = ti.sqrt(vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i])

            # Dormant particle logic
            if dormant_ticks[i] > 0:
                # Dormant: very low metabolism, check if can wake up
                energy[i] -= self.dormant_metabolism

                # Wake up if nutrient is nearby
                if nut_val > 0.1:
                    dormant_ticks[i] = 0
                    # Small energy boost from waking near nutrient
                    energy[i] += nut_val * self.absorb_rate * 0.5
                    absorbed_total += nut_val * self.absorb_rate * 0.5

                # Death if dormant too long
                if dormant_ticks[i] >= self.max_dormant_ticks:
                    alive[i] = 0
                    energy[i] = 0.0

                dormant_ticks[i] += 1
                continue

            # Active particle: absorb nutrient
            nutrient_gain = nut_val * self.absorb_rate
            energy[i] += nutrient_gain
            absorbed_total += nutrient_gain

            # Consume nutrient from field (partial)
            self.nutrient_field[row, col] = ti.max(nut_val - nutrient_gain * 0.5, 0.0)

            # Produce waste proportional to nutrient consumed
            waste_produced = nutrient_gain * self.waste_production_rate
            self.waste_field[row, col] += waste_produced

            # Metabolism cost
            metabolize = self.base_metabolism + self.move_cost * spd
            # Waste penalty: high waste → increased metabolism
            metabolize *= (1.0 + waste_val * self.waste_metabolism_factor)
            energy[i] -= metabolize

            # Dormancy check: energy <= 0 → enter dormant state
            if energy[i] <= 0.0:
                energy[i] = 0.01  # tiny residual energy
                dormant_ticks[i] = 1  # start dormant counter
                vel_x[i] = 0.0
                vel_y[i] = 0.0
                continue

            # Clamp energy to reasonable range
            energy[i] = ti.min(energy[i], 10.0)

        return absorbed_total

    @ti.func
    def sample_field(self, field: ti.template(), x: ti.f32, y: ti.f32) -> ti.f32:
        """Bilinear interpolation sampling of a scalar field at continuous position (x, y)."""
        cs = self.cell_size
        # Convert to grid coordinates
        gx = x / cs
        gy = y / cs
        j0 = ti.math.clamp(int(gx), 0, self.cols - 1)
        i0 = ti.math.clamp(int(gy), 0, self.rows - 1)
        j1 = ti.math.clamp(j0 + 1, 0, self.cols - 1)
        i1 = ti.math.clamp(i0 + 1, 0, self.rows - 1)

        # Fractional parts
        fx = gx - float(j0)
        fy = gy - float(i0)

        # Bilinear interpolation
        v00 = field[i0, j0]
        v01 = field[i0, j1]
        v10 = field[i1, j0]
        v11 = field[i1, j1]

        return (v00 * (1.0 - fx) * (1.0 - fy) +
                v01 * fx * (1.0 - fy) +
                v10 * (1.0 - fx) * fy +
                v11 * fx * fy)

    @ti.func
    def sample_gradient(self, grad_x: ti.template(), grad_y: ti.template(),
                        x: ti.f32, y: ti.f32) -> tuple:
        """Bilinear interpolation sampling of gradient field at continuous position."""
        cs = self.cell_size
        gx = x / cs
        gy = y / cs
        j0 = ti.math.clamp(int(gx), 0, self.cols - 1)
        i0 = ti.math.clamp(int(gy), 0, self.rows - 1)
        j1 = ti.math.clamp(j0 + 1, 0, self.cols - 1)
        i1 = ti.math.clamp(i0 + 1, 0, self.rows - 1)

        fx = gx - float(j0)
        fy = gy - float(i0)

        vx = (grad_x[i0, j0] * (1.0 - fx) * (1.0 - fy) +
              grad_x[i0, j1] * fx * (1.0 - fy) +
              grad_x[i1, j0] * (1.0 - fx) * fy +
              grad_x[i1, j1] * fx * fy)

        vy = (grad_y[i0, j0] * (1.0 - fx) * (1.0 - fy) +
              grad_y[i0, j1] * fx * (1.0 - fy) +
              grad_y[i1, j0] * (1.0 - fx) * fy +
              grad_y[i1, j1] * fx * fy)

        return vx, vy

    def get_avg_nutrient(self) -> float:
        """Get average nutrient field concentration (for monitoring)."""
        arr = self.nutrient_field.to_numpy()
        return float(np.mean(arr))

    def get_avg_waste(self) -> float:
        """Get average waste field concentration (for monitoring)."""
        arr = self.waste_field.to_numpy()
        return float(np.mean(arr))

    def to_numpy(self) -> dict:
        """Export fields to numpy for checkpointing."""
        return {
            'nutrient': self.nutrient_field.to_numpy(),
            'waste': self.waste_field.to_numpy(),
            'patch_cx': self.patch_cx.to_numpy(),
            'patch_cy': self.patch_cy.to_numpy(),
            'patch_angle': self.patch_angle.to_numpy(),
            'step_count': self.step_count,
        }

    def from_numpy(self, data: dict):
        """Restore fields from numpy (checkpoint restore)."""
        self.nutrient_field.from_numpy(data['nutrient'])
        self.waste_field.from_numpy(data['waste'])
        self.patch_cx.from_numpy(data['patch_cx'])
        self.patch_cy.from_numpy(data['patch_cy'])
        self.patch_angle.from_numpy(data['patch_angle'])
        self.step_count = data.get('step_count', 0)
