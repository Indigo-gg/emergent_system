"""
Single-step simulation: assembles spatial hash + VM force computation + integration.

This is the main simulation kernel that ties everything together.
For Phase 1, uses a hardcoded potential formula for testing.
"""

import taichi as ti
import numpy as np

from src.simulation.vm import vm_execute
from src.simulation.potential import (
    Const, Var, Add, Mul, Neg, Sin, compile_potential
)


@ti.data_oriented
class SimulationStep:
    """Orchestrates one simulation step."""

    def __init__(self, spatial_hash, integrator, cfg: dict):
        self.spatial_hash = spatial_hash
        self.integrator = integrator
        self.n = cfg['simulation']['num_particles']
        self.n_state = cfg['simulation']['particle_state_dim']
        self.vm_stack_depth = cfg['gep']['vm_stack_depth']
        self.bytecode_len = cfg['gep']['bytecode_length']
        self.sense_radius = cfg['world']['cell_size'] * 1.5

        # Default potential: U = -dist (simple attraction)
        # This will be replaced by GEP-evolved formulas in Phase 2
        self.default_potential = Neg(Var('dist', 0))

        # Compile to bytecode (dU/dr)
        self._compile_default()

    def _compile_default(self):
        """Compile default potential to bytecode."""
        dudr_bc, constants = compile_potential(
            self.default_potential, self.bytecode_len
        )
        self.bytecode_np = np.array(dudr_bc, dtype=np.int32)
        self.constants_np = np.array(constants, dtype=np.float32)
        # Pad constants to at least 1 element
        if len(self.constants_np) == 0:
            self.constants_np = np.array([0.0], dtype=np.float32)

    def set_potential(self, potential_tree):
        """Set a new potential energy formula (from GEP evolution)."""
        dudr_bc, constants = compile_potential(
            potential_tree, self.bytecode_len
        )
        self.bytecode_np = np.array(dudr_bc, dtype=np.int32)
        self.constants_np = np.array(
            constants if constants else [0.0],
            dtype=np.float32
        )

    @ti.kernel
    def compute_forces(self,
                       pos_x: ti.types.ndarray(), pos_y: ti.types.ndarray(),
                       vel_x: ti.types.ndarray(), vel_y: ti.types.ndarray(),
                       state: ti.types.ndarray(),
                       force_x: ti.types.ndarray(), force_y: ti.types.ndarray(),
                       alive: ti.types.ndarray(),
                       bytecode: ti.types.ndarray(),
                       constants: ti.types.ndarray()):
        """Compute forces on all particles using VM-executed bytecode."""
        for i in range(pos_x.shape[0]):
            if alive[i] == 0:
                continue

            qx = pos_x[i]
            qy = pos_y[i]

            # Query neighbors from spatial hash
            # We need a per-thread neighbor buffer
            # Since Taichi doesn't support per-thread dynamic allocation easily,
            # we iterate through the spatial hash directly

            fx_sum = 0.0
            fy_sum = 0.0

            cell_size = self.spatial_hash.cell_size
            radius = self.sense_radius
            r2 = radius * radius

            min_col = ti.math.clamp(int((qx - radius) / cell_size), 0, self.spatial_hash.cols - 1)
            max_col = ti.math.clamp(int((qx + radius) / cell_size), 0, self.spatial_hash.cols - 1)
            min_row = ti.math.clamp(int((qy - radius) / cell_size), 0, self.spatial_hash.rows - 1)
            max_row = ti.math.clamp(int((qy + radius) / cell_size), 0, self.spatial_hash.rows - 1)

            neighbor_count = 0

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell = row * self.spatial_hash.cols + col
                    idx = self.spatial_hash.cell_head[cell]
                    iter_count = 0
                    while idx != -1 and iter_count < 128:
                        iter_count += 1
                        dx = qx - pos_x[idx]
                        dy = qy - pos_y[idx]
                        d2 = dx * dx + dy * dy

                        if d2 < r2 and d2 > 0.0 and idx != i:
                            dist = ti.sqrt(d2)
                            angle = ti.atan2(dy, dx)

                            # Build variables for VM
                            # vars = [dist, density, speed, angle, s0, s1, s2, s3, n_count]
                            speed_i = ti.sqrt(vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i])

                            # Execute VM to get dU/dr
                            # We pass vars as individual values since we can't create
                            # a dynamic array inside a kernel easily
                            dudr = self._vm_force_single(
                                dist, 0.0, speed_i, angle,
                                state[i, 0], state[i, 1], state[i, 2], state[i, 3],
                                float(neighbor_count),
                                bytecode, constants
                            )

                            # Force direction: pointing from i toward j (attractive if negative)
                            # F = -dU/dr, direction = (dx, dy) / dist normalized toward j
                            # Actually: force on i from j = -dU/dr * (j - i) / |j - i|
                            # = -dU/dr * (-dx, -dy) / dist
                            fx_ij = dudr * dx / dist  # dx points from i to j is actually (qx - pos_x[idx])
                            fy_ij = dudr * dy / dist

                            fx_sum += fx_ij
                            fy_sum += fy_ij
                            neighbor_count += 1

                        idx = self.spatial_hash.particle_next[idx]

            force_x[i] = fx_sum
            force_y[i] = fy_sum

    @ti.func
    def _vm_force_single(self, dist, density, speed, angle,
                         s0, s1, s2, s3, n_count,
                         bytecode, constants) -> ti.f32:
        """Execute VM for a single particle pair to get dU/dr value."""
        vars = ti.Vector([dist, density, speed, angle, s0, s1, s2, s3, n_count])
        return vm_execute(bytecode, constants, vars, self.vm_stack_depth)

    def step(self, particles):
        """Execute one full simulation step."""
        # 1. Build spatial hash
        self.spatial_hash.build(particles.pos_x, particles.pos_y)

        # 2. Compute forces via VM
        self.compute_forces(
            particles.pos_x, particles.pos_y,
            particles.vel_x, particles.vel_y,
            particles.state,
            particles.force_x, particles.force_y,
            particles.alive,
            self.bytecode_np, self.constants_np
        )

        # 3. Integrate
        self.integrator.step(
            particles.pos_x, particles.pos_y,
            particles.vel_x, particles.vel_y,
            particles.force_x, particles.force_y,
            particles.alive
        )
