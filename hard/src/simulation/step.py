"""
Single-step simulation: assembles spatial hash + dual-force computation + integration.

v6 dual-force architecture:
  F_total = F_particle + F_env

  F_particle: particle-particle interaction via GEP potential U
    - Terminal set includes symmetric avg_nutrient/avg_waste at midpoint
    - Force from symbolic diff: F = -dU/dr

  F_env: environment gradient chemotaxis (independent of neighbors)
    - GEP-evolved chemotaxis formula
    - Reads ∇nutrient and ∇waste at particle position

Hard collision: post-integration safety net for overflow particles.
"""

import taichi as ti
import numpy as np

from src.simulation.vm import vm_execute, vm_execute_chemotaxis
from src.simulation.potential import (
    Const, Var, Add, Mul, Neg, Sin, compile_potential, compile_chemotaxis
)

HARD_REPULSION_EPSILON = 0.01
HARD_REPULSION_STRENGTH = 100.0


@ti.data_oriented
class SimulationStep:
    """Orchestrates one simulation step with dual-force architecture."""

    def __init__(self, spatial_hash, integrator, cfg: dict):
        self.spatial_hash = spatial_hash
        self.integrator = integrator
        self.n = cfg['simulation']['num_particles']
        self.n_state = cfg['simulation']['particle_state_dim']
        self.vm_stack_depth = cfg['gep']['vm_stack_depth']
        self.bytecode_len = cfg['gep']['bytecode_length']
        self.sense_radius = cfg['world']['cell_size'] * 1.5

        # Hard collision parameters
        self.hard_eps = cfg['safety'].get('hard_repulsion_epsilon', 0.01)
        self.hard_strength = cfg['safety'].get('hard_repulsion_strength', 100.0)

        # Default potential: U = -dist (simple attraction)
        self.default_potential = Neg(Var('dist', 0))

        # Compile to bytecode (dU/dr)
        self._compile_default()
        self._compile_default_chemotaxis()

    def _compile_default(self):
        """Compile default potential to bytecode."""
        dudr_bc, constants = compile_potential(
            self.default_potential, self.bytecode_len
        )
        self.bytecode_np = np.array(dudr_bc, dtype=np.int32)
        self.constants_np = np.array(constants, dtype=np.float32)
        if len(self.constants_np) == 0:
            self.constants_np = np.array([0.0], dtype=np.float32)

    def _compile_default_chemotaxis(self):
        """Compile default chemotaxis formula: F_env = 0 (no env force)."""
        # Default: zero chemotaxis (purely constant 0)
        default_chemo = Const(0.0)
        chemo_bc, chemo_consts = compile_chemotaxis(
            default_chemo, self.bytecode_len
        )
        self.chemo_bytecode_np = np.array(chemo_bc, dtype=np.int32)
        self.chemo_constants_np = np.array(
            chemo_consts if chemo_consts else [0.0],
            dtype=np.float32
        )

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

    def set_chemotaxis(self, chemotaxis_tree):
        """Set a new chemotaxis formula (from GEP evolution)."""
        chemo_bc, chemo_consts = compile_chemotaxis(
            chemotaxis_tree, self.bytecode_len
        )
        self.chemo_bytecode_np = np.array(chemo_bc, dtype=np.int32)
        self.chemo_constants_np = np.array(
            chemo_consts if chemo_consts else [0.0],
            dtype=np.float32
        )

    @ti.kernel
    def compute_forces(self,
                       pos_x: ti.template(), pos_y: ti.template(),
                       vel_x: ti.template(), vel_y: ti.template(),
                       state: ti.template(),
                       force_x: ti.template(), force_y: ti.template(),
                       alive: ti.template(),
                       bytecode: ti.types.ndarray(),
                       constants: ti.types.ndarray(),
                       chemo_bytecode: ti.types.ndarray(),
                       chemo_constants: ti.types.ndarray(),
                       nutrient_field: ti.template(),
                       waste_field: ti.template(),
                       grad_nut_x: ti.template(),
                       grad_nut_y: ti.template(),
                       grad_waste_x: ti.template(),
                       grad_waste_y: ti.template()):
        """
        Compute dual-force on all particles:

        F_particle: particle-particle interaction via GEP potential
          - Uses symmetric avg_nutrient/avg_waste at midpoint
          - Force = -dU/dr

        F_env: environment gradient chemotaxis
          - Reads ∇nutrient, ∇waste at particle position
          - GEP-evolved chemotaxis formula
        """
        for i in range(self.n):
            if alive[i] == 0:
                continue

            qx = pos_x[i]
            qy = pos_y[i]
            speed_i = ti.sqrt(vel_x[i] * vel_x[i] + vel_y[i] * vel_y[i])

            # ── F_particle: neighbor-based interaction force ──
            fx_particle = 0.0
            fy_particle = 0.0

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

                            # Compute avg_nutrient and avg_waste at midpoint (symmetric)
                            mid_x = (qx + pos_x[idx]) * 0.5
                            mid_y = (qy + pos_y[idx]) * 0.5
                            mid_col = ti.math.clamp(int(mid_x / cell_size), 0, self.spatial_hash.cols - 1)
                            mid_row = ti.math.clamp(int(mid_y / cell_size), 0, self.spatial_hash.rows - 1)
                            avg_nut = nutrient_field[mid_row, mid_col]
                            avg_wst = waste_field[mid_row, mid_col]

                            # Execute potential VM with symmetric environment variables
                            dudr = vm_execute(
                                bytecode, constants,
                                dist, 0.0, speed_i, angle,
                                state[i, 0], state[i, 1], state[i, 2], state[i, 3],
                                float(neighbor_count),
                                avg_nut, avg_wst,
                                self.vm_stack_depth
                            )

                            # Force: F = -dU/dr, direction toward neighbor
                            fx_ij = dudr * dx / dist
                            fy_ij = dudr * dy / dist

                            fx_particle += fx_ij
                            fy_particle += fy_ij
                            neighbor_count += 1

                        idx = self.spatial_hash.particle_next[idx]

            # ── F_env: environment gradient chemotaxis force ──
            fx_env = 0.0
            fy_env = 0.0

            # Sample gradients at particle position
            cs = cell_size
            gx = qx / cs
            gy = qy / cs
            j0 = ti.math.clamp(int(gx), 0, self.spatial_hash.cols - 1)
            i0 = ti.math.clamp(int(gy), 0, self.spatial_hash.rows - 1)
            j1 = ti.math.clamp(j0 + 1, 0, self.spatial_hash.cols - 1)
            i1 = ti.math.clamp(i0 + 1, 0, self.spatial_hash.rows - 1)
            fx_frac = gx - float(j0)
            fy_frac = gy - float(i0)

            # Bilinear interpolation of gradients
            gnx = (grad_nut_x[i0, j0] * (1.0 - fx_frac) * (1.0 - fy_frac) +
                   grad_nut_x[i0, j1] * fx_frac * (1.0 - fy_frac) +
                   grad_nut_x[i1, j0] * (1.0 - fx_frac) * fy_frac +
                   grad_nut_x[i1, j1] * fx_frac * fy_frac)
            gny = (grad_nut_y[i0, j0] * (1.0 - fx_frac) * (1.0 - fy_frac) +
                   grad_nut_y[i0, j1] * fx_frac * (1.0 - fy_frac) +
                   grad_nut_y[i1, j0] * (1.0 - fx_frac) * fy_frac +
                   grad_nut_y[i1, j1] * fx_frac * fy_frac)
            gwx = (grad_waste_x[i0, j0] * (1.0 - fx_frac) * (1.0 - fy_frac) +
                   grad_waste_x[i0, j1] * fx_frac * (1.0 - fy_frac) +
                   grad_waste_x[i1, j0] * (1.0 - fx_frac) * fy_frac +
                   grad_waste_x[i1, j1] * fx_frac * fy_frac)
            gwy = (grad_waste_y[i0, j0] * (1.0 - fx_frac) * (1.0 - fy_frac) +
                   grad_waste_y[i0, j1] * fx_frac * (1.0 - fy_frac) +
                   grad_waste_y[i1, j0] * (1.0 - fx_frac) * fy_frac +
                   grad_waste_y[i1, j1] * fx_frac * fy_frac)

            # Local nutrient/waste values
            nut_local = (nutrient_field[i0, j0] * (1.0 - fx_frac) * (1.0 - fy_frac) +
                         nutrient_field[i0, j1] * fx_frac * (1.0 - fy_frac) +
                         nutrient_field[i1, j0] * (1.0 - fx_frac) * fy_frac +
                         nutrient_field[i1, j1] * fx_frac * fy_frac)
            wst_local = (waste_field[i0, j0] * (1.0 - fx_frac) * (1.0 - fy_frac) +
                         waste_field[i0, j1] * fx_frac * (1.0 - fy_frac) +
                         waste_field[i1, j0] * (1.0 - fx_frac) * fy_frac +
                         waste_field[i1, j1] * fx_frac * fy_frac)

            # Execute chemotaxis VM: formula outputs scalar magnitude
            # Force direction comes from nutrient gradient
            chemo_magnitude = vm_execute_chemotaxis(
                chemo_bytecode, chemo_constants,
                gnx, gny, gwx, gwy,
                nut_local, wst_local,
                speed_i,
                self.vm_stack_depth
            )

            # Direction from nutrient gradient (normalize)
            grad_len = ti.sqrt(gnx * gnx + gny * gny)
            if grad_len > 1e-6:
                fx_env = chemo_magnitude * gnx / grad_len
                fy_env = chemo_magnitude * gny / grad_len
            else:
                fx_env = 0.0
                fy_env = 0.0

            # ── Total force: F_particle + F_env ──
            force_x[i] = fx_particle + fx_env
            force_y[i] = fy_particle + fy_env

    @ti.kernel
    def apply_hard_collision(self,
                             pos_x: ti.template(),
                             pos_y: ti.template(),
                             force_x: ti.template(),
                             force_y: ti.template(),
                             alive: ti.template()):
        """
        Post-integration hard collision protection.

        For any pair of alive particles closer than HARD_REPULSION_EPSILON,
        apply a strong repulsion force. This is a safety net for particles
        that overflow the spatial hash bucket limit and become "ghosts".
        """
        cell_size = self.spatial_hash.cell_size
        radius = self.hard_eps * 3.0  # search slightly beyond epsilon
        r2 = radius * radius
        eps = self.hard_eps
        strength = self.hard_strength

        for i in range(self.n):
            if alive[i] == 0:
                continue

            qx = pos_x[i]
            qy = pos_y[i]

            min_col = ti.math.clamp(int((qx - radius) / cell_size), 0, self.spatial_hash.cols - 1)
            max_col = ti.math.clamp(int((qx + radius) / cell_size), 0, self.spatial_hash.cols - 1)
            min_row = ti.math.clamp(int((qy - radius) / cell_size), 0, self.spatial_hash.rows - 1)
            max_row = ti.math.clamp(int((qy + radius) / cell_size), 0, self.spatial_hash.rows - 1)

            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    cell = row * self.spatial_hash.cols + col
                    idx = self.spatial_hash.cell_head[cell]
                    while idx != -1:
                        if idx != i and alive[idx] == 1:
                            dx = qx - pos_x[idx]
                            dy = qy - pos_y[idx]
                            d2 = dx * dx + dy * dy
                            if d2 < eps * eps and d2 > 0.0:
                                dist = ti.sqrt(d2)
                                overlap = eps - dist
                                force_mag = strength * overlap
                                fx = force_mag * dx / dist
                                fy = force_mag * dy / dist
                                force_x[i] += fx
                                force_y[i] += fy
                        idx = self.spatial_hash.particle_next[idx]

    def step(self, particles, environment):
        """Execute one full simulation step.

        Args:
            particles: ParticleSystem instance
            environment: EnvironmentLayer instance (required for v6 dual-force)
        """
        # 1. Build spatial hash
        pos_x_np = particles.pos_x.to_numpy()
        pos_y_np = particles.pos_y.to_numpy()
        self.spatial_hash.build(pos_x_np, pos_y_np)

        # 2. Environment step (diffuse + decay + inject + gradients)
        environment.environment_step()
        environment.compute_gradients()

        # 3. Compute dual-force (F_particle + F_env)
        self.compute_forces(
            particles.pos_x, particles.pos_y,
            particles.vel_x, particles.vel_y,
            particles.state,
            particles.force_x, particles.force_y,
            particles.alive,
            self.bytecode_np, self.constants_np,
            self.chemo_bytecode_np, self.chemo_constants_np,
            environment.nutrient_field, environment.waste_field,
            environment.grad_nut_x, environment.grad_nut_y,
            environment.grad_waste_x, environment.grad_waste_y
        )

        # 4. Integrate — pass Taichi fields directly (runs on GPU, no transfer)
        self.integrator.step(
            particles.pos_x, particles.pos_y,
            particles.vel_x, particles.vel_y,
            particles.force_x, particles.force_y,
            particles.alive
        )

        # 5. Hard collision protection (post-integration)
        self.apply_hard_collision(
            particles.pos_x, particles.pos_y,
            particles.force_x, particles.force_y,
            particles.alive
        )

        # 6. Particle-environment interaction (energy, waste, dormancy)
        environment.particle_environment_interaction(
            particles.pos_x, particles.pos_y,
            particles.vel_x, particles.vel_y,
            particles.energy, particles.dormant_ticks,
            particles.alive,
            0.0
        )
