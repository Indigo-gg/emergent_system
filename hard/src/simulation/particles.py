"""
Particle state management using Taichi fields.

State per particle: [x, y, vx, vy, state_0..state_3, energy, force_x, force_y, alive, dormant_ticks]
All stored as separate Taichi fields (SoA layout for GPU coalescing).
"""

import taichi as ti


@ti.data_oriented
class ParticleSystem:
    """Manages particle state as Taichi fields."""

    def __init__(self, cfg: dict):
        self.n = cfg['simulation']['num_particles']
        self.state_dim = cfg['simulation']['particle_state_dim']
        self.world_w = cfg['world']['width']
        self.world_h = cfg['world']['height']

        # Position
        self.pos_x = ti.field(dtype=ti.f32, shape=self.n)
        self.pos_y = ti.field(dtype=ti.f32, shape=self.n)

        # Velocity
        self.vel_x = ti.field(dtype=ti.f32, shape=self.n)
        self.vel_y = ti.field(dtype=ti.f32, shape=self.n)

        # Internal state (configurable dimensions)
        self.state = ti.field(dtype=ti.f32, shape=(self.n, self.state_dim))

        # Energy (v6: particle energy for metabolism/dormancy)
        self.energy = ti.field(dtype=ti.f32, shape=self.n)

        # Dormant ticks (v6: how long particle has been dormant, 0 = active)
        self.dormant_ticks = ti.field(dtype=ti.i32, shape=self.n)

        # Force accumulators
        self.force_x = ti.field(dtype=ti.f32, shape=self.n)
        self.force_y = ti.field(dtype=ti.f32, shape=self.n)

        # Alive flag
        self.alive = ti.field(dtype=ti.i32, shape=self.n)

    @ti.kernel
    def initialize(self, seed: ti.i32):
        """Randomly initialize all particles using seeded PRNG.

        Uses mulberry32 hash-based PRNG for reproducibility across runs.
        ti.random() cannot be seeded, so we implement our own.
        """
        C1 = ti.u32(1664525)
        C2 = ti.u32(1013904223)
        C3 = ti.u32(2654435769)
        MAX_U32 = ti.f32(4294967296.0)

        for i in range(self.n):
            # Seeded PRNG: mulberry32 hash
            state = ti.u32(seed) * C1 + ti.u32(i) * C2 + ti.u32(12345)
            state = state ^ (state >> 16)
            state = state * C3
            state = state ^ (state >> 16)
            state = state * C3
            state = state ^ (state >> 16)
            r1 = ti.cast(state, ti.f32) / MAX_U32

            state2 = state * C1 + C2
            state2 = state2 ^ (state2 >> 16)
            state2 = state2 * C3
            r2 = ti.cast(state2, ti.f32) / MAX_U32

            state3 = state2 * C1 + C2
            state3 = state3 ^ (state3 >> 16)
            state3 = state3 * C3
            r3 = ti.cast(state3, ti.f32) / MAX_U32

            state4 = state3 * C1 + C2
            state4 = state4 ^ (state4 >> 16)
            state4 = state4 * C3
            r4 = ti.cast(state4, ti.f32) / MAX_U32

            self.pos_x[i] = r1 * self.world_w
            self.pos_y[i] = r2 * self.world_h
            self.vel_x[i] = (r3 - 0.5) * 0.1
            self.vel_y[i] = (r4 - 0.5) * 0.1
            for d in range(self.state_dim):
                state_d = state4 * (ti.u32(d) + ti.u32(1)) * C1 + C2
                state_d = state_d ^ (state_d >> 16)
                state_d = state_d * C3
                r_d = ti.cast(state_d, ti.f32) / MAX_U32
                self.state[i, d] = (r_d - 0.5) * 0.1
            self.force_x[i] = 0.0
            self.force_y[i] = 0.0
            self.alive[i] = 1
            self.energy[i] = 1.0
            self.dormant_ticks[i] = 0

    @ti.kernel
    def reset_forces(self):
        """Zero out force accumulators before each step."""
        for i in range(self.n):
            self.force_x[i] = 0.0
            self.force_y[i] = 0.0

    @ti.kernel
    def count_alive(self) -> ti.i32:
        """Count particles still alive."""
        count = 0
        for i in range(self.n):
            count += self.alive[i]
        return count

    @ti.func
    def get_pos(self, i: int) -> ti.math.vec2:
        return ti.math.vec2(self.pos_x[i], self.pos_y[i])

    @ti.func
    def get_vel(self, i: int) -> ti.math.vec2:
        return ti.math.vec2(self.vel_x[i], self.vel_y[i])
