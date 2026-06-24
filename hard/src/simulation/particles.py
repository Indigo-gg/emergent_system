"""
Particle state management using Taichi fields.

State per particle: [x, y, vx, vy, state_0..state_3, force_x, force_y]
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

        # Force accumulators
        self.force_x = ti.field(dtype=ti.f32, shape=self.n)
        self.force_y = ti.field(dtype=ti.f32, shape=self.n)

        # Alive flag
        self.alive = ti.field(dtype=ti.i32, shape=self.n)

    @ti.kernel
    def initialize(self, seed: ti.i32):
        """Randomly initialize all particles."""
        rng = ti.random()
        for i in range(self.n):
            self.pos_x[i] = ti.random() * self.world_w
            self.pos_y[i] = ti.random() * self.world_h
            self.vel_x[i] = (ti.random() - 0.5) * 0.1
            self.vel_y[i] = (ti.random() - 0.5) * 0.1
            for d in range(self.state_dim):
                self.state[i, d] = (ti.random() - 0.5) * 0.1
            self.force_x[i] = 0.0
            self.force_y[i] = 0.0
            self.alive[i] = 1

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
