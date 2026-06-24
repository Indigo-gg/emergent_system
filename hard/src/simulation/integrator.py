"""
Integrator: applies forces, damping, velocity/position limits, and boundary conditions.

Physics per step:
  F_total = F_potential - gamma * v   (viscous damping)
  v' = v + F_total * dt               (velocity update)
  v' = clamp(v', max_speed)           (speed limit)
  dx = v' * dt                         (displacement)
  dx = clamp(dx, max_disp)            (displacement limit)
  x' = x + dx                          (position update)
  x' = wrap(x', boundary)             (periodic boundary)
"""

import taichi as ti


@ti.data_oriented
class Integrator:
    """Velocity Verlet-like integrator with damping and limits."""

    def __init__(self, cfg: dict):
        self.dt = cfg['simulation']['dt']
        self.gamma = cfg['simulation']['damping_gamma']
        self.max_speed = cfg['safety']['max_speed']
        self.max_force = cfg['safety']['max_force']
        self.cell_size = cfg['world']['cell_size']
        self.max_disp = cfg['safety']['max_displacement_ratio'] * self.cell_size
        self.world_w = cfg['world']['width']
        self.world_h = cfg['world']['height']

    @ti.kernel
    def step(self,
             pos_x: ti.types.ndarray(), pos_y: ti.types.ndarray(),
             vel_x: ti.types.ndarray(), vel_y: ti.types.ndarray(),
             force_x: ti.types.ndarray(), force_y: ti.types.ndarray(),
             alive: ti.types.ndarray()):
        """Advance all particles by one timestep."""
        n = pos_x.shape[0]
        for i in range(n):
            if alive[i] == 0:
                continue

            # Clamp force magnitude
            fx = force_x[i]
            fy = force_y[i]
            f_mag = ti.sqrt(fx * fx + fy * fy)
            if f_mag > self.max_force:
                fx *= self.max_force / f_mag
                fy *= self.max_force / f_mag

            # Viscous damping: F_total = F - gamma * v
            fx -= self.gamma * vel_x[i]
            fy -= self.gamma * vel_y[i]

            # Velocity update
            vx = vel_x[i] + fx * self.dt
            vy = vel_y[i] + fy * self.dt

            # Speed limit
            speed = ti.sqrt(vx * vx + vy * vy)
            if speed > self.max_speed:
                vx *= self.max_speed / speed
                vy *= self.max_speed / speed

            # Displacement limit
            dx = vx * self.dt
            dy = vy * self.dt
            disp = ti.sqrt(dx * dx + dy * dy)
            if disp > self.max_disp:
                dx *= self.max_disp / disp
                dy *= self.max_disp / disp

            # Position update
            px = pos_x[i] + dx
            py = pos_y[i] + dy

            # Periodic boundary
            px = px - ti.floor(px / self.world_w) * self.world_w
            py = py - ti.floor(py / self.world_h) * self.world_h

            # Write back
            vel_x[i] = vx
            vel_y[i] = vy
            pos_x[i] = px
            pos_y[i] = py
