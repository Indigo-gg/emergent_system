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

        # L-J forces can be large at close range; use separate limit
        lj_cfg = cfg.get('lj', {})
        if lj_cfg.get('enabled', False):
            self.max_force = max(self.max_force, lj_cfg.get('max_force', 50.0))

    @ti.kernel
    def step(self,
             pos_x: ti.template(), pos_y: ti.template(),
             vel_x: ti.template(), vel_y: ti.template(),
             force_x: ti.template(), force_y: ti.template(),
             alive: ti.template(),
             phase: ti.template(),
             phase_enabled: ti.i32):
        """Advance all particles by one timestep.

        Phase 4: When phase_enabled, max_speed is modulated by oscillation phase.
        phase_mod = 0.5 + 0.5 * sin(phase) → range [0.0, 1.0]
        effective_speed = max_speed * phase_mod
        This creates collective "breathing" when particles synchronize.
        """
        n = pos_x.shape[0]
        for i in range(n):
            if alive[i] == 0:
                continue

            # Phase modulation: oscillates between 50% and 100% of max_speed
            speed_limit = self.max_speed
            if phase_enabled == 1:
                phase_mod = 0.5 + 0.5 * ti.sin(phase[i])
                speed_limit = self.max_speed * phase_mod

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

            # Speed limit (phase-modulated)
            speed = ti.sqrt(vx * vx + vy * vy)
            if speed > speed_limit:
                vx *= speed_limit / speed
                vy *= speed_limit / speed

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
