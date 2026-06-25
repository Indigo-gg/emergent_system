"""
Integrator tests: damping, speed limit, displacement limit, periodic boundary.

Uses Taichi fields (GPU) instead of numpy arrays to match production usage.
Run with: python -m pytest tests/test_integrator.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti


def _make_cfg():
    return {
        'simulation': {
            'num_particles': 100,
            'particle_state_dim': 4,
            'dt': 0.01,
            'damping_gamma': 0.1,
            'bucket_max': 128,
        },
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
        'gep': {'vm_stack_depth': 16, 'bytecode_length': 128},
        'safety': {
            'max_speed': 5.0,
            'max_force': 10.0,
            'max_displacement_ratio': 0.5,
        },
    }


def _make_fields(n):
    """Create Taichi fields for testing."""
    return {
        'pos_x': ti.field(dtype=ti.f32, shape=n),
        'pos_y': ti.field(dtype=ti.f32, shape=n),
        'vel_x': ti.field(dtype=ti.f32, shape=n),
        'vel_y': ti.field(dtype=ti.f32, shape=n),
        'force_x': ti.field(dtype=ti.f32, shape=n),
        'force_y': ti.field(dtype=ti.f32, shape=n),
        'alive': ti.field(dtype=ti.i32, shape=n),
    }


def _load(fields, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive):
    """Load numpy arrays into Taichi fields."""
    fields['pos_x'].from_numpy(pos_x)
    fields['pos_y'].from_numpy(pos_y)
    fields['vel_x'].from_numpy(vel_x)
    fields['vel_y'].from_numpy(vel_y)
    fields['force_x'].from_numpy(force_x)
    fields['force_y'].from_numpy(force_y)
    fields['alive'].from_numpy(alive)


def _dump(fields):
    """Read Taichi fields back to numpy."""
    return (fields['pos_x'].to_numpy(), fields['pos_y'].to_numpy(),
            fields['vel_x'].to_numpy(), fields['vel_y'].to_numpy())


def test_zero_force_no_movement():
    """With zero force and zero velocity, particles don't move."""
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    integ = Integrator(cfg)

    n = 10
    pos_x = np.random.uniform(10, 90, n).astype(np.float32)
    pos_y = np.random.uniform(10, 90, n).astype(np.float32)
    vel_x = np.zeros(n, dtype=np.float32)
    vel_y = np.zeros(n, dtype=np.float32)
    force_x = np.zeros(n, dtype=np.float32)
    force_y = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    pos_x_orig = pos_x.copy()
    pos_y_orig = pos_y.copy()

    f = _make_fields(n)
    _load(f, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)
    integ.step(f['pos_x'], f['pos_y'], f['vel_x'], f['vel_y'],
               f['force_x'], f['force_y'], f['alive'])
    px, py, _, _ = _dump(f)

    np.testing.assert_allclose(px, pos_x_orig, atol=1e-6,
                                err_msg="pos_x changed with zero force/velocity")
    np.testing.assert_allclose(py, pos_y_orig, atol=1e-6,
                                err_msg="pos_y changed with zero force/velocity")
    print("PASS: test_zero_force_no_movement")


def test_damping_reduces_velocity():
    """Viscous damping should reduce velocity over time."""
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    integ = Integrator(cfg)

    n = 10
    pos_x = np.full(n, 50.0, dtype=np.float32)
    pos_y = np.full(n, 50.0, dtype=np.float32)
    vel_x = np.full(n, 3.0, dtype=np.float32)
    vel_y = np.full(n, 4.0, dtype=np.float32)
    force_x = np.zeros(n, dtype=np.float32)
    force_y = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    speed_before = np.sqrt(vel_x[0]**2 + vel_y[0]**2)

    f = _make_fields(n)
    _load(f, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    for _ in range(100):
        integ.step(f['pos_x'], f['pos_y'], f['vel_x'], f['vel_y'],
                   f['force_x'], f['force_y'], f['alive'])

    _, _, vx, vy = _dump(f)
    speed_after = np.sqrt(vx[0]**2 + vy[0]**2)
    assert speed_after < speed_before, \
        f"Speed should decrease: before={speed_before:.4f}, after={speed_after:.4f}"
    print(f"PASS: test_damping_reduces_velocity ({speed_before:.4f} → {speed_after:.4f})")


def test_speed_limit():
    """Velocity should be clamped to max_speed."""
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    cfg['safety']['max_speed'] = 2.0
    integ = Integrator(cfg)

    n = 10
    pos_x = np.full(n, 50.0, dtype=np.float32)
    pos_y = np.full(n, 50.0, dtype=np.float32)
    vel_x = np.zeros(n, dtype=np.float32)
    vel_y = np.zeros(n, dtype=np.float32)
    force_x = np.full(n, 1000.0, dtype=np.float32)
    force_y = np.full(n, 1000.0, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    f = _make_fields(n)
    _load(f, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)
    integ.step(f['pos_x'], f['pos_y'], f['vel_x'], f['vel_y'],
               f['force_x'], f['force_y'], f['alive'])
    _, _, vx, vy = _dump(f)

    speeds = np.sqrt(vx**2 + vy**2)
    assert np.all(speeds <= 2.0 + 0.01), \
        f"Speeds exceed limit: max={np.max(speeds):.4f}, limit=2.0"
    print(f"PASS: test_speed_limit (max speed={np.max(speeds):.4f})")


def test_force_limit():
    """Forces should be clamped to max_force."""
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    cfg['safety']['max_force'] = 5.0
    integ = Integrator(cfg)

    n = 10
    pos_x = np.full(n, 50.0, dtype=np.float32)
    pos_y = np.full(n, 50.0, dtype=np.float32)
    vel_x = np.zeros(n, dtype=np.float32)
    vel_y = np.zeros(n, dtype=np.float32)
    force_x = np.full(n, 9999.0, dtype=np.float32)
    force_y = np.full(n, 9999.0, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    f = _make_fields(n)
    _load(f, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)
    integ.step(f['pos_x'], f['pos_y'], f['vel_x'], f['vel_y'],
               f['force_x'], f['force_y'], f['alive'])
    _, _, vx, vy = _dump(f)

    speeds = np.sqrt(vx**2 + vy**2)
    assert np.all(speeds <= cfg['safety']['max_speed'] + 0.1), \
        f"Speeds after large force: max={np.max(speeds):.4f}"
    print("PASS: test_force_limit")


def test_periodic_boundary():
    """Particles that go past boundary should wrap around."""
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    cfg['simulation']['dt'] = 1.0
    integ = Integrator(cfg)

    n = 5
    pos_x = np.array([99.0, 99.5, 0.5, 50.0, 50.0], dtype=np.float32)
    pos_y = np.array([50.0, 50.0, 50.0, 99.0, 0.5], dtype=np.float32)
    vel_x = np.array([4.0, 4.0, -4.0, 0.0, 0.0], dtype=np.float32)
    vel_y = np.array([0.0, 0.0, 0.0, 4.0, -4.0], dtype=np.float32)
    force_x = np.zeros(n, dtype=np.float32)
    force_y = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    f = _make_fields(n)
    _load(f, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)
    integ.step(f['pos_x'], f['pos_y'], f['vel_x'], f['vel_y'],
               f['force_x'], f['force_y'], f['alive'])
    px, py, _, _ = _dump(f)

    assert np.all(px >= 0) and np.all(px < 100), f"pos_x out of bounds: {px}"
    assert np.all(py >= 0) and np.all(py < 100), f"pos_y out of bounds: {py}"
    print(f"PASS: test_periodic_boundary (pos_x={px}, pos_y={py})")


def test_dead_particles_skipped():
    """Dead particles (alive=0) should not be updated."""
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    integ = Integrator(cfg)

    n = 5
    pos_x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    pos_y = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    vel_x = np.ones(n, dtype=np.float32)
    vel_y = np.ones(n, dtype=np.float32)
    force_x = np.ones(n, dtype=np.float32)
    force_y = np.ones(n, dtype=np.float32)
    alive = np.array([1, 0, 1, 0, 1], dtype=np.int32)

    pos_x_orig = pos_x.copy()

    f = _make_fields(n)
    _load(f, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)
    integ.step(f['pos_x'], f['pos_y'], f['vel_x'], f['vel_y'],
               f['force_x'], f['force_y'], f['alive'])
    px, _, _, _ = _dump(f)

    assert px[1] == pos_x_orig[1], "Dead particle 1 moved"
    assert px[3] == pos_x_orig[3], "Dead particle 3 moved"
    assert px[0] != pos_x_orig[0], "Alive particle 0 didn't move"
    print("PASS: test_dead_particles_skipped")


def test_no_nan_after_steps():
    """No NaN in positions/velocities after many steps."""
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    integ = Integrator(cfg)

    n = 100
    pos_x = np.random.uniform(10, 90, n).astype(np.float32)
    pos_y = np.random.uniform(10, 90, n).astype(np.float32)
    vel_x = np.random.uniform(-1, 1, n).astype(np.float32)
    vel_y = np.random.uniform(-1, 1, n).astype(np.float32)
    force_x = np.random.uniform(-5, 5, n).astype(np.float32)
    force_y = np.random.uniform(-5, 5, n).astype(np.float32)
    alive = np.ones(n, dtype=np.int32)

    f = _make_fields(n)
    _load(f, pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    for _ in range(1000):
        force_x[:] = np.random.uniform(-10, 10, n).astype(np.float32)
        force_y[:] = np.random.uniform(-10, 10, n).astype(np.float32)
        f['force_x'].from_numpy(force_x)
        f['force_y'].from_numpy(force_y)
        integ.step(f['pos_x'], f['pos_y'], f['vel_x'], f['vel_y'],
                   f['force_x'], f['force_y'], f['alive'])

    px, py, vx, vy = _dump(f)
    assert not np.any(np.isnan(px)), "NaN in pos_x"
    assert not np.any(np.isnan(py)), "NaN in pos_y"
    assert not np.any(np.isnan(vx)), "NaN in vel_x"
    assert not np.any(np.isnan(vy)), "NaN in vel_y"
    print("PASS: test_no_nan_after_steps")


if __name__ == '__main__':
    test_zero_force_no_movement()
    test_damping_reduces_velocity()
    test_speed_limit()
    test_force_limit()
    test_periodic_boundary()
    test_dead_particles_skipped()
    test_no_nan_after_steps()
    print("\nAll integrator tests passed!")
