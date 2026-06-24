"""
Integrator tests: damping, speed limit, displacement limit, periodic boundary.
Run with: python -m pytest tests/test_integrator.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def test_zero_force_no_movement():
    """With zero force and zero velocity, particles don't move."""
    import taichi as ti
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

    integ.step(pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    np.testing.assert_allclose(pos_x, pos_x_orig, atol=1e-6,
                                err_msg="pos_x changed with zero force/velocity")
    np.testing.assert_allclose(pos_y, pos_y_orig, atol=1e-6,
                                err_msg="pos_y changed with zero force/velocity")
    print("PASS: test_zero_force_no_movement")


def test_damping_reduces_velocity():
    """Viscous damping should reduce velocity over time."""
    import taichi as ti
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

    # Run several steps — velocity should decrease due to damping
    for _ in range(100):
        integ.step(pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    speed_after = np.sqrt(vel_x[0]**2 + vel_y[0]**2)
    assert speed_after < speed_before, \
        f"Speed should decrease: before={speed_before:.4f}, after={speed_after:.4f}"
    print(f"PASS: test_damping_reduces_velocity ({speed_before:.4f} → {speed_after:.4f})")


def test_speed_limit():
    """Velocity should be clamped to max_speed."""
    import taichi as ti
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
    # Large force to push speed above limit
    force_x = np.full(n, 1000.0, dtype=np.float32)
    force_y = np.full(n, 1000.0, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    integ.step(pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    speeds = np.sqrt(vel_x**2 + vel_y**2)
    max_speed = 2.0
    assert np.all(speeds <= max_speed + 0.01), \
        f"Speeds exceed limit: max={np.max(speeds):.4f}, limit={max_speed}"
    print(f"PASS: test_speed_limit (max speed={np.max(speeds):.4f})")


def test_force_limit():
    """Forces should be clamped to max_force."""
    import taichi as ti
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
    # Huge force
    force_x = np.full(n, 9999.0, dtype=np.float32)
    force_y = np.full(n, 9999.0, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    # One step with large force — velocity should be limited
    integ.step(pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    # After one step, velocity should be bounded
    speeds = np.sqrt(vel_x**2 + vel_y**2)
    assert np.all(speeds <= cfg['safety']['max_speed'] + 0.1), \
        f"Speeds after large force: max={np.max(speeds):.4f}"
    print("PASS: test_force_limit")


def test_periodic_boundary():
    """Particles that go past boundary should wrap around."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.integrator import Integrator

    cfg = _make_cfg()
    cfg['simulation']['dt'] = 1.0  # large dt to move particles far
    integ = Integrator(cfg)

    n = 5
    # Particles near boundary with velocity pointing outward
    pos_x = np.array([99.0, 99.5, 0.5, 50.0, 50.0], dtype=np.float32)
    pos_y = np.array([50.0, 50.0, 50.0, 99.0, 0.5], dtype=np.float32)
    vel_x = np.array([4.0, 4.0, -4.0, 0.0, 0.0], dtype=np.float32)
    vel_y = np.array([0.0, 0.0, 0.0, 4.0, -4.0], dtype=np.float32)
    force_x = np.zeros(n, dtype=np.float32)
    force_y = np.zeros(n, dtype=np.float32)
    alive = np.ones(n, dtype=np.int32)

    integ.step(pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    # All positions should be within [0, 100)
    assert np.all(pos_x >= 0) and np.all(pos_x < 100), \
        f"pos_x out of bounds: {pos_x}"
    assert np.all(pos_y >= 0) and np.all(pos_y < 100), \
        f"pos_y out of bounds: {pos_y}"
    print(f"PASS: test_periodic_boundary (pos_x={pos_x}, pos_y={pos_y})")


def test_dead_particles_skipped():
    """Dead particles (alive=0) should not be updated."""
    import taichi as ti
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
    alive = np.array([1, 0, 1, 0, 1], dtype=np.int32)  # particles 1,3 are dead

    pos_x_orig = pos_x.copy()
    pos_y_orig = pos_y.copy()

    integ.step(pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    # Dead particles should not have moved
    assert pos_x[1] == pos_x_orig[1], "Dead particle 1 moved"
    assert pos_x[3] == pos_x_orig[3], "Dead particle 3 moved"
    # Alive particles should have moved
    assert pos_x[0] != pos_x_orig[0], "Alive particle 0 didn't move"
    print("PASS: test_dead_particles_skipped")


def test_no_nan_after_steps():
    """No NaN in positions/velocities after many steps."""
    import taichi as ti
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

    for _ in range(1000):
        # Randomize forces each step
        force_x[:] = np.random.uniform(-10, 10, n).astype(np.float32)
        force_y[:] = np.random.uniform(-10, 10, n).astype(np.float32)
        integ.step(pos_x, pos_y, vel_x, vel_y, force_x, force_y, alive)

    assert not np.any(np.isnan(pos_x)), "NaN in pos_x"
    assert not np.any(np.isnan(pos_y)), "NaN in pos_y"
    assert not np.any(np.isnan(vel_x)), "NaN in vel_x"
    assert not np.any(np.isnan(vel_y)), "NaN in vel_y"
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
