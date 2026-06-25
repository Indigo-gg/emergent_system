"""
Environment layer tests: nutrient/waste fields, gradients, diffusion.
Run with: python -m pytest tests/test_environment.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import taichi as ti
_ti_initialized = False


def _ensure_ti():
    global _ti_initialized
    if not _ti_initialized:
        ti.init(arch=ti.cpu, debug=True)
        _ti_initialized = True


def _make_cfg():
    return {
        'simulation': {'num_particles': 100, 'particle_state_dim': 4,
                       'dt': 0.01, 'damping_gamma': 0.1, 'bucket_max': 128},
        'world': {'width': 50.0, 'height': 50.0, 'cell_size': 2.0},
        'environment': {
            'nutrient_diffuse_rate': 0.08,
            'nutrient_decay_rate': 0.001,
            'nutrient_inject_interval': 60,
            'nutrient_patch_count': 3,
            'nutrient_patch_amount': 1.5,
            'nutrient_drift_speed': 0.002,
            'waste_production_rate': 0.15,
            'waste_diffuse_rate': 0.05,
            'waste_decay_rate': 0.005,
            'waste_metabolism_factor': 2.0,
            'base_metabolism': 0.01,
            'move_cost': 0.005,
            'absorb_rate': 0.5,
            'dormant_metabolism': 0.001,
            'max_dormant_ticks': 600,
        },
        'safety': {
            'hard_repulsion_epsilon': 0.01,
            'hard_repulsion_strength': 100.0,
        },
    }


def test_environment_init():
    """EnvironmentLayer initializes with correct grid dimensions."""
    _ensure_ti()
    from src.simulation.environment import EnvironmentLayer

    cfg = _make_cfg()
    env = EnvironmentLayer(cfg)

    assert env.rows > 0
    assert env.cols > 0
    assert env.num_cells == env.rows * env.cols
    assert env.n_patches == 3
    print("PASS: test_environment_init")


def test_environment_initialize():
    """initialize() sets fields to zero and places hotspots."""
    _ensure_ti()
    from src.simulation.environment import EnvironmentLayer

    cfg = _make_cfg()
    env = EnvironmentLayer(cfg)
    env.initialize(seed=42)

    nut = env.nutrient_field.to_numpy()
    waste = env.waste_field.to_numpy()

    assert np.all(nut == 0.0), "Nutrient field should be zero after init"
    assert np.all(waste == 0.0), "Waste field should be zero after init"

    # Check hotspots are placed
    patch_cx = env.patch_cx.to_numpy()
    assert patch_cx[0] != -1.0, "First patch should be active"
    print("PASS: test_environment_initialize")


def test_environment_step_diffusion():
    """environment_step() applies diffusion and decay."""
    _ensure_ti()
    from src.simulation.environment import EnvironmentLayer

    cfg = _make_cfg()
    env = EnvironmentLayer(cfg)
    env.initialize(seed=42)

    # Manually inject some nutrient
    @ti.kernel
    def _inject():
        for i, j in env.nutrient_field:
            if i == 5 and j == 5:
                env.nutrient_field[i, j] = 10.0

    _inject()

    nut_before = env.nutrient_field.to_numpy()
    assert nut_before[5, 5] == 10.0

    # Run environment step
    env.environment_step()

    nut_after = env.nutrient_field.to_numpy()

    # Diffusion should spread the nutrient
    assert nut_after[5, 5] < 10.0, "Nutrient should decay/diffuse"
    assert np.sum(nut_after) > 0, "Total nutrient should remain > 0"
    print("PASS: test_environment_step_diffusion")


def test_compute_gradients():
    """compute_gradients() produces non-zero gradients near injected nutrient."""
    _ensure_ti()
    from src.simulation.environment import EnvironmentLayer

    cfg = _make_cfg()
    env = EnvironmentLayer(cfg)
    env.initialize(seed=42)

    # Inject nutrient at center
    @ti.kernel
    def _inject():
        for i, j in env.nutrient_field:
            ci = env.rows // 2
            cj = env.cols // 2
            if i == ci and j == cj:
                env.nutrient_field[i, j] = 10.0

    _inject()

    env.compute_gradients()

    grad_nx = env.grad_nut_x.to_numpy()
    grad_ny = env.grad_nut_y.to_numpy()

    # Gradients should be non-zero at cells ADJACENT to the injection point
    # (the injection point itself has zero gradient due to symmetric neighbors)
    ci = env.rows // 2
    cj = env.cols // 2
    has_nonzero = False
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = ci + di, cj + dj
            if 0 <= ni < env.rows and 0 <= nj < env.cols:
                if grad_nx[ni, nj] != 0.0 or grad_ny[ni, nj] != 0.0:
                    has_nonzero = True
    assert has_nonzero, "Gradients should be non-zero near nutrient source"
    print("PASS: test_compute_gradients")


def test_sample_field():
    """sample_field() interpolates correctly."""
    _ensure_ti()
    from src.simulation.environment import EnvironmentLayer

    cfg = _make_cfg()
    env = EnvironmentLayer(cfg)
    env.initialize(seed=42)

    # Set a known value
    @ti.kernel
    def _set():
        env.nutrient_field[5, 5] = 1.0

    _set()

    # sample_field is a @ti.func — must be called from a kernel
    cs = env.cell_size
    # Sample at exact grid point (5, 5) center: x = 5*cs + cs*0.5, y = 5*cs + cs*0.5
    # But sample_field converts (x,y) to grid via gx = x/cs, so:
    # To land exactly on grid cell (5,5), use x = 5.0*cs + 0.01 (very close to grid edge)
    # Actually, to get exact value, sample at the center of cell (5,5):
    x = 5.0 * cs + cs * 0.5  # = 5.5 * cs
    y = 5.0 * cs + cs * 0.5  # = 5.5 * cs
    # This gives gx=5.5, gy=5.5 → bilinear of 4 cells → 0.25
    # Instead, sample at grid integer position: x = 5.0*cs, y = 5.0*cs
    # This gives gx=5.0, gy=5.0 → j0=5, i0=5, fx=0, fy=0 → exact value
    x = 5.0 * cs
    y = 5.0 * cs

    result = ti.field(dtype=ti.f32, shape=(1,))

    @ti.kernel
    def _sample():
        result[0] = env.sample_field(env.nutrient_field, x, y)

    _sample()
    val = result.to_numpy()[0]
    assert abs(val - 1.0) < 0.01, f"Expected ~1.0 at grid point, got {val}"
    print("PASS: test_sample_field")


def test_particle_environment_interaction():
    """Particles absorb nutrient and gain energy."""
    _ensure_ti()
    from src.simulation.particles import ParticleSystem
    from src.simulation.environment import EnvironmentLayer

    cfg_full = _make_cfg()
    particles = ParticleSystem(cfg_full)
    particles.initialize(42)

    env = EnvironmentLayer(cfg_full)
    env.initialize(seed=42)

    # Inject nutrient near particles
    @ti.kernel
    def _inject():
        for i, j in env.nutrient_field:
            env.nutrient_field[i, j] = 5.0

    _inject()

    energy_before = particles.energy.to_numpy().copy()

    env.particle_environment_interaction(
        particles.pos_x, particles.pos_y,
        particles.vel_x, particles.vel_y,
        particles.energy, particles.dormant_ticks,
        particles.alive,
        0.0
    )

    energy_after = particles.energy.to_numpy()
    # At least some particles should have gained energy
    assert np.any(energy_after > energy_before), "Some particles should gain energy from nutrient"
    print("PASS: test_particle_environment_interaction")


def test_to_numpy_from_numpy():
    """Checkpoint: export and restore environment state."""
    _ensure_ti()
    from src.simulation.environment import EnvironmentLayer

    cfg = _make_cfg()
    env = EnvironmentLayer(cfg)
    env.initialize(seed=42)

    # Modify state
    @ti.kernel
    def _modify():
        env.nutrient_field[3, 3] = 7.0
        env.waste_field[4, 4] = 3.0

    _modify()

    # Export
    data = env.to_numpy()
    assert 'nutrient' in data
    assert 'waste' in data

    # Create new env and restore
    env2 = EnvironmentLayer(cfg)
    env2.from_numpy(data)

    nut2 = env2.nutrient_field.to_numpy()
    assert abs(nut2[3, 3] - 7.0) < 0.001, "Nutrient should be restored"
    print("PASS: test_to_numpy_from_numpy")


if __name__ == '__main__':
    test_environment_init()
    test_environment_initialize()
    test_environment_step_diffusion()
    test_compute_gradients()
    test_sample_field()
    test_particle_environment_interaction()
    test_to_numpy_from_numpy()
    print("\nAll environment tests passed!")
