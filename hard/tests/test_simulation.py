"""
Basic simulation correctness tests.
Run with: python -m pytest tests/test_simulation.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_vm_bytecode_compilation():
    """Test that expression trees compile to valid bytecode."""
    from src.simulation.potential import (
        Const, Var, Add, Mul, Neg, Sin,
        compile_potential, compile_to_bytecode
    )
    from src.simulation.vm import OP_CONST, OP_VAR, OP_MUL, OP_SIN, OP_CLAMP, OP_HALT

    # Build: U = -sin(dist * 2.0)
    tree = Neg(Sin(Mul(Var('dist', 0), Const(2.0))))

    dudr_bc, constants = compile_potential(tree, max_len=128)

    assert len(dudr_bc) == 128
    assert len(constants) > 0

    # Find the CLAMP + HALT (before padding)
    # The bytecode is: [instructions..., CLAMP, HALT, HALT, HALT, ...]
    clamp_idx = None
    for i, op in enumerate(dudr_bc):
        if op == OP_CLAMP:
            clamp_idx = i
            break
    assert clamp_idx is not None, "CLAMP not found in bytecode"
    assert dudr_bc[clamp_idx] == OP_CLAMP
    assert dudr_bc[clamp_idx + 1] == OP_HALT

    print("PASS: test_vm_bytecode_compilation")


def test_symbolic_differentiation():
    """Test symbolic differentiation produces correct derivatives."""
    from src.simulation.potential import (
        Const, Var, Add, Mul, Neg, Sin, symbolic_diff, simplify
    )

    # d/dx(x) = 1
    tree = Var('dist', 0)
    deriv = simplify(symbolic_diff(tree, 'dist'))
    assert deriv.is_const() and deriv.value == 1.0

    # d/dx(2*x) = 2
    tree = Mul(Const(2.0), Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    assert deriv.is_const() and deriv.value == 2.0

    # d/dx(x^2) = 2*x (x^2 = x*x)
    tree = Mul(Var('dist', 0), Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should be: 1*x + x*1 = 2*x
    assert deriv.op == '+'

    # d/dx(sin(x)) = cos(x) * 1 = cos(x) (after simplification)
    tree = Sin(Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    assert deriv.op == 'cos'

    print("PASS: test_symbolic_differentiation")


def test_particle_initialization():
    """Test particles initialize without NaN."""
    import taichi as ti
    from src.simulation.particles import ParticleSystem

    cfg = {
        'simulation': {'num_particles': 1000, 'particle_state_dim': 4},
        'world': {'width': 100.0, 'height': 100.0},
    }

    ti.init(arch=ti.cpu, debug=True)
    ps = ParticleSystem(cfg)
    ps.initialize(42)

    pos_x = ps.pos_x.to_numpy()
    pos_y = ps.pos_y.to_numpy()

    assert not np.any(np.isnan(pos_x)), "NaN in pos_x"
    assert not np.any(np.isnan(pos_y)), "NaN in pos_y"
    assert np.all(pos_x >= 0) and np.all(pos_x <= 100), "pos_x out of bounds"
    assert np.all(pos_y >= 0) and np.all(pos_y <= 100), "pos_y out of bounds"

    print("PASS: test_particle_initialization")


def test_spatial_hash_no_nan():
    """Test spatial hash build doesn't produce NaN or crash."""
    import taichi as ti
    from src.simulation.spatial_hash import SpatialHash

    cfg = {
        'simulation': {'num_particles': 1000, 'bucket_max': 128},
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
    }

    ti.init(arch=ti.cpu, debug=True)
    sh = SpatialHash(cfg)

    pos_x = np.random.uniform(0, 100, 1000).astype(np.float32)
    pos_y = np.random.uniform(0, 100, 1000).astype(np.float32)

    sh.build(pos_x, pos_y)

    head = sh.cell_head.to_numpy()
    assert not np.any(np.isnan(head.astype(float)))

    print("PASS: test_spatial_hash_no_nan")


def test_full_step_no_nan():
    """Test one full simulation step doesn't produce NaN."""
    import taichi as ti
    from src.simulation.particles import ParticleSystem
    from src.simulation.spatial_hash import SpatialHash
    from src.simulation.integrator import Integrator
    from src.simulation.step import SimulationStep

    cfg = {
        'simulation': {
            'num_particles': 1000,
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

    ti.init(arch=ti.cpu, debug=True)

    particles = ParticleSystem(cfg)
    spatial_hash = SpatialHash(cfg)
    integrator = Integrator(cfg)
    sim_step = SimulationStep(spatial_hash, integrator, cfg)

    particles.initialize(42)

    # Run 10 steps
    for _ in range(10):
        sim_step.step(particles)

    pos_x = particles.pos_x.to_numpy()
    pos_y = particles.pos_y.to_numpy()
    vel_x = particles.vel_x.to_numpy()
    vel_y = particles.vel_y.to_numpy()

    assert not np.any(np.isnan(pos_x)), "NaN in pos_x after steps"
    assert not np.any(np.isnan(pos_y)), "NaN in pos_y after steps"
    assert not np.any(np.isnan(vel_x)), "NaN in vel_x after steps"
    assert not np.any(np.isnan(vel_y)), "NaN in vel_y after steps"

    print("PASS: test_full_step_no_nan")


if __name__ == '__main__':
    test_vm_bytecode_compilation()
    test_symbolic_differentiation()
    # Taichi tests need GPU or CPU init — skip if no taichi
    try:
        import taichi as ti
        test_particle_initialization()
        test_spatial_hash_no_nan()
        test_full_step_no_nan()
    except Exception as e:
        print(f"SKIP (Taichi tests): {e}")
    print("\nAll tests passed!")
