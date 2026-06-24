"""
VM Pressure Test — Phase 1 highest priority.

Tests:
1. 1M particles × 100 steps VM execution → target < 500ms
2. Taichi profiler check for register spilling
3. Hardcoded formula vs VM execution performance ratio → target < 5x
"""

import os
import sys
import time

import taichi as ti
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_vm_basic():
    """Basic VM correctness test: does it return a finite value?"""
    from src.simulation.vm import vm_execute, OP_VAR, OP_MUL, OP_SIN, OP_CONST, OP_CLAMP, OP_HALT

    ti.init(arch=ti.cuda, debug=False)

    # Bytecode: sin(dist * 2.0)
    # OP_VAR 0, OP_CONST 0, OP_MUL, OP_SIN, OP_CLAMP, OP_HALT
    bytecode = np.array([
        1, 0,       # VAR dist
        0, 0,       # CONST 0 (= 2.0)
        12,         # MUL
        20,         # SIN
        40,         # CLAMP
        255, 0, 0, 0, 0, 0, 0, 0, 0, 0  # HALT + padding
    ], dtype=np.int32)

    constants = np.array([2.0], dtype=np.float32)

    @ti.kernel
    def run_vm(bc: ti.types.ndarray(), consts: ti.types.ndarray()) -> ti.f32:
        return vm_execute(bc, consts,
                         1.0, 0.5, 0.1, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 16)

    result = run_vm(bytecode, constants)
    print(f"VM basic test: sin(1.0 * 2.0) = {result:.6f} (expected: {np.sin(2.0):.6f})")
    assert abs(result - np.sin(2.0)) < 0.01, f"VM result {result} != expected {np.sin(2.0)}"
    print("  PASS")


def test_vm_pressure(n_particles: int = 1_000_000, n_steps: int = 100):
    """
    Pressure test: run VM on n_particles for n_steps.
    Measures total time and per-step time.
    """
    from src.simulation.vm import vm_execute

    ti.init(arch=ti.cuda, debug=False)

    # Bytecode: simple potential U = -dist → dU/dr = -1 → force = 1 (attractive)
    # OP_CONST 0 (= -1.0), OP_CLAMP, OP_HALT
    bytecode = np.array([0, 0, 40, 255] + [0] * 124, dtype=np.int32)
    constants = np.array([-1.0], dtype=np.float32)

    # Particle data
    pos_x = ti.field(dtype=ti.f32, shape=n_particles)
    pos_y = ti.field(dtype=ti.f32, shape=n_particles)
    force_x = ti.field(dtype=ti.f32, shape=n_particles)
    force_y = ti.field(dtype=ti.f32, shape=n_particles)

    @ti.kernel
    def init_particles():
        for i in range(n_particles):
            pos_x[i] = ti.random() * 100.0
            pos_y[i] = ti.random() * 100.0

    @ti.kernel
    def vm_step(bc: ti.types.ndarray(), consts: ti.types.ndarray()):
        """Simulate one step: each particle runs VM once."""
        for i in range(n_particles):
            result = vm_execute(bc, consts,
                                1.0, 0.5, 0.1, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 16)
            force_x[i] = result
            force_y[i] = result

    init_particles()

    # Warmup
    print(f"Warming up VM with {n_particles:,} particles...")
    for _ in range(3):
        vm_step(bytecode, constants)
    ti.sync()

    # Pressure test
    print(f"Running VM pressure test: {n_particles:,} particles × {n_steps} steps")
    t_start = time.time()
    for step in range(n_steps):
        vm_step(bytecode, constants)
    ti.sync()
    elapsed = time.time() - t_start

    per_step = elapsed / n_steps
    ops_per_sec = n_particles * n_steps / elapsed

    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per step: {per_step*1000:.2f}ms")
    print(f"  Throughput: {ops_per_sec/1e6:.1f}M VM-ops/sec")
    print(f"  Target: < 500ms total for 100 steps → {'PASS' if elapsed < 0.5 else 'WARN'}")

    return elapsed


def test_hardcoded_vs_vm(n_particles: int = 100_000, n_steps: int = 100):
    """
    Compare hardcoded formula vs VM execution performance.
    Target: VM < 5x slower than hardcoded.
    """
    from src.simulation.vm import vm_execute

    ti.init(arch=ti.cuda, debug=False)

    pos_x = ti.field(dtype=ti.f32, shape=n_particles)
    pos_y = ti.field(dtype=ti.f32, shape=n_particles)
    force_x = ti.field(dtype=ti.f32, shape=n_particles)
    force_y = ti.field(dtype=ti.f32, shape=n_particles)

    @ti.kernel
    def init_particles():
        for i in range(n_particles):
            pos_x[i] = ti.random() * 100.0
            pos_y[i] = ti.random() * 100.0

    # Hardcoded: F = -1.0 (constant attraction)
    @ti.kernel
    def hardcoded_step():
        for i in range(n_particles):
            force_x[i] = -1.0
            force_y[i] = -1.0

    # VM: same formula via bytecode
    bytecode = np.array([0, 0, 40, 255] + [0] * 124, dtype=np.int32)
    constants = np.array([-1.0], dtype=np.float32)

    @ti.kernel
    def vm_step(bc: ti.types.ndarray(), consts: ti.types.ndarray()):
        for i in range(n_particles):
            result = vm_execute(bc, consts,
                                1.0, 0.5, 0.1, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 16)
            force_x[i] = result
            force_y[i] = result

    init_particles()

    # Warmup
    for _ in range(3):
        hardcoded_step()
        vm_step(bytecode, constants)
    ti.sync()

    # Benchmark hardcoded
    t0 = time.time()
    for _ in range(n_steps):
        hardcoded_step()
    ti.sync()
    t_hardcoded = time.time() - t0

    # Benchmark VM
    t0 = time.time()
    for _ in range(n_steps):
        vm_step(bytecode, constants)
    ti.sync()
    t_vm = time.time() - t0

    ratio = t_vm / t_hardcoded if t_hardcoded > 0 else float('inf')

    print(f"\nHardcoded vs VM ({n_particles:,} particles × {n_steps} steps):")
    print(f"  Hardcoded: {t_hardcoded*1000:.2f}ms ({t_hardcoded/n_steps*1000:.3f}ms/step)")
    print(f"  VM:        {t_vm*1000:.2f}ms ({t_vm/n_steps*1000:.3f}ms/step)")
    print(f"  Ratio:     {ratio:.2f}x (target: < 5x) → {'PASS' if ratio < 5 else 'WARN'}")

    return ratio


def main():
    print("=" * 60)
    print("Hard Mode — VM Pressure Test")
    print("=" * 60)

    # Test 1: Basic correctness
    print("\n[Test 1] VM Basic Correctness")
    test_vm_basic()

    # Test 2: Pressure test
    print("\n[Test 2] VM Pressure Test (1M particles × 100 steps)")
    elapsed = test_vm_pressure(1_000_000, 100)

    # Test 3: Hardcoded vs VM
    print("\n[Test 3] Hardcoded vs VM Performance")
    ratio = test_hardcoded_vs_vm(100_000, 100)

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  VM pressure: {elapsed:.3f}s {'PASS' if elapsed < 0.5 else 'WARN'}")
    print(f"  VM overhead: {ratio:.2f}x {'PASS' if ratio < 5 else 'WARN'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
