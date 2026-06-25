"""
VM opcode correctness tests.
Run with: python -m pytest tests/test_vm.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Shared Taichi init (only once per process)
_ti_initialized = False


def _ensure_ti():
    global _ti_initialized
    if not _ti_initialized:
        import taichi as ti
        ti.init(arch=ti.cpu, debug=True)
        _ti_initialized = True


def _run_vm(bytecode, constants, var_values, stack_depth=16):
    """Helper: run VM on CPU and return result.

    var_values: list of 11 values [dist, density, speed, angle, state0-3, neighbor_count, avg_nutrient, avg_waste]
    """
    import taichi as ti
    _ensure_ti()
    from src.simulation.vm import vm_execute

    bc_np = np.array(bytecode, dtype=np.int32)
    const_np = np.array(constants, dtype=np.float32)

    # Pad var_values to 11 elements if needed
    v = list(var_values) + [0.0] * (11 - len(var_values))

    @ti.kernel
    def _run(bc: ti.types.ndarray(), consts: ti.types.ndarray(),
             d: ti.f32, dn: ti.f32, sp: ti.f32, an: ti.f32,
             s0: ti.f32, s1: ti.f32, s2: ti.f32, s3: ti.f32,
             nc: ti.f32, avg_nut: ti.f32, avg_wst: ti.f32) -> ti.f32:
        return vm_execute(bc, consts, d, dn, sp, an, s0, s1, s2, s3, nc, avg_nut, avg_wst, 16)

    return _run(bc_np, const_np, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10])


# ── Opcode Tests ──

def test_op_const():
    """OP_CONST: push a constant value."""
    from src.simulation.vm import OP_CONST, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CLAMP, OP_HALT]
    constants = [3.14]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 3.14) < 0.01, f"Expected 3.14, got {result}"
    print("PASS: test_op_const")


def test_op_var():
    """OP_VAR: push a variable value."""
    from src.simulation.vm import OP_VAR, OP_CLAMP, OP_HALT
    bytecode = [OP_VAR, 0, OP_CLAMP, OP_HALT]
    constants = [0.0]
    result = _run_vm(bytecode, constants, [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(result - 1.5) < 0.01, f"Expected 1.5, got {result}"
    print("PASS: test_op_var")


def test_op_var_all_indices():
    """OP_VAR: push each variable index (11 variables for v6)."""
    from src.simulation.vm import OP_VAR, OP_CLAMP, OP_HALT
    for idx in range(11):
        bytecode = [OP_VAR, idx, OP_CLAMP, OP_HALT]
        constants = [0.0]
        vars = [float(i) for i in range(11)]
        result = _run_vm(bytecode, constants, vars)
        assert abs(result - float(idx)) < 0.01, f"var[{idx}] expected {idx}, got {result}"
    print("PASS: test_op_var_all_indices")


def test_op_add():
    """OP_ADD: pop two, push sum."""
    from src.simulation.vm import OP_CONST, OP_ADD, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CONST, 1, OP_ADD, OP_CLAMP, OP_HALT]
    constants = [2.0, 3.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 5.0) < 0.01, f"Expected 5.0, got {result}"
    print("PASS: test_op_add")


def test_op_sub():
    """OP_SUB: pop two, push difference."""
    from src.simulation.vm import OP_CONST, OP_SUB, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CONST, 1, OP_SUB, OP_CLAMP, OP_HALT]
    constants = [5.0, 2.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 3.0) < 0.01, f"Expected 3.0, got {result}"
    print("PASS: test_op_sub")


def test_op_mul():
    """OP_MUL: pop two, push product."""
    from src.simulation.vm import OP_CONST, OP_MUL, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CONST, 1, OP_MUL, OP_CLAMP, OP_HALT]
    constants = [3.0, 4.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 12.0) < 0.01, f"Expected 12.0, got {result}"
    print("PASS: test_op_mul")


def test_op_div():
    """OP_DIV: pop two, push safe division."""
    from src.simulation.vm import OP_CONST, OP_DIV, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CONST, 1, OP_DIV, OP_CLAMP, OP_HALT]
    constants = [10.0, 4.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 2.5) < 0.01, f"Expected 2.5, got {result}"
    print("PASS: test_op_div")


def test_op_div_by_zero():
    """OP_DIV: division by zero should be safe (clamped)."""
    from src.simulation.vm import OP_CONST, OP_DIV, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CONST, 1, OP_DIV, OP_CLAMP, OP_HALT]
    constants = [10.0, 0.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert np.isfinite(result), f"Expected finite, got {result}"
    print("PASS: test_op_div_by_zero")


def test_op_sin():
    """OP_SIN: pop, push sin."""
    from src.simulation.vm import OP_CONST, OP_SIN, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_SIN, OP_CLAMP, OP_HALT]
    constants = [1.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - np.sin(1.0)) < 0.01, f"Expected {np.sin(1.0):.4f}, got {result}"
    print("PASS: test_op_sin")


def test_op_cos():
    """OP_COS: pop, push cos."""
    from src.simulation.vm import OP_CONST, OP_COS, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_COS, OP_CLAMP, OP_HALT]
    constants = [1.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - np.cos(1.0)) < 0.01, f"Expected {np.cos(1.0):.4f}, got {result}"
    print("PASS: test_op_cos")


def test_op_tanh():
    """OP_TANH: pop, push tanh."""
    from src.simulation.vm import OP_CONST, OP_TANH, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_TANH, OP_CLAMP, OP_HALT]
    constants = [0.5]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - np.tanh(0.5)) < 0.01, f"Expected {np.tanh(0.5):.4f}, got {result}"
    print("PASS: test_op_tanh")


def test_op_sqrt():
    """OP_SQRT: pop, push sqrt(|a|)."""
    from src.simulation.vm import OP_CONST, OP_SQRT, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_SQRT, OP_CLAMP, OP_HALT]
    constants = [9.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 3.0) < 0.01, f"Expected 3.0, got {result}"
    print("PASS: test_op_sqrt")


def test_op_sqrt_negative():
    """OP_SQRT: sqrt of negative uses abs first."""
    from src.simulation.vm import OP_CONST, OP_SQRT, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_SQRT, OP_CLAMP, OP_HALT]
    constants = [-9.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 3.0) < 0.01, f"Expected 3.0 (sqrt of |−9|), got {result}"
    print("PASS: test_op_sqrt_negative")


def test_op_abs():
    """OP_ABS: pop, push |a|."""
    from src.simulation.vm import OP_CONST, OP_ABS, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_ABS, OP_CLAMP, OP_HALT]
    constants = [-7.5]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 7.5) < 0.01, f"Expected 7.5, got {result}"
    print("PASS: test_op_abs")


def test_op_neg():
    """OP_NEG: pop, push -a."""
    from src.simulation.vm import OP_CONST, OP_NEG, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_NEG, OP_CLAMP, OP_HALT]
    constants = [4.2]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - (-4.2)) < 0.01, f"Expected -4.2, got {result}"
    print("PASS: test_op_neg")


def test_op_max():
    """OP_MAX: pop two, push max."""
    from src.simulation.vm import OP_CONST, OP_MAX, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CONST, 1, OP_MAX, OP_CLAMP, OP_HALT]
    constants = [3.0, 7.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 7.0) < 0.01, f"Expected 7.0, got {result}"
    print("PASS: test_op_max")


def test_op_min():
    """OP_MIN: pop two, push min."""
    from src.simulation.vm import OP_CONST, OP_MIN, OP_CLAMP, OP_HALT
    bytecode = [OP_CONST, 0, OP_CONST, 1, OP_MIN, OP_CLAMP, OP_HALT]
    constants = [3.0, 7.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 3.0) < 0.01, f"Expected 3.0, got {result}"
    print("PASS: test_op_min")


def test_op_clamp():
    """OP_CLAMP: clamp value to [-100, 100]."""
    from src.simulation.vm import OP_CONST, OP_CLAMP, OP_HALT
    # Test positive overflow
    bytecode = [OP_CONST, 0, OP_CLAMP, OP_HALT]
    constants = [999.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 100.0) < 0.01, f"Expected 100.0, got {result}"
    # Test negative overflow
    constants2 = [-999.0]
    bytecode2 = [OP_CONST, 0, OP_CLAMP, OP_HALT]
    result2 = _run_vm(bytecode2, constants2, [0.0]*9)
    assert abs(result2 - (-100.0)) < 0.01, f"Expected -100.0, got {result2}"
    print("PASS: test_op_clamp")


def test_op_halt():
    """OP_HALT: execution stops."""
    from src.simulation.vm import OP_CONST, OP_HALT, OP_ADD
    # Push 1.0, halt, then ADD (should not execute)
    bytecode = [OP_CONST, 0, OP_HALT, OP_ADD, OP_ADD]
    constants = [1.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert abs(result - 1.0) < 0.01, f"Expected 1.0, got {result}"
    print("PASS: test_op_halt")


def test_complex_expression():
    """Test: sin(dist * 2.0) + cos(1.0)."""
    from src.simulation.vm import (OP_CONST, OP_VAR, OP_MUL, OP_SIN,
                                    OP_COS, OP_ADD, OP_CLAMP, OP_HALT)
    # dist=1.0, expected = sin(1.0*2.0) + cos(1.0) = sin(2) + cos(1)
    bytecode = [
        OP_VAR, 0,       # push dist (=1.0)
        OP_CONST, 0,     # push 2.0
        OP_MUL,          # dist * 2.0 = 2.0
        OP_SIN,          # sin(2.0)
        OP_CONST, 1,     # push 1.0
        OP_COS,          # cos(1.0)
        OP_ADD,          # sin(2.0) + cos(1.0)
        OP_CLAMP,
        OP_HALT,
    ]
    constants = [2.0, 1.0]
    vars = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = _run_vm(bytecode, constants, vars)
    expected = np.sin(2.0) + np.cos(1.0)
    assert abs(result - expected) < 0.01, f"Expected {expected:.4f}, got {result}"
    print("PASS: test_complex_expression")


def test_vm_unknown_opcode():
    """Unknown opcode should halt gracefully."""
    from src.simulation.vm import OP_CONST, OP_HALT
    bytecode = [OP_CONST, 0, 99, OP_HALT]  # 99 is unknown
    constants = [1.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert np.isfinite(result), f"Expected finite, got {result}"
    print("PASS: test_vm_unknown_opcode")


def test_vm_stack_underflow():
    """Stack underflow should be handled gracefully."""
    from src.simulation.vm import OP_ADD, OP_CLAMP, OP_HALT
    # ADD with empty stack
    bytecode = [OP_ADD, OP_CLAMP, OP_HALT]
    constants = [0.0]
    result = _run_vm(bytecode, constants, [0.0]*9)
    assert np.isfinite(result), f"Expected finite, got {result}"
    print("PASS: test_vm_stack_underflow")


if __name__ == '__main__':
    test_op_const()
    test_op_var()
    test_op_var_all_indices()
    test_op_add()
    test_op_sub()
    test_op_mul()
    test_op_div()
    test_op_div_by_zero()
    test_op_sin()
    test_op_cos()
    test_op_tanh()
    test_op_sqrt()
    test_op_sqrt_negative()
    test_op_abs()
    test_op_neg()
    test_op_max()
    test_op_min()
    test_op_clamp()
    test_op_halt()
    test_complex_expression()
    test_vm_unknown_opcode()
    test_vm_stack_underflow()
    print("\nAll VM tests passed!")
