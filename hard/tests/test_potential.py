"""
Symbolic differentiation and bytecode compilation tests.
Run with: python -m pytest tests/test_potential.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Symbolic Differentiation Tests ──

def test_diff_constant():
    """d/dx(const) = 0."""
    from src.simulation.potential import Const, symbolic_diff, simplify
    deriv = simplify(symbolic_diff(Const(5.0), 'dist'))
    assert deriv.is_const() and deriv.value == 0.0
    print("PASS: test_diff_constant")


def test_diff_variable_self():
    """d/dx(x) = 1."""
    from src.simulation.potential import Var, symbolic_diff, simplify
    deriv = simplify(symbolic_diff(Var('dist', 0), 'dist'))
    assert deriv.is_const() and deriv.value == 1.0
    print("PASS: test_diff_variable_self")


def test_diff_variable_other():
    """d/dx(y) = 0 (y != x)."""
    from src.simulation.potential import Var, symbolic_diff, simplify
    deriv = simplify(symbolic_diff(Var('density', 1), 'dist'))
    assert deriv.is_const() and deriv.value == 0.0
    print("PASS: test_diff_variable_other")


def test_diff_add():
    """d/dx(f + g) = f' + g'."""
    from src.simulation.potential import Const, Var, Add, symbolic_diff, simplify
    # U = dist + 3.0 → dU/d(dist) = 1
    tree = Add(Var('dist', 0), Const(3.0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    assert deriv.is_const() and deriv.value == 1.0
    print("PASS: test_diff_add")


def test_diff_sub():
    """d/dx(f - g) = f' - g'."""
    from src.simulation.potential import Const, Var, Sub, symbolic_diff, simplify
    # U = dist - 3.0 → dU/d(dist) = 1
    tree = Sub(Var('dist', 0), Const(3.0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    assert deriv.is_const() and deriv.value == 1.0
    print("PASS: test_diff_sub")


def test_diff_mul_constant():
    """d/dx(c * x) = c."""
    from src.simulation.potential import Const, Var, Mul, symbolic_diff, simplify
    # U = 3.0 * dist → dU/d(dist) = 3.0
    tree = Mul(Const(3.0), Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    assert deriv.is_const() and deriv.value == 3.0
    print("PASS: test_diff_mul_constant")


def test_diff_mul_x_squared():
    """d/dx(x * x) = 2*x."""
    from src.simulation.potential import Var, Mul, symbolic_diff, simplify
    # U = dist * dist → dU/d(dist) = 1*dist + dist*1 = 2*dist
    tree = Mul(Var('dist', 0), Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should be Add(dist, dist) or Const(2.0)*dist after simplification
    # The simplify may not fully collapse Add(Var, Var) to Mul(2, Var)
    # but the structure should be an addition
    assert deriv.op == '+'
    print("PASS: test_diff_mul_x_squared")


def test_diff_sin():
    """d/dx(sin(x)) = cos(x)."""
    from src.simulation.potential import Var, Sin, symbolic_diff, simplify
    # U = sin(dist) → dU/d(dist) = cos(dist) * 1 = cos(dist)
    tree = Sin(Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # simplify eliminates *1, so result is just cos(dist)
    assert deriv.op == 'cos', f"Expected 'cos', got '{deriv.op}'"
    print("PASS: test_diff_sin")


def test_diff_cos():
    """d/dx(cos(x)) = -sin(x)."""
    from src.simulation.potential import Var, Cos, symbolic_diff, simplify
    tree = Cos(Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should be Mul(Neg(Sin(dist)), 1) = Neg(Sin(dist))
    assert deriv.op == 'neg' or deriv.op == '*'
    print("PASS: test_diff_cos")


def test_diff_tanh():
    """d/dx(tanh(x)) = 1 - tanh^2(x)."""
    from src.simulation.potential import Var, Tanh, symbolic_diff, simplify
    tree = Tanh(Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should be Mul(Sub(1, Mul(Tanh, Tanh)), 1) = Sub(1, Mul(Tanh, Tanh))
    assert deriv.op in ('-', '*')
    print("PASS: test_diff_tanh")


def test_diff_sqrt():
    """d/dx(sqrt(x)) = 1/(2*sqrt(x))."""
    from src.simulation.potential import Var, Sqrt, symbolic_diff, simplify
    tree = Sqrt(Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should be Div(1, Mul(2, Sqrt(...)))
    assert deriv.op == '/'
    print("PASS: test_diff_sqrt")


def test_diff_abs():
    """d/dx(|x|) = sign(x)."""
    from src.simulation.potential import Var, Abs, symbolic_diff, simplify
    tree = Abs(Var('dist', 0))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should be Mul(sign(dist), 1) = sign(dist)
    assert deriv.op in ('/', '*')
    print("PASS: test_diff_abs")


def test_diff_neg():
    """d/dx(-f) = -f'."""
    from src.simulation.potential import Const, Var, Mul, Neg, symbolic_diff, simplify
    # U = -(3.0 * dist) → dU/d(dist) = -3.0
    tree = Neg(Mul(Const(3.0), Var('dist', 0)))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    assert deriv.is_const() and abs(deriv.value - (-3.0)) < 0.01
    print("PASS: test_diff_neg")


def test_diff_chain_rule():
    """d/dx(sin(2*x)) = 2*cos(2*x)."""
    from src.simulation.potential import Const, Var, Mul, Sin, symbolic_diff, simplify
    # U = sin(2.0 * dist)
    tree = Sin(Mul(Const(2.0), Var('dist', 0)))
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should be Mul(Cos(Mul(2, dist)), 2) after simplification
    # The outer structure should be a multiplication
    assert deriv.op == '*'
    print("PASS: test_diff_chain_rule")


def test_diff_complex():
    """d/dx(1.5*x - sin(x*3.14) * 0.5) — a realistic GEP potential."""
    from src.simulation.potential import (
        Const, Var, Add, Sub, Mul, Sin, symbolic_diff, simplify
    )
    # U = 1.5 * dist - sin(dist * 3.14) * 0.5
    tree = Sub(
        Mul(Const(1.5), Var('dist', 0)),
        Mul(Sin(Mul(Var('dist', 0), Const(3.14))), Const(0.5))
    )
    deriv = simplify(symbolic_diff(tree, 'dist'))
    # Should produce a finite expression
    assert deriv is not None
    # Verify it compiles to bytecode
    from src.simulation.potential import compile_to_bytecode
    constants = []
    bc = compile_to_bytecode(deriv, constants, 128)
    assert len(bc) == 128
    print("PASS: test_diff_complex")


# ── Simplification Tests ──

def test_simplify_add_zero():
    """x + 0 = x."""
    from src.simulation.potential import Var, Const, Add, simplify
    tree = Add(Var('dist', 0), Const(0.0))
    result = simplify(tree)
    assert result.is_var() and result.var_name == 'dist'
    print("PASS: test_simplify_add_zero")


def test_simplify_mul_zero():
    """x * 0 = 0."""
    from src.simulation.potential import Var, Const, Mul, simplify
    tree = Mul(Var('dist', 0), Const(0.0))
    result = simplify(tree)
    assert result.is_const() and result.value == 0.0
    print("PASS: test_simplify_mul_zero")


def test_simplify_mul_one():
    """x * 1 = x."""
    from src.simulation.potential import Var, Const, Mul, simplify
    tree = Mul(Var('dist', 0), Const(1.0))
    result = simplify(tree)
    assert result.is_var() and result.var_name == 'dist'
    print("PASS: test_simplify_mul_one")


def test_simplify_double_neg():
    """-(-x) = x."""
    from src.simulation.potential import Var, Neg, simplify
    tree = Neg(Neg(Var('dist', 0)))
    result = simplify(tree)
    assert result.is_var() and result.var_name == 'dist'
    print("PASS: test_simplify_double_neg")


def test_simplify_const_fold():
    """3 + 4 = 7."""
    from src.simulation.potential import Const, Add, simplify
    tree = Add(Const(3.0), Const(4.0))
    result = simplify(tree)
    assert result.is_const() and abs(result.value - 7.0) < 1e-10
    print("PASS: test_simplify_const_fold")


# ── Bytecode Compilation Tests ──

def test_compile_simple():
    """Compile U = dist → bytecode."""
    from src.simulation.potential import Var, compile_to_bytecode
    from src.simulation.vm import OP_VAR, OP_CLAMP, OP_HALT
    tree = Var('dist', 0)
    constants = []
    bc = compile_to_bytecode(tree, constants, 16)
    # Bytecode: [VAR 0, CLAMP, HALT, HALT, HALT, ...] (padded)
    assert bc[0] == OP_VAR and bc[1] == 0
    assert bc[2] == OP_CLAMP, f"Expected CLAMP at index 2, got {bc[2]}"
    assert bc[3] == OP_HALT, f"Expected HALT at index 3, got {bc[3]}"
    # Rest is HALT padding
    assert all(bc[i] == OP_HALT for i in range(3, 16))
    print("PASS: test_compile_simple")


def test_compile_potential_dudr():
    """Compile U = -dist → dU/dr bytecode."""
    from src.simulation.potential import Var, Neg, compile_potential
    from src.simulation.vm import OP_CONST, OP_CLAMP, OP_HALT
    tree = Neg(Var('dist', 0))
    dudr_bc, constants = compile_potential(tree, 128)
    # d(-dist)/d(dist) = -1
    # Should be: CONST(-1), CLAMP, HALT
    assert dudr_bc[0] == OP_CONST
    assert len(constants) > 0
    assert abs(constants[dudr_bc[1]] - (-1.0)) < 0.01
    print("PASS: test_compile_potential_dudr")


def test_compile_end_to_end():
    """Compile U = sin(dist) → dU/dr → execute on VM."""
    from src.simulation.potential import Var, Sin, compile_potential
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)
    from src.simulation.vm import vm_execute

    tree = Sin(Var('dist', 0))
    dudr_bc, constants = compile_potential(tree, 128)

    bc_np = np.array(dudr_bc, dtype=np.int32)
    const_np = np.array(constants if constants else [0.0], dtype=np.float32)

    @ti.kernel
    def _run(bc: ti.types.ndarray(), consts: ti.types.ndarray(),
             d: ti.f32, dn: ti.f32, sp: ti.f32, an: ti.f32,
             s0: ti.f32, s1: ti.f32, s2: ti.f32, s3: ti.f32,
             nc: ti.f32) -> ti.f32:
        return vm_execute(bc, consts, d, dn, sp, an, s0, s1, s2, s3, nc, 16)

    # Test at dist = 1.0: d(sin(x))/dx = cos(x), so cos(1.0) ≈ 0.5403
    result = _run(bc_np, const_np, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    expected = np.cos(1.0)
    assert abs(result - expected) < 0.1, f"Expected ~{expected:.4f}, got {result}"
    print("PASS: test_compile_end_to_end")


# ── Node Utility Tests ──

def test_node_depth():
    """Node depth calculation."""
    from src.simulation.potential import Var, Const, Add, Mul, Sin
    # sin(dist * 2.0) + 1.0
    tree = Add(Sin(Mul(Var('dist', 0), Const(2.0))), Const(1.0))
    assert tree.depth() == 4  # Add → Sin → Mul → Var/Const
    print("PASS: test_node_depth")


def test_node_size():
    """Node size calculation."""
    from src.simulation.potential import Var, Const, Add, Mul
    # dist + 3.0
    tree = Add(Var('dist', 0), Const(3.0))
    assert tree.size() == 3  # Add + Var + Const
    print("PASS: test_node_size")


if __name__ == '__main__':
    test_diff_constant()
    test_diff_variable_self()
    test_diff_variable_other()
    test_diff_add()
    test_diff_sub()
    test_diff_mul_constant()
    test_diff_mul_x_squared()
    test_diff_sin()
    test_diff_cos()
    test_diff_tanh()
    test_diff_sqrt()
    test_diff_abs()
    test_diff_neg()
    test_diff_chain_rule()
    test_diff_complex()
    test_simplify_add_zero()
    test_simplify_mul_zero()
    test_simplify_mul_one()
    test_simplify_double_neg()
    test_simplify_const_fold()
    test_compile_simple()
    test_compile_potential_dudr()
    test_compile_end_to_end()
    test_node_depth()
    test_node_size()
    print("\nAll potential tests passed!")
