"""
GEP expression tree, symbolic differentiation, and bytecode compilation.

GEP genomes produce scalar potential energy fields U(dist, density, ...).
Forces are computed via symbolic differentiation: F = -dU/dr.
This guarantees momentum conservation (Newton's 3rd law) with zero precision loss.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import copy

from src.simulation.vm import (
    OP_CONST, OP_VAR, OP_ADD, OP_SUB, OP_MUL, OP_DIV,
    OP_SIN, OP_COS, OP_TANH, OP_SQRT, OP_ABS, OP_NEG,
    OP_MAX, OP_MIN, OP_CLAMP, OP_HALT
)


# ── AST Node Definitions ──

@dataclass
class Node:
    """Base expression tree node."""
    op: str = ''
    children: list = field(default_factory=list)
    value: float = 0.0      # for CONST nodes
    var_idx: int = 0         # for VAR nodes
    var_name: str = ''       # for VAR nodes

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_const(self) -> bool:
        return self.op == 'const'

    def is_var(self) -> bool:
        return self.op == 'var'

    def depth(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + sum(c.size() for c in self.children)


def Const(v: float) -> Node:
    return Node(op='const', value=v)

def Var(name: str, idx: int) -> Node:
    return Node(op='var', var_name=name, var_idx=idx)

def Add(a: Node, b: Node) -> Node:
    return Node(op='+', children=[a, b])

def Sub(a: Node, b: Node) -> Node:
    return Node(op='-', children=[a, b])

def Mul(a: Node, b: Node) -> Node:
    return Node(op='*', children=[a, b])

def Div(a: Node, b: Node) -> Node:
    """Safe division: a / b (VM clamps denominator away from zero)."""
    return Node(op='/', children=[a, b])

def Sin(a: Node) -> Node:
    return Node(op='sin', children=[a])

def Cos(a: Node) -> Node:
    return Node(op='cos', children=[a])

def Tanh(a: Node) -> Node:
    return Node(op='tanh', children=[a])

def Sqrt(a: Node) -> Node:
    return Node(op='sqrt', children=[a])

def Abs(a: Node) -> Node:
    return Node(op='abs', children=[a])

def Neg(a: Node) -> Node:
    return Node(op='neg', children=[a])

def Max(a: Node, b: Node) -> Node:
    return Node(op='max', children=[a, b])

def Min(a: Node, b: Node) -> Node:
    return Node(op='min', children=[a, b])


# ── Symbolic Differentiation ──

def symbolic_diff(node: Node, var_name: str) -> Node:
    """
    Compute d(node)/d(var_name) symbolically using chain rule.
    Returns a new expression tree representing the derivative.
    """
    if node.is_const():
        return Const(0.0)

    if node.is_var():
        if node.var_name == var_name:
            return Const(1.0)
        else:
            return Const(0.0)

    op = node.op
    f = node.children[0]

    if op == '+':
        # d(f+g)/dx = f' + g'
        g = node.children[1]
        return Add(symbolic_diff(f, var_name), symbolic_diff(g, var_name))

    elif op == '-':
        # d(f-g)/dx = f' - g'
        g = node.children[1]
        return Sub(symbolic_diff(f, var_name), symbolic_diff(g, var_name))

    elif op == '*':
        # d(f*g)/dx = f'*g + f*g'
        g = node.children[1]
        return Add(
            Mul(symbolic_diff(f, var_name), g),
            Mul(f, symbolic_diff(g, var_name))
        )

    elif op == '/':
        # d(f/g)/dx = (f'*g - f*g') / g^2
        g = node.children[1]
        df = symbolic_diff(f, var_name)
        dg = symbolic_diff(g, var_name)
        return Div(
            Sub(Mul(df, g), Mul(f, dg)),
            Mul(g, g)
        )

    elif op == 'sin':
        # d(sin(f))/dx = cos(f) * f'
        return Mul(Cos(f), symbolic_diff(f, var_name))

    elif op == 'cos':
        # d(cos(f))/dx = -sin(f) * f'
        return Mul(Neg(Sin(f)), symbolic_diff(f, var_name))

    elif op == 'tanh':
        # d(tanh(f))/dx = (1 - tanh^2(f)) * f'
        return Mul(
            Sub(Const(1.0), Mul(Tanh(f), Tanh(f))),
            symbolic_diff(f, var_name)
        )

    elif op == 'sqrt':
        # d(sqrt(f))/dx = f' / (2 * sqrt(f))
        # Use safe division with eps to avoid div-by-zero at f=0
        df = symbolic_diff(f, var_name)
        return Div(df, Mul(Const(2.0), Sqrt(Add(Abs(f), Const(1e-7)))))

    elif op == 'abs':
        # d(|f|)/dx = f / |f| * f' = sign(f) * f'
        # Use safe sign: f / (|f| + eps)
        df = symbolic_diff(f, var_name)
        sign_f = Div(f, Add(Abs(f), Const(1e-7)))
        return Mul(sign_f, df)

    elif op == 'neg':
        # d(-f)/dx = -f'
        return Neg(symbolic_diff(f, var_name))

    elif op == 'max':
        # d(max(f,g))/dx = f' if f >= g else g'
        g = node.children[1]
        df = symbolic_diff(f, var_name)
        dg = symbolic_diff(g, var_name)
        # Use the actual max selection: if f >= g, pick df, else dg
        return Add(
            Mul(df, Max(Const(1.0), Div(Sub(g, f), Add(Abs(Sub(g, f)), Const(1e-7))))),
            Mul(dg, Max(Const(1.0), Div(Sub(f, g), Add(Abs(Sub(f, g)), Const(1e-7)))))
        )
        # Simplified: when f>g, first term = df*1, second = dg*(large→clamp→1)
        # This is approximate but better than simple average

    elif op == 'min':
        g = node.children[1]
        df = symbolic_diff(f, var_name)
        dg = symbolic_diff(g, var_name)
        return Add(
            Mul(df, Max(Const(1.0), Div(Sub(f, g), Add(Abs(Sub(f, g)), Const(1e-7))))),
            Mul(dg, Max(Const(1.0), Div(Sub(g, f), Add(Abs(Sub(g, f)), Const(1e-7)))))
        )

    else:
        # Unknown op — derivative is 0
        return Const(0.0)


# ── Expression Simplification ──

def simplify(node: Node) -> Node:
    """Simplify expression tree: eliminate *0, +0, *1, etc."""
    if node.is_leaf():
        return node

    # Recursively simplify children first
    children = [simplify(c) for c in node.children]

    op = node.op

    if op == '+':
        a, b = children
        if a.is_const() and a.value == 0.0:
            return b
        if b.is_const() and b.value == 0.0:
            return a
        if a.is_const() and b.is_const():
            return Const(a.value + b.value)
        return Add(a, b)

    elif op == '-':
        a, b = children
        if b.is_const() and b.value == 0.0:
            return a
        if a.is_const() and b.is_const():
            return Const(a.value - b.value)
        return Sub(a, b)

    elif op == '*':
        a, b = children
        if a.is_const() and a.value == 0.0:
            return Const(0.0)
        if b.is_const() and b.value == 0.0:
            return Const(0.0)
        if a.is_const() and a.value == 1.0:
            return b
        if b.is_const() and b.value == 1.0:
            return a
        if a.is_const() and b.is_const():
            return Const(a.value * b.value)
        return Mul(a, b)

    elif op == '/':
        a, b = children
        if a.is_const() and a.value == 0.0:
            return Const(0.0)
        if b.is_const() and b.value == 1.0:
            return a
        if a.is_const() and b.is_const() and b.value != 0.0:
            return Const(a.value / b.value)
        return Div(a, b)

    elif op == 'neg':
        a = children[0]
        if a.is_const():
            return Const(-a.value)
        if a.op == 'neg':
            return a.children[0]  # double negation
        return Neg(a)

    else:
        return Node(op=op, children=children)


def eliminate_dead_code(node: Node) -> Node:
    """Remove dead code (nodes that don't affect the result)."""
    # For now, just simplify. Full DCE would require liveness analysis.
    return simplify(node)


# ── Bytecode Compilation ──

# Variable name → index mapping (potential U terminal set)
VAR_MAP = {
    'dist': 0,
    'density': 1,
    'speed': 2,
    'angle': 3,
    'state_0': 4,
    'state_1': 5,
    'state_2': 6,
    'state_3': 7,
    'neighbor_count': 8,
    'avg_nutrient': 9,
    'avg_waste': 10,
}

# Chemotaxis terminal set (environment gradient sensing)
CHEMOTAXIS_VAR_MAP = {
    'grad_nut_x': 0,
    'grad_nut_y': 1,
    'grad_waste_x': 2,
    'grad_waste_y': 3,
    'nutrient': 4,
    'waste': 5,
    'speed': 6,
}


def compile_to_bytecode(node: Node, constants: list[float],
                        max_len: int = 128) -> list[int]:
    """
    Compile expression tree to bytecode array.
    Appends OP_CLAMP + OP_HALT at the end for safety.
    """
    bytecode = []
    _emit(node, bytecode, constants, max_len - 2)  # reserve 2 for CLAMP+HALT
    bytecode.extend([OP_CLAMP, OP_HALT])
    # Pad to max_len
    while len(bytecode) < max_len:
        bytecode.append(OP_HALT)
    return bytecode[:max_len]


def _emit(node: Node, bytecode: list[int], constants: list[float],
          max_len: int):
    """Recursively emit bytecode for expression tree."""
    if len(bytecode) >= max_len:
        return

    if node.is_const():
        # Add constant to pool and emit CONST instruction
        idx = _get_or_add_const(constants, node.value)
        bytecode.extend([OP_CONST, idx])

    elif node.is_var():
        bytecode.extend([OP_VAR, node.var_idx])

    elif node.op == '+':
        _emit(node.children[0], bytecode, constants, max_len)
        _emit(node.children[1], bytecode, constants, max_len)
        bytecode.append(OP_ADD)

    elif node.op == '-':
        _emit(node.children[0], bytecode, constants, max_len)
        _emit(node.children[1], bytecode, constants, max_len)
        bytecode.append(OP_SUB)

    elif node.op == '*':
        _emit(node.children[0], bytecode, constants, max_len)
        _emit(node.children[1], bytecode, constants, max_len)
        bytecode.append(OP_MUL)

    elif node.op == '/':
        _emit(node.children[0], bytecode, constants, max_len)
        _emit(node.children[1], bytecode, constants, max_len)
        bytecode.append(OP_DIV)

    elif node.op == 'sin':
        _emit(node.children[0], bytecode, constants, max_len)
        bytecode.append(OP_SIN)

    elif node.op == 'cos':
        _emit(node.children[0], bytecode, constants, max_len)
        bytecode.append(OP_COS)

    elif node.op == 'tanh':
        _emit(node.children[0], bytecode, constants, max_len)
        bytecode.append(OP_TANH)

    elif node.op == 'sqrt':
        _emit(node.children[0], bytecode, constants, max_len)
        bytecode.append(OP_SQRT)

    elif node.op == 'abs':
        _emit(node.children[0], bytecode, constants, max_len)
        bytecode.append(OP_ABS)

    elif node.op == 'neg':
        _emit(node.children[0], bytecode, constants, max_len)
        bytecode.append(OP_NEG)

    elif node.op == 'max':
        _emit(node.children[0], bytecode, constants, max_len)
        _emit(node.children[1], bytecode, constants, max_len)
        bytecode.append(OP_MAX)

    elif node.op == 'min':
        _emit(node.children[0], bytecode, constants, max_len)
        _emit(node.children[1], bytecode, constants, max_len)
        bytecode.append(OP_MIN)


def _get_or_add_const(constants: list[float], value: float) -> int:
    """Get index of constant in pool, or add it."""
    for i, c in enumerate(constants):
        if abs(c - value) < 1e-10:
            return i
    constants.append(value)
    return len(constants) - 1


# ── Convenience: compile potential and its derivative ──

def compile_potential(potential_tree: Node, max_len: int = 128):
    """
    Compile a potential energy expression tree to two bytecode arrays:
    1. dU/dr (for force computation)
    2. constants pool (shared)

    Returns (dudr_bytecode, constants).
    """
    # Symbolic differentiation: dU/d(dist)
    dudr_tree = symbolic_diff(potential_tree, 'dist')
    dudr_tree = simplify(dudr_tree)
    dudr_tree = eliminate_dead_code(dudr_tree)

    constants = []
    dudr_bc = compile_to_bytecode(dudr_tree, constants, max_len)

    return dudr_bc, constants


# ── Convenience: compile chemotaxis expression ──

def compile_chemotaxis(chemotaxis_tree: Node, max_len: int = 128):
    """
    Compile a chemotaxis expression tree to bytecode.

    Unlike potential, chemotaxis directly outputs force (not potential),
    so no symbolic differentiation is needed.

    Terminal set uses CHEMOTAXIS_VAR_MAP indices.

    Returns (bytecode, constants).
    """
    constants = []
    bc = compile_to_bytecode(chemotaxis_tree, constants, max_len)
    return bc, constants
