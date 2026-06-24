"""
GEP Genome: data structure for gene expression programming.

A genome contains one or more genes (linear symbol sequences) that encode
expression trees. The gene has a head (can contain functions + terminals)
and a tail (terminals only), guaranteeing valid decodability.

Gene structure:
  gene = head + tail
  head_length = h (configurable, default 8)
  tail_length = h * (n_max - 1) + 1, where n_max = max arity in function set
  gene_length = h + tail_length

For binary functions (n_max=2): tail = h + 1, gene = 2h + 1
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional

from src.simulation.potential import (
    Node, Const, Var, Add, Sub, Mul, Sin, Cos, Tanh, Sqrt, Abs, Neg, Max, Min,
    compile_potential, compile_to_bytecode, VAR_MAP
)
from src.simulation.vm import OP_CLAMP, OP_HALT


# ── Symbol Sets ──

FUNCTION_SET = ['+', '-', '*', 'sin', 'cos', 'tanh', 'sqrt', 'abs', 'max', 'min']
FUNCTION_ARITY = {
    '+': 2, '-': 2, '*': 2, 'max': 2, 'min': 2,
    'sin': 1, 'cos': 1, 'tanh': 1, 'sqrt': 1, 'abs': 1,
}

TERMINAL_SET = list(VAR_MAP.keys())  # 9 terminals

# Default constant pool for random generation
DEFAULT_CONSTANT_RANGE = (-5.0, 5.0)


def random_constant(rng: random.Random, low=-5.0, high=5.0) -> str:
    """Generate a random constant string."""
    return f"{rng.uniform(low, high):.4f}"


def is_function(sym: str) -> bool:
    return sym in FUNCTION_SET


def is_terminal(sym: str) -> bool:
    """Check if symbol is a terminal (variable or constant)."""
    if sym in VAR_MAP:
        return True
    try:
        float(sym)
        return True
    except ValueError:
        return False


# ── Genome Data Structure ──

@dataclass
class GEPGenome:
    """A GEP genome containing genes for potential, state, and sense rules."""

    # Genes (linear symbol sequences)
    potential_gene: list = field(default_factory=list)
    state_gene: list = field(default_factory=list)
    sense_gene: list = field(default_factory=list)

    # Configuration
    head_length: int = 8

    # Metadata
    generation: int = 0
    parent_id: str = ''
    mutation_history: list = field(default_factory=list)
    random_seed: int = 0

    # Evaluation results (filled after simulation)
    fitness: float = 0.0
    features_3d: tuple = (0.0, 0.0, 0.0)
    features_12d: list = field(default_factory=list)

    # Compiled bytecode cache
    _potential_bytecode: list = field(default_factory=list, repr=False)
    _potential_constants: list = field(default_factory=list, repr=False)
    _compiled: bool = field(default=False, repr=False)

    @property
    def tail_length(self) -> int:
        """Tail length for binary function set (n_max=2)."""
        return self.head_length + 1

    @property
    def gene_length(self) -> int:
        return self.head_length + self.tail_length

    def get_id(self) -> str:
        """Generate a unique ID from the potential gene."""
        return '_'.join(str(s) for s in self.potential_gene[:8])

    def to_formula(self) -> str:
        """Convert potential gene to human-readable formula string."""
        tree = decode_gene(self.potential_gene, self.head_length)
        return _tree_to_formula(tree)

    def compile(self, max_bytecode_len: int = 128):
        """Compile potential gene to bytecode (cached)."""
        if self._compiled:
            return
        tree = decode_gene(self.potential_gene, self.head_length)
        bc, consts = compile_potential(tree, max_bytecode_len)
        self._potential_bytecode = bc
        self._potential_constants = consts if consts else [0.0]
        self._compiled = True

    def invalidate_cache(self):
        """Invalidate bytecode cache (after mutation)."""
        self._compiled = False
        self._potential_bytecode = []
        self._potential_constants = []


# ── Gene Decoding (gene → AST) ──

def decode_gene(gene: list, head_length: int) -> Node:
    """
    Decode a GEP gene (linear symbol sequence) into an expression tree.

    The gene is read left-to-right in BFS order. Functions consume their
    arity's worth of subsequent symbols as children.
    """
    if not gene:
        return Const(0.0)

    root, _ = _decode_node(gene, 0, head_length)
    return root


def _decode_node(gene: list, pos: int, head_length: int) -> tuple:
    """
    Recursively decode a node starting at gene[pos].
    Returns (Node, next_position).
    """
    if pos >= len(gene):
        return Const(0.0), pos

    sym = gene[pos]

    if is_function(sym):
        arity = FUNCTION_ARITY[sym]
        children = []
        next_pos = pos + 1
        for _ in range(arity):
            if next_pos >= len(gene):
                # Out of gene — fill with zero constant
                children.append(Const(0.0))
            else:
                child, next_pos = _decode_node(gene, next_pos, head_length)
                children.append(child)

        node = Node(op=sym, children=children)
        return node, next_pos
    else:
        # Terminal
        if sym in VAR_MAP:
            return Var(sym, VAR_MAP[sym]), pos + 1
        else:
            try:
                val = float(sym)
                return Const(val), pos + 1
            except ValueError:
                # Unknown symbol → treat as constant 0
                return Const(0.0), pos + 1


# ── Gene Encoding (AST → gene) ──

def encode_tree(tree: Node, head_length: int) -> list:
    """
    Encode an expression tree into a GEP gene (linear symbol sequence).
    Uses DFS pre-order traversal to match the decoding order.
    """
    tail_length = head_length + 1  # for binary functions
    gene_length = head_length + tail_length

    gene = []
    _encode_dfs(tree, gene, gene_length)

    # Pad with random terminals if gene is too short
    rng = random.Random(42)
    while len(gene) < gene_length:
        gene.append(rng.choice(TERMINAL_SET))

    return gene[:gene_length]


def _encode_dfs(node: Node, gene: list, max_len: int):
    """DFS pre-order encoding of expression tree."""
    if len(gene) >= max_len:
        return

    if node.is_const():
        gene.append(f"{node.value:.4f}")
    elif node.is_var():
        gene.append(node.var_name)
    else:
        gene.append(node.op)
        for child in node.children:
            _encode_dfs(child, gene, max_len)


# ── Random Genome Generation ──

def random_genome(cfg: dict, rng: random.Random = None) -> GEPGenome:
    """Generate a random GEP genome."""
    if rng is None:
        rng = random.Random()

    head_length = cfg['gep']['head_length']
    tail_length = head_length + 1
    gene_length = head_length + tail_length
    constant_range = cfg['gep'].get('constant_range', [-5.0, 5.0])

    # Generate potential gene
    potential_gene = _random_gene(head_length, tail_length, constant_range, rng)

    # Generate state gene (simpler: just use a default)
    state_gene = _random_gene(head_length, tail_length, constant_range, rng)

    # Generate sense gene
    sense_gene = _random_gene(head_length, tail_length, constant_range, rng)

    return GEPGenome(
        potential_gene=potential_gene,
        state_gene=state_gene,
        sense_gene=sense_gene,
        head_length=head_length,
        random_seed=rng.randint(0, 2**31),
    )


def _random_gene(head_length: int, tail_length: int,
                 constant_range: tuple, rng: random.Random) -> list:
    """Generate a random gene string."""
    gene = []

    # Head: mix of functions and terminals
    for _ in range(head_length):
        if rng.random() < 0.6:  # 60% functions in head
            gene.append(rng.choice(FUNCTION_SET))
        else:
            if rng.random() < 0.3:  # 30% of terminals are constants
                gene.append(random_constant(rng, *constant_range))
            else:
                gene.append(rng.choice(TERMINAL_SET))

    # Tail: terminals only
    for _ in range(tail_length):
        if rng.random() < 0.3:
            gene.append(random_constant(rng, *constant_range))
        else:
            gene.append(rng.choice(TERMINAL_SET))

    return gene


# ── Formula String Conversion ──

def _tree_to_formula(node: Node) -> str:
    """Convert expression tree to human-readable formula string."""
    if node.is_const():
        return f"{node.value:.4f}"
    if node.is_var():
        return node.var_name

    op = node.op
    if len(node.children) == 1:
        child_str = _tree_to_formula(node.children[0])
        return f"{op}({child_str})"
    elif len(node.children) == 2:
        left = _tree_to_formula(node.children[0])
        right = _tree_to_formula(node.children[1])
        if op in ('+', '-', '*', '/'):
            return f"({left} {op} {right})"
        else:
            return f"{op}({left}, {right})"
    else:
        return f"{op}(...)"
