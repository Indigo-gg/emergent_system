"""
GEP genome encoding/decoding tests.
Run with: python -m pytest tests/test_gep.py -v
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_random_genome_creation():
    """Random genome has correct gene lengths (2 active genes)."""
    from src.evolution.genome import random_genome, GEPGenome

    cfg = {
        'gep': {
            'head_length': 8,
            'constant_range': [-5.0, 5.0],
        }
    }
    rng = random.Random(42)
    genome = random_genome(cfg, rng)

    assert isinstance(genome, GEPGenome)
    assert len(genome.potential_gene) == 17  # 8 + 9
    assert len(genome.chemotaxis_gene) == 17  # v6: chemotaxis gene
    # state_gene, sense_gene, metabolism_gene removed (unused in simulation)
    assert len(genome.state_gene) == 0
    assert len(genome.sense_gene) == 0
    assert len(genome.metabolism_gene) == 0
    assert genome.head_length == 8
    assert genome.random_seed != 0
    print("PASS: test_random_genome_creation")


def test_gene_decode_simple_terminal():
    """Decode a gene that's just a terminal."""
    from src.evolution.genome import decode_gene
    from src.simulation.potential import Var

    gene = ['dist'] + ['speed'] * 16  # pad
    tree = decode_gene(gene, head_length=8)
    assert tree.is_var() and tree.var_name == 'dist'
    print("PASS: test_gene_decode_simple_terminal")


def test_gene_decode_simple_constant():
    """Decode a gene that's just a constant."""
    from src.evolution.genome import decode_gene
    from src.simulation.potential import Const

    gene = ['3.14'] + ['0.0'] * 16
    tree = decode_gene(gene, head_length=8)
    assert tree.is_const() and abs(tree.value - 3.14) < 0.01
    print("PASS: test_gene_decode_simple_constant")


def test_gene_decode_unary_function():
    """Decode sin(dist)."""
    from src.evolution.genome import decode_gene
    from src.simulation.potential import Node

    gene = ['sin', 'dist'] + ['0.0'] * 15
    tree = decode_gene(gene, head_length=8)
    assert tree.op == 'sin'
    assert len(tree.children) == 1
    assert tree.children[0].is_var() and tree.children[0].var_name == 'dist'
    print("PASS: test_gene_decode_unary_function")


def test_gene_decode_binary_function():
    """Decode dist + density."""
    from src.evolution.genome import decode_gene

    gene = ['+', 'dist', 'density'] + ['0.0'] * 14
    tree = decode_gene(gene, head_length=8)
    assert tree.op == '+'
    assert len(tree.children) == 2
    assert tree.children[0].var_name == 'dist'
    assert tree.children[1].var_name == 'density'
    print("PASS: test_gene_decode_binary_function")


def test_gene_decode_nested():
    """Decode sin(dist * 2.0)."""
    from src.evolution.genome import decode_gene

    gene = ['sin', '*', 'dist', '2.0'] + ['0.0'] * 13
    tree = decode_gene(gene, head_length=8)
    assert tree.op == 'sin'
    inner = tree.children[0]
    assert inner.op == '*'
    assert inner.children[0].var_name == 'dist'
    assert abs(inner.children[1].value - 2.0) < 0.01
    print("PASS: test_gene_decode_nested")


def test_gene_decode_complex():
    """Decode: (dist + sin(density)) * cos(speed)."""
    from src.evolution.genome import decode_gene

    gene = ['*', '+', 'dist', 'sin', 'density', 'cos', 'speed', 'angle',
            '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0', '0.0']
    tree = decode_gene(gene, head_length=8)
    assert tree.op == '*'
    assert tree.children[0].op == '+'
    assert tree.children[1].op == 'cos'
    print("PASS: test_gene_decode_complex")


def test_gene_decode_empty():
    """Decode empty gene returns constant 0."""
    from src.evolution.genome import decode_gene

    tree = decode_gene([], head_length=8)
    assert tree.is_const() and tree.value == 0.0
    print("PASS: test_gene_decode_empty")


def test_tree_encode_decode_roundtrip():
    """Encode a tree to gene, then decode back — structure should be preserved."""
    from src.simulation.potential import Var, Const, Add, Mul, Sin
    from src.evolution.genome import encode_tree, decode_gene

    # Original tree: sin(dist * 2.0) + density
    original = Add(
        Sin(Mul(Var('dist', 0), Const(2.0))),
        Var('density', 1)
    )

    gene = encode_tree(original, head_length=8)
    assert len(gene) == 17

    decoded = decode_gene(gene, head_length=8)
    # The decoded tree should have the same top-level structure
    assert decoded.op == '+'
    assert decoded.children[0].op == 'sin'
    assert decoded.children[1].is_var()
    print("PASS: test_tree_encode_decode_roundtrip")


def test_genome_to_formula():
    """Genome.to_formula() produces readable string."""
    from src.evolution.genome import GEPGenome, decode_gene

    genome = GEPGenome(
        potential_gene=['sin', '*', 'dist', '2.0', 'density',
                        'cos', 'speed', 'angle'] + ['0.0'] * 9,
        head_length=8,
    )
    formula = genome.to_formula()
    assert isinstance(formula, str)
    assert len(formula) > 0
    print(f"PASS: test_genome_to_formula → {formula}")


def test_genome_compile():
    """Genome compiles to valid bytecode."""
    from src.evolution.genome import GEPGenome

    genome = GEPGenome(
        potential_gene=['*', 'dist', '2.0'] + ['0.0'] * 14,
        head_length=8,
    )
    genome.compile(128)

    assert genome._compiled
    assert len(genome._potential_bytecode) == 128
    assert len(genome._potential_constants) > 0
    print("PASS: test_genome_compile")


def test_genome_invalidate_cache():
    """Invalidate cache resets compiled state."""
    from src.evolution.genome import GEPGenome

    genome = GEPGenome(
        potential_gene=['dist'] + ['0.0'] * 16,
        head_length=8,
    )
    genome.compile(128)
    assert genome._compiled

    genome.invalidate_cache()
    assert not genome._compiled
    assert genome._potential_bytecode == []
    print("PASS: test_genome_invalidate_cache")


def test_genome_get_id():
    """Genome ID is derived from gene prefix."""
    from src.evolution.genome import GEPGenome

    genome = GEPGenome(
        potential_gene=['+', 'dist', 'density', 'sin', 'speed',
                        'cos', 'angle', '2.0'] + ['0.0'] * 9,
        head_length=8,
    )
    gid = genome.get_id()
    assert isinstance(gid, str)
    assert 'dist' in gid
    print(f"PASS: test_genome_get_id → {gid}")


if __name__ == '__main__':
    test_random_genome_creation()
    test_gene_decode_simple_terminal()
    test_gene_decode_simple_constant()
    test_gene_decode_unary_function()
    test_gene_decode_binary_function()
    test_gene_decode_nested()
    test_gene_decode_complex()
    test_gene_decode_empty()
    test_tree_encode_decode_roundtrip()
    test_genome_to_formula()
    test_genome_compile()
    test_genome_invalidate_cache()
    test_genome_get_id()
    print("\nAll GEP tests passed!")
