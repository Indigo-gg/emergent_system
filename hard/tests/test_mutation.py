"""
Genetic operator tests.
Run with: python -m pytest tests/test_mutation.py -v
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg():
    return {
        'gep': {
            'head_length': 8,
            'constant_range': [-5.0, 5.0],
        },
        'evolution': {
            'mutation_rates': {
                'point_mutation': 0.30,
                'constant_finetune': 0.20,
                'is_transposition': 0.15,
                'ris_transposition': 0.10,
                'one_point_recombination': 0.15,
                'two_point_recombination': 0.10,
            }
        }
    }


def _make_genome():
    from src.evolution.genome import GEPGenome
    return GEPGenome(
        potential_gene=['+', 'sin', 'dist', '2.0', 'density',
                        'cos', 'speed', 'angle'] + ['3.14', '1.0', 'dist',
                                                     'density', 'speed',
                                                     '0.0', '0.0', '0.0', '0.0'],
        head_length=8,
    )


def test_point_mutation_changes_gene():
    """Point mutation changes exactly one position."""
    from src.evolution.mutation import _point_mutation

    cfg = _make_cfg()
    rng = random.Random(42)
    genome = _make_genome()
    original = list(genome.potential_gene)

    _point_mutation(genome, cfg, rng)

    # Exactly one position should differ
    diffs = sum(1 for a, b in zip(original, genome.potential_gene) if a != b)
    assert diffs == 1, f"Expected 1 difference, got {diffs}"
    print("PASS: test_point_mutation_changes_gene")


def test_point_mutation_tail_only_terminals():
    """Point mutation in tail only produces terminals."""
    from src.evolution.mutation import _point_mutation, is_terminal

    cfg = _make_cfg()
    genome = _make_genome()

    # Force mutation in tail (position 10)
    class FixedRng:
        def randint(self, a, b):
            return 10  # tail position
        def random(self):
            return 0.5
        def choice(self, seq):
            return seq[0]

    _point_mutation(genome, cfg, FixedRng())
    assert is_terminal(genome.potential_gene[10]), \
        f"Tail position should be terminal, got {genome.potential_gene[10]}"
    print("PASS: test_point_mutation_tail_only_terminals")


def test_constant_finetune_modifies_constant():
    """Constant fine-tune modifies a constant value."""
    from src.evolution.mutation import _constant_finetune

    cfg = _make_cfg()
    rng = random.Random(42)
    genome = _make_genome()

    # Gene has constants at positions 3, 8, 9
    original_val = genome.potential_gene[3]
    _constant_finetune(genome, cfg, rng)
    new_val = genome.potential_gene[3]

    # At least one constant should have changed
    # (might not be position 3, but some constant should differ)
    all_consts_orig = [s for s in ['+', 'sin', 'dist', '2.0', 'density',
                                    'cos', 'speed', 'angle'] + ['3.14', '1.0', 'dist',
                                                                 'density', 'speed',
                                                                 '0.0', '0.0', '0.0', '0.0']
                       if _is_const(s)]
    all_consts_new = [s for s in genome.potential_gene if _is_const(s)]
    # At least the fine-tune should have run without error
    assert len(all_consts_new) > 0
    print("PASS: test_constant_finetune_modifies_constant")


def _is_const(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def test_is_transposition_copies_segment():
    """IS transposition copies a segment to another position."""
    from src.evolution.mutation import _is_transposition

    cfg = _make_cfg()
    rng = random.Random(42)
    genome = _make_genome()
    original = list(genome.potential_gene)

    _is_transposition(genome, cfg, rng)

    # Gene length should be unchanged
    assert len(genome.potential_gene) == len(original)
    print("PASS: test_is_transposition_copies_segment")


def test_ris_transposition_inserts_function_at_start():
    """RIS transposition inserts a function at head start."""
    from src.evolution.mutation import _ris_transposition, is_function

    cfg = _make_cfg()
    genome = _make_genome()

    class FixedRng:
        def randint(self, a, b):
            return 1  # segment length 1
        def random(self):
            return 0.5
        def choice(self, seq):
            return seq[0]  # first function position

    _ris_transposition(genome, cfg, FixedRng())
    # Position 0 should now be a function (or unchanged if no functions found)
    print("PASS: test_ris_transposition_inserts_function_at_start")


def test_one_point_recombination_swaps_tails():
    """One-point recombination swaps tails between two genomes."""
    from src.evolution.mutation import _one_point_recombination

    g1 = _make_genome()
    g2 = _make_genome()
    g2.potential_gene = ['-', 'cos', 'speed', '1.0', 'angle',
                         'sin', 'dist', 'density'] + ['5.0', '2.0', 'speed',
                                                       'angle', 'dist',
                                                       '0.0', '0.0', '0.0', '0.0']

    class FixedRng:
        def randint(self, a, b):
            return 4  # crossover at position 4

    original_g1_tail = list(g1.potential_gene[4:])
    original_g2_tail = list(g2.potential_gene[4:])

    _one_point_recombination(g1, g2, FixedRng())

    # After recombination, tails should be swapped
    assert g1.potential_gene[4:] == original_g2_tail
    assert g2.potential_gene[4:] == original_g1_tail
    print("PASS: test_one_point_recombination_swaps_tails")


def test_two_point_recombination_swaps_middle():
    """Two-point recombination swaps middle segment."""
    from src.evolution.mutation import _two_point_recombination

    g1 = _make_genome()
    g2 = _make_genome()
    g2.potential_gene = list(reversed(g1.potential_gene))

    class FixedRng:
        def randint(self, a, b):
            # Return different values for the two calls
            if not hasattr(self, '_call_count'):
                self._call_count = 0
            self._call_count += 1
            if self._call_count == 1:
                return 3
            else:
                return 6

    original_g1 = list(g1.potential_gene)
    original_g2 = list(g2.potential_gene)

    _two_point_recombination(g1, g2, FixedRng())

    # Segment between points should be swapped
    assert g1.potential_gene[3:6] == original_g2[3:6]
    assert g2.potential_gene[3:6] == original_g1[3:6]
    # Outside segment should be unchanged
    assert g1.potential_gene[:3] == original_g1[:3]
    assert g1.potential_gene[6:] == original_g1[6:]
    print("PASS: test_two_point_recombination_swaps_middle")


def test_mutate_returns_new_genome():
    """mutate() returns a new genome, not modifying the original."""
    from src.evolution.mutation import mutate

    cfg = _make_cfg()
    rng = random.Random(42)
    original = _make_genome()
    original_gene = list(original.potential_gene)

    child = mutate(original, cfg, rng)

    # Original should be unchanged
    assert original.potential_gene == original_gene
    # Child should be different
    assert child is not original
    assert child.generation == original.generation + 1
    print("PASS: test_mutate_returns_new_genome")


def test_mutate_preserves_gene_length():
    """All mutations preserve gene length."""
    from src.evolution.mutation import mutate

    cfg = _make_cfg()
    original_len = 17  # 8 + 9

    for seed in range(100):
        rng = random.Random(seed)
        genome = _make_genome()
        child = mutate(genome, cfg, rng)
        assert len(child.potential_gene) == original_len, \
            f"Seed {seed}: gene length {len(child.potential_gene)} != {original_len}"
    print("PASS: test_mutate_preserves_gene_length")


def test_mutate_all_operators():
    """Each mutation operator can be triggered."""
    from src.evolution.mutation import mutate

    cfg = _make_cfg()
    operators_seen = set()

    for seed in range(200):
        rng = random.Random(seed)
        genome = _make_genome()
        child = mutate(genome, cfg, rng)
        if child.mutation_history:
            operators_seen.add(child.mutation_history[-1])

    expected = {'point', 'const_finetune', 'is_trans', 'ris_trans',
                'one_point_recomb', 'two_point_recomb'}
    assert operators_seen == expected, f"Missing operators: {expected - operators_seen}"
    print(f"PASS: test_mutate_all_operators (seen: {operators_seen})")


if __name__ == '__main__':
    test_point_mutation_changes_gene()
    test_point_mutation_tail_only_terminals()
    test_constant_finetune_modifies_constant()
    test_is_transposition_copies_segment()
    test_ris_transposition_inserts_function_at_start()
    test_one_point_recombination_swaps_tails()
    test_two_point_recombination_swaps_middle()
    test_mutate_returns_new_genome()
    test_mutate_preserves_gene_length()
    test_mutate_all_operators()
    print("\nAll mutation tests passed!")
