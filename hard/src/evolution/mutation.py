"""
Genetic operators for GEP evolution.

6 operators (selected by probability):
1. Point mutation (30%)      — change one symbol
2. Constant fine-tune (20%)  — add Gaussian noise to a constant
3. IS transposition (15%)    — copy segment to another position
4. RIS transposition (10%)   — copy function-starting segment to head start
5. One-point recombination (15%) — swap tails of two genomes
6. Two-point recombination (10%) — swap middle segments of two genomes
"""

import copy
import random
import math
from typing import Optional

from src.evolution.genome import (
    GEPGenome, FUNCTION_SET, TERMINAL_SET, FUNCTION_ARITY,
    is_function, is_terminal, random_constant
)


def mutate(genome: GEPGenome, cfg: dict, rng: random.Random = None) -> GEPGenome:
    """
    Apply a random mutation operator to a genome.
    Returns a new (mutated) genome.
    """
    if rng is None:
        rng = random.Random()

    rates = cfg['evolution']['mutation_rates']
    child = copy.deepcopy(genome)
    child.invalidate_cache()
    child.generation = genome.generation + 1
    child.parent_id = genome.get_id()

    # Select operator by probability
    r = rng.random()
    cumulative = 0.0

    cumulative += rates.get('point_mutation', 0.30)
    if r < cumulative:
        _point_mutation(child, cfg, rng)
        child.mutation_history.append('point')
        return child

    cumulative += rates.get('constant_finetune', 0.20)
    if r < cumulative:
        _constant_finetune(child, cfg, rng)
        child.mutation_history.append('const_finetune')
        return child

    cumulative += rates.get('is_transposition', 0.15)
    if r < cumulative:
        _is_transposition(child, cfg, rng)
        child.mutation_history.append('is_trans')
        return child

    cumulative += rates.get('ris_transposition', 0.10)
    if r < cumulative:
        _ris_transposition(child, cfg, rng)
        child.mutation_history.append('ris_trans')
        return child

    cumulative += rates.get('one_point_recombination', 0.15)
    if r < cumulative:
        # One-point recombination needs a second parent — use self for now
        # (will be replaced with actual second parent in evolution loop)
        _one_point_recombination(child, genome, rng)
        child.mutation_history.append('one_point_recomb')
        return child

    # Two-point recombination (remaining probability)
    _two_point_recombination(child, genome, rng)
    child.mutation_history.append('two_point_recomb')
    return child


def mutate_crossover(parent1: GEPGenome, parent2: GEPGenome,
                     cfg: dict, rng: random.Random = None) -> GEPGenome:
    """
    Produce a child via crossover of two parents.
    Uses one-point or two-point recombination.
    """
    if rng is None:
        rng = random.Random()

    child = copy.deepcopy(parent1)
    child.invalidate_cache()
    child.generation = max(parent1.generation, parent2.generation) + 1
    child.parent_id = f"{parent1.get_id()}+{parent2.get_id()}"

    if rng.random() < 0.6:  # 60% one-point, 40% two-point
        _one_point_recombination(child, parent2, rng)
        child.mutation_history.append('one_point_recomb')
    else:
        _two_point_recombination(child, parent2, rng)
        child.mutation_history.append('two_point_recomb')

    return child


# ── Operator Implementations ──

def _point_mutation(genome: GEPGenome, cfg: dict, rng: random.Random):
    """Randomly change one symbol in the potential gene."""
    gene = genome.potential_gene
    if not gene:
        return

    pos = rng.randint(0, len(gene) - 1)
    head_length = genome.head_length

    if pos < head_length:
        # Head position: can be function or terminal
        if rng.random() < 0.5:
            gene[pos] = rng.choice(FUNCTION_SET)
        else:
            if rng.random() < 0.3:
                constant_range = cfg['gep'].get('constant_range', [-5.0, 5.0])
                gene[pos] = random_constant(rng, *constant_range)
            else:
                gene[pos] = rng.choice(TERMINAL_SET)
    else:
        # Tail position: terminals only
        if rng.random() < 0.3:
            constant_range = cfg['gep'].get('constant_range', [-5.0, 5.0])
            gene[pos] = random_constant(rng, *constant_range)
        else:
            gene[pos] = rng.choice(TERMINAL_SET)


def _constant_finetune(genome: GEPGenome, cfg: dict, rng: random.Random):
    """Add Gaussian noise to a random constant in the gene."""
    gene = genome.potential_gene

    # Find all constant positions
    const_positions = []
    for i, sym in enumerate(gene):
        try:
            float(sym)
            const_positions.append(i)
        except ValueError:
            continue

    if not const_positions:
        # No constants — add one
        pos = rng.randint(0, len(gene) - 1)
        constant_range = cfg['gep'].get('constant_range', [-5.0, 5.0])
        gene[pos] = random_constant(rng, *constant_range)
        return

    pos = rng.choice(const_positions)
    old_val = float(gene[pos])

    # Add Gaussian noise
    noise = rng.gauss(0, 0.5)
    new_val = old_val + noise

    # Clamp to range
    constant_range = cfg['gep'].get('constant_range', [-5.0, 5.0])
    new_val = max(constant_range[0], min(constant_range[1], new_val))

    gene[pos] = f"{new_val:.4f}"


def _is_transposition(genome: GEPGenome, cfg: dict, rng: random.Random):
    """IS transposition: copy a segment to another position in the head."""
    gene = genome.potential_gene
    head_length = genome.head_length

    if len(gene) < 3:
        return

    # Source: random segment (length 1-3)
    src_len = rng.randint(1, min(3, len(gene) - 1))
    src_start = rng.randint(0, len(gene) - src_len)

    # Destination: somewhere in the head (not position 0)
    if head_length <= 1:
        return
    dst = rng.randint(1, head_length - 1)

    # Extract segment
    segment = gene[src_start:src_start + src_len]

    # Insert at destination (overwrite, don't extend)
    for i, sym in enumerate(segment):
        insert_pos = dst + i
        if insert_pos < len(gene):
            gene[insert_pos] = sym


def _ris_transposition(genome: GEPGenome, cfg: dict, rng: random.Random):
    """RIS transposition: copy a function-starting segment to head start."""
    gene = genome.potential_gene
    head_length = genome.head_length

    if len(gene) < 2:
        return

    # Find positions in head that start with a function
    func_positions = []
    for i in range(min(head_length, len(gene))):
        if is_function(gene[i]):
            func_positions.append(i)

    if not func_positions:
        return

    # Pick a function-starting position
    src_start = rng.choice(func_positions)

    # Segment length 1-3
    src_len = rng.randint(1, min(3, len(gene) - src_start))
    segment = gene[src_start:src_start + src_len]

    # Insert at position 0 (overwrite head start)
    for i, sym in enumerate(segment):
        if i < len(gene):
            gene[i] = sym


def _one_point_recombination(child: GEPGenome, parent2: GEPGenome,
                             rng: random.Random):
    """One-point recombination: swap tails after a random crossover point."""
    gene1 = child.potential_gene
    gene2 = parent2.potential_gene

    if len(gene1) < 2 or len(gene2) < 2:
        return

    # Crossover point (1 to min(len1, len2) - 1)
    max_point = min(len(gene1), len(gene2)) - 1
    if max_point < 1:
        return
    point = rng.randint(1, max_point)

    # Swap tails
    for i in range(point, min(len(gene1), len(gene2))):
        gene1[i], gene2[i] = gene2[i], gene1[i]


def _two_point_recombination(child: GEPGenome, parent2: GEPGenome,
                             rng: random.Random):
    """Two-point recombination: swap segment between two crossover points."""
    gene1 = child.potential_gene
    gene2 = parent2.potential_gene

    min_len = min(len(gene1), len(gene2))
    if min_len < 3:
        return

    # Two crossover points
    pt1 = rng.randint(1, min_len - 2)
    pt2 = rng.randint(pt1 + 1, min_len - 1)

    # Swap segment between pt1 and pt2
    for i in range(pt1, pt2):
        gene1[i], gene2[i] = gene2[i], gene1[i]
