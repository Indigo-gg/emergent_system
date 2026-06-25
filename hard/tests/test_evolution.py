"""
Evolution integration tests.
Run with: python -m pytest tests/test_evolution.py -v
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_random_genome_compiles():
    """Random genome can be compiled to bytecode."""
    from src.evolution.genome import random_genome

    cfg = {
        'gep': {
            'head_length': 8,
            'constant_range': [-5.0, 5.0],
            'bytecode_length': 128,
        }
    }
    rng = random.Random(42)

    for i in range(20):
        genome = random_genome(cfg, rng)
        try:
            genome.compile(128)
            assert len(genome._potential_bytecode) == 128
        except Exception as e:
            assert False, f"Genome {i} failed to compile: {e}"

    print("PASS: test_random_genome_compiles")


def test_random_genome_produces_valid_formula():
    """Random genome produces a non-empty formula string."""
    from src.evolution.genome import random_genome

    cfg = {
        'gep': {
            'head_length': 8,
            'constant_range': [-5.0, 5.0],
        }
    }
    rng = random.Random(42)

    for i in range(10):
        genome = random_genome(cfg, rng)
        formula = genome.to_formula()
        assert isinstance(formula, str)
        assert len(formula) > 0

    print("PASS: test_random_genome_produces_valid_formula")


def test_mutated_genome_still_compiles():
    """Mutated genomes can still be compiled."""
    from src.evolution.genome import random_genome
    from src.evolution.mutation import mutate

    cfg = {
        'gep': {
            'head_length': 8,
            'constant_range': [-5.0, 5.0],
            'bytecode_length': 128,
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

    rng = random.Random(42)
    parent = random_genome(cfg, rng)

    for i in range(50):
        child = mutate(parent, cfg, rng)
        try:
            child.compile(128)
            assert len(child._potential_bytecode) == 128
        except Exception as e:
            assert False, f"Mutation {i} produced uncompileable genome: {e}"
        parent = child  # chain mutations

    print("PASS: test_mutated_genome_still_compiles")


def test_evolution_loop_basic():
    """Basic evolution loop runs without errors."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)

    from src.evolution.genome import random_genome
    from src.evolution.mutation import mutate
    from src.evolution.gep import evaluate_fitness
    from src.simulation.particles import ParticleSystem
    from src.simulation.spatial_hash import SpatialHash
    from src.simulation.integrator import Integrator
    from src.simulation.step import SimulationStep

    cfg = {
        'experiment': {'name': 'test_evolution', 'seed': 42},
        'simulation': {
            'num_particles': 100,
            'particle_state_dim': 4,
            'dt': 0.01,
            'damping_gamma': 0.1,
            'bucket_max': 128,
            'steps_per_eval': 500,
        },
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
        'gep': {
            'head_length': 8,
            'constant_range': [-5.0, 5.0],
            'bytecode_length': 128,
            'vm_stack_depth': 16,
        },
        'evolution': {
            'population_size': 3,
            'parsimony_pressure': 0.001,
            'mutation_rates': {
                'point_mutation': 0.30,
                'constant_finetune': 0.20,
                'is_transposition': 0.15,
                'ris_transposition': 0.10,
                'one_point_recombination': 0.15,
                'two_point_recombination': 0.10,
            }
        },
        'novelty': {
            'sample_interval': 200,
            'min_survival_rate': 0.01,
            'min_speed_variance': 0.0001,
        },
        'safety': {
            'max_speed': 5.0,
            'max_force': 10.0,
            'max_displacement_ratio': 0.5,
        },
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
    }

    # Setup simulation components
    from src.simulation.environment import EnvironmentLayer
    particles = ParticleSystem(cfg)
    spatial_hash = SpatialHash(cfg)
    integrator = Integrator(cfg)
    sim_step = SimulationStep(spatial_hash, integrator, cfg)
    environment = EnvironmentLayer(cfg)

    sim_components = {
        'particles': particles,
        'spatial_hash': spatial_hash,
        'integrator': integrator,
        'step': sim_step,
        'environment': environment,
    }

    rng = random.Random(42)

    # Run 3 generations
    best_fitness = 0.0
    for gen in range(3):
        parent = random_genome(cfg, rng)
        child = mutate(parent, cfg, rng)

        fitness, _, _ = evaluate_fitness(child, sim_components, cfg, environment=environment)
        assert fitness >= 0.0, f"Negative fitness: {fitness}"
        best_fitness = max(best_fitness, fitness)
        print(f"  gen={gen} fitness={fitness:.4f} formula={child.to_formula()[:50]}")

    print(f"PASS: test_evolution_loop_basic (best_fitness={best_fitness:.4f})")


def test_select_parent():
    """Parent selection works with grid and archive."""
    from src.evolution.gep import select_parent
    from src.evolution.genome import GEPGenome

    # Create mock grid
    genome1 = GEPGenome(potential_gene=['dist'] + ['0.0'] * 16, head_length=8, fitness=0.8)
    genome2 = GEPGenome(potential_gene=['density'] + ['0.0'] * 16, head_length=8, fitness=0.5)

    grid = {
        '0-0-0': {'genome': genome1, 'fitness': 0.8},
        '1-0-0': {'genome': genome2, 'fitness': 0.5},
    }
    archive = [{'genome': genome1, 'novelty_score': 1.5}]

    rng = random.Random(42)

    # Select from grid (high probability)
    for _ in range(10):
        parent = select_parent(grid, archive, grid_prob=0.9, rng=rng)
        assert parent is not None

    # Select from empty grid → should fall back to archive
    parent = select_parent({}, archive, grid_prob=0.7, rng=rng)
    assert parent is not None

    # Select from empty everything → should return None
    parent = select_parent({}, [], grid_prob=0.7, rng=rng)
    assert parent is None

    print("PASS: test_select_parent")


if __name__ == '__main__':
    test_random_genome_compiles()
    test_random_genome_produces_valid_formula()
    test_mutated_genome_still_compiles()
    test_evolution_loop_basic()
    test_select_parent()
    print("\nAll evolution tests passed!")
