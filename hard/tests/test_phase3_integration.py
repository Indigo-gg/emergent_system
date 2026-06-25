"""
Phase 3 Integration Test: Hybrid Feature Archive System

Tests the complete integration of:
- 12D time-invariant feature extraction
- 3D MAP-Elites grid
- Novelty Archive with k-NN scoring
- Hybrid parent selection (70% grid + 30% archive)
- Adaptive threshold logic
- Dead universe filter
- Integration with evolution loop

Run with: python -m pytest tests/test_phase3_integration.py -v
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_full_cfg():
    """Create a full configuration for integration testing."""
    return {
        'experiment': {'name': 'test_phase3', 'seed': 42},
        'simulation': {
            'num_particles': 200,
            'particle_state_dim': 4,
            'dt': 0.01,
            'damping_gamma': 0.1,
            'bucket_max': 128,
            'steps_per_eval': 1000,
        },
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
        'gep': {
            'head_length': 8,
            'constant_range': [-5.0, 5.0],
            'bytecode_length': 128,
            'vm_stack_depth': 16,
        },
        'evolution': {
            'population_size': 5,
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
        'map_elites': {
            'resolution_per_dim': 15,
        },
        'novelty': {
            'behavior_vector_dim': 12,
            'k_neighbors': 5,
            'threshold_adaptive': True,
            'stale_generations': 3,
            'sample_interval': 200,
            'grid_selection_prob': 0.7,
            'min_survival_rate': 0.01,
            'min_speed_variance': 0.0001,
            'max_entropy_ratio': 0.99,
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


def test_feature_extractor_with_simulation():
    """FeatureExtractor works with actual simulation data."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)

    from src.simulation.particles import ParticleSystem
    from src.evolution.features import FeatureExtractor

    cfg = _make_full_cfg()
    particles = ParticleSystem(cfg)
    particles.initialize(42)

    fe = FeatureExtractor(cfg)

    # Collect multiple samples
    for i in range(5):
        # Advance simulation a few steps (simplified)
        fe.sample(particles, cfg)

    features = fe.compute_features()
    feat3d = fe.get_3d_features()

    assert features.shape == (15,), f"Expected 15D, got {features.shape}"
    assert len(feat3d) == 3
    assert not np.any(np.isnan(features))
    print(f"PASS: test_feature_extractor_with_simulation → features={features[:5]}")


def test_map_elites_with_features():
    """MAP-Elites archives genomes based on 3D features."""
    from src.evolution.map_elites import MAPElites
    from src.evolution.genome import GEPGenome

    cfg = _make_full_cfg()
    me = MAPElites(cfg)

    # Archive several genomes with different features
    rng = random.Random(42)
    for i in range(20):
        genome = GEPGenome(
            potential_gene=[f'g{i}'] + ['0.0'] * 16,
            head_length=8,
            generation=i,
        )
        # Vary features across the grid
        entropy = rng.uniform(0.0, 1.0)
        islands = rng.uniform(0.0, 50.0)
        fft_amp = rng.uniform(0.0, 1.0)
        fitness = rng.uniform(0.0, 1.0)

        me.try_archive(genome, (entropy, islands, fft_amp), fitness)

    assert me.get_filled_count() > 0
    assert me.get_fill_ratio() > 0.0

    # Test random selection
    for _ in range(10):
        cell = me.random_non_empty_cell(rng)
        assert cell is not None
        assert 'genome' in cell

    print(f"PASS: test_map_elites_with_features → filled={me.get_filled_count()}")


def test_novelty_archive_with_dead_filter():
    """NoveltyArchive correctly filters dead universes."""
    from src.evolution.map_elites import NoveltyArchive
    from src.evolution.genome import GEPGenome

    cfg = _make_full_cfg()
    na = NoveltyArchive(cfg)
    rng = random.Random(42)

    # Add some live entries
    for i in range(10):
        genome = GEPGenome(
            potential_gene=[f'g{i}'] + ['0.0'] * 16,
            head_length=8,
            generation=i,
        )
        vec = np.zeros(12, dtype=np.float32)
        vec[0] = rng.uniform(0.0, 1.0)  # entropy
        vec[4] = 0.1  # speed variance (above threshold)
        vec[10] = 0.9  # survival rate (above threshold)

        na.try_add(genome, vec, fitness=0.5)

    # Try adding a dead universe
    dead_genome = GEPGenome(
        potential_gene=['dead'] + ['0.0'] * 16,
        head_length=8,
    )
    dead_vec = np.zeros(12, dtype=np.float32)
    dead_vec[10] = 0.01  # 1% survival (below threshold)

    result = na.try_add(dead_genome, dead_vec, fitness=0.0)
    assert result == False  # Should be rejected

    print(f"PASS: test_novelty_archive_with_dead_filter → size={na.size()}")


def test_hybrid_parent_selection():
    """Hybrid selection draws from both grid and archive."""
    from src.evolution.gep import select_parent
    from src.evolution.genome import GEPGenome

    rng = random.Random(42)

    # Create mock grid
    grid = {}
    for i in range(5):
        genome = GEPGenome(
            potential_gene=[f'grid{i}'] + ['0.0'] * 16,
            head_length=8,
            fitness=0.5 + 0.1 * i,
        )
        grid[f'{i}-0-0'] = {'genome': genome, 'fitness': 0.5 + 0.1 * i}

    # Create mock archive
    archive = []
    for i in range(5):
        genome = GEPGenome(
            potential_gene=[f'archive{i}'] + ['0.0'] * 16,
            head_length=8,
        )
        archive.append({'genome': genome, 'novelty_score': 1.0 + 0.5 * i})

    # Test selection with both available
    grid_count = 0
    archive_count = 0
    for _ in range(100):
        parent = select_parent(grid, archive, grid_prob=0.7, rng=rng)
        assert parent is not None
        gene_str = '_'.join(str(s) for s in parent.potential_gene[:1])
        if gene_str.startswith('grid'):
            grid_count += 1
        elif gene_str.startswith('archive'):
            archive_count += 1

    # With grid_prob=0.7, we should see more grid selections
    assert grid_count > archive_count
    print(f"PASS: test_hybrid_parent_selection → grid={grid_count}, archive={archive_count}")


def test_adaptive_threshold_integration():
    """Adaptive threshold adjusts based on novelty findings."""
    from src.novelty.filter import AdaptiveThreshold

    cfg = _make_full_cfg()
    at = AdaptiveThreshold(cfg)
    initial_threshold = at.threshold

    # Simulate a period with no novelty (stale)
    for _ in range(5):
        at.update(was_novel=False)

    stale_threshold = at.threshold
    assert stale_threshold < initial_threshold

    # Simulate finding novelty
    for _ in range(20):
        at.update(was_novel=True, score=5.0)

    novel_threshold = at.threshold
    assert novel_threshold >= stale_threshold

    print(f"PASS: test_adaptive_threshold_integration → "
          f"initial={initial_threshold:.3f}, stale={stale_threshold:.3f}, novel={novel_threshold:.3f}")


def test_dead_universe_filter_integration():
    """Dead universe filter rejects uninteresting simulations."""
    from src.novelty.filter import DeadUniverseFilter

    cfg = _make_full_cfg()
    duf = DeadUniverseFilter(cfg)

    # Test various feature vectors
    # Note: min_survival_rate=0.01, min_speed_variance=0.0001, max_entropy_ratio=0.99
    test_cases = [
        # (features, expected_dead, description)
        (np.array([0.5, 0.1, 5.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5]), False, "normal"),
        (np.array([0.5, 0.1, 5.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.5]), True, "low survival"),
        (np.array([0.995, 0.1, 5.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5]), True, "high entropy"),
        (np.array([0.5, 0.1, 5.0, 1.0, 0.00005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5]), True, "static"),
    ]

    for features, expected_dead, desc in test_cases:
        result = duf.is_dead(features)
        assert result == expected_dead, f"Failed for {desc}: expected dead={expected_dead}, got {result}"

    print("PASS: test_dead_universe_filter_integration")


def test_full_evolution_step():
    """Full evolution step: select parent → mutate → evaluate → archive."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)

    from src.evolution.genome import random_genome
    from src.evolution.mutation import mutate
    from src.evolution.gep import evaluate_fitness
    from src.evolution.features import FeatureExtractor
    from src.evolution.map_elites import MAPElites, NoveltyArchive
    from src.simulation.particles import ParticleSystem
    from src.simulation.spatial_hash import SpatialHash
    from src.simulation.integrator import Integrator
    from src.simulation.step import SimulationStep
    from src.simulation.environment import EnvironmentLayer

    cfg = _make_full_cfg()
    rng = random.Random(42)

    # Initialize simulation components
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
    }

    # Initialize Phase 3 components
    me = MAPElites(cfg)
    na = NoveltyArchive(cfg)
    fe = FeatureExtractor(cfg)

    # Run 3 evolution steps
    for gen in range(3):
        # 1. Create and mutate genome
        parent = random_genome(cfg, rng)
        child = mutate(parent, cfg, rng)

        # 2. Evaluate fitness
        fitness, _, _ = evaluate_fitness(child, sim_components, cfg, environment=environment)

        # 3. Extract features (simplified: use random for testing)
        features_3d = (rng.uniform(0, 1), rng.uniform(0, 50), rng.uniform(0, 1))
        features_15d = np.random.rand(15).astype(np.float32)
        features_15d[10] = 0.9  # good survival
        features_15d[4] = 0.1   # ok speed var

        # 4. Archive to MAP-Elites
        me.try_archive(child, features_3d, fitness, features_15d, child.random_seed)

        # 5. Try adding to Novelty Archive
        na.try_add(child, features_15d, fitness, child.random_seed)

        print(f"  gen={gen} fitness={fitness:.4f} grid={me.get_filled_count()} archive={na.size()}")

    # Verify results
    assert me.get_filled_count() > 0
    print(f"PASS: test_full_evolution_step → grid={me.get_filled_count()}, archive={na.size()}")


def test_grid_and_archive_stats():
    """Grid and archive provide useful statistics."""
    from src.evolution.map_elites import MAPElites, NoveltyArchive
    from src.evolution.genome import GEPGenome

    cfg = _make_full_cfg()
    me = MAPElites(cfg)
    na = NoveltyArchive(cfg)

    rng = random.Random(42)

    # Add some entries
    for i in range(15):
        genome = GEPGenome(
            potential_gene=[f'g{i}'] + ['0.0'] * 16,
            head_length=8,
            generation=i,
        )

        features_3d = (rng.uniform(0, 1), rng.uniform(0, 50), rng.uniform(0, 1))
        fitness = rng.uniform(0, 1)

        me.try_archive(genome, features_3d, fitness)

        features_12d = np.random.rand(12).astype(np.float32)
        features_12d[10] = 0.9
        features_12d[4] = 0.1
        na.try_add(genome, features_12d, fitness)

    # Test grid statistics
    grid_summary = me.export_summary()
    assert 'MAP-Elites' in grid_summary
    assert f'{me.get_filled_count()}' in grid_summary

    # Test archive statistics
    archive_stats = na.get_stats()
    assert archive_stats['size'] > 0
    assert 'threshold' in archive_stats

    # Test formula export
    grid_formulas = me.get_all_formulas()
    archive_formulas = na.get_all_formulas()

    print(f"PASS: test_grid_and_archive_stats → "
          f"grid={me.get_filled_count()}, archive={na.size()}, "
          f"grid_formulas={len(grid_formulas)}, archive_formulas={len(archive_formulas)}")


if __name__ == '__main__':
    test_feature_extractor_with_simulation()
    test_map_elites_with_features()
    test_novelty_archive_with_dead_filter()
    test_hybrid_parent_selection()
    test_adaptive_threshold_integration()
    test_dead_universe_filter_integration()
    test_full_evolution_step()
    test_grid_and_archive_stats()
    print("\nAll Phase 3 integration tests passed!")
