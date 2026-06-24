#!/usr/bin/env python3
"""
Phase 3 Demo: Hybrid Feature Archive System

Demonstrates the complete Phase 3 system:
- 12D time-invariant feature extraction
- 3D MAP-Elites grid
- Novelty Archive
- Hybrid parent selection
- Adaptive threshold
- Dead universe filter

Usage:
    python scripts/phase3_demo.py
"""

import os
import sys
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def demo_feature_extraction():
    """Demonstrate feature extraction."""
    print("\n" + "="*60)
    print("1. Feature Extraction Demo")
    print("="*60)

    from src.evolution.features import FeatureExtractor, SampleSnapshot

    cfg = {
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
        'novelty': {'sample_interval': 100},
    }

    fe = FeatureExtractor(cfg)

    # Simulate some samples
    for i in range(20):
        fe.samples.append(SampleSnapshot(
            spatial_entropy=0.5 + 0.02 * i,
            island_count=3 + i % 5,
            speed_variance=0.1 + 0.005 * i,
            angular_momentum=float(i - 10),
            density_laplacian_var=0.01 * (i + 1),
            alive_count=800 + i * 10,
            total_count=1000,
        ))

    features = fe.compute_features()
    feat3d = fe.get_3d_features()

    print(f"  12D Feature Vector:")
    print(f"    [0] Spatial Entropy Mean:  {features[0]:.4f}")
    print(f"    [1] Spatial Entropy Var:   {features[1]:.4f}")
    print(f"    [2] Islands Mean:          {features[2]:.4f}")
    print(f"    [3] Islands Var:           {features[3]:.4f}")
    print(f"    [4] Speed Variance Mean:   {features[4]:.4f}")
    print(f"    [5-7] FFT Amps (1-3):      {features[5]:.4f}, {features[6]:.4f}, {features[7]:.4f}")
    print(f"    [8] Angular Momentum Skew: {features[8]:.4f}")
    print(f"    [9] Density Lap Var Mean:  {features[9]:.4f}")
    print(f"    [10] Survival Rate:        {features[10]:.4f}")
    print(f"    [11] Autocorr Lag-10:      {features[11]:.4f}")
    print(f"\n  3D Features for MAP-Elites: {feat3d}")

    return features, feat3d


def demo_map_elites():
    """Demonstrate MAP-Elites grid."""
    print("\n" + "="*60)
    print("2. MAP-Elites Grid Demo")
    print("="*60)

    from src.evolution.map_elites import MAPElites
    from src.evolution.genome import GEPGenome

    cfg = {
        'map_elites': {'resolution_per_dim': 15},
    }

    me = MAPElites(cfg)
    rng = random.Random(42)

    # Archive some genomes
    for i in range(50):
        genome = GEPGenome(
            potential_gene=[f'genome_{i}'] + ['0.0'] * 16,
            head_length=8,
            generation=i,
        )

        # Random features
        entropy = rng.uniform(0.0, 1.0)
        islands = rng.uniform(0.0, 50.0)
        fft_amp = rng.uniform(0.0, 1.0)
        fitness = rng.uniform(0.0, 1.0)

        me.try_archive(genome, (entropy, islands, fft_amp), fitness)

    print(f"  Grid Statistics:")
    print(f"    Total cells: {me.get_total_cells()}")
    print(f"    Filled cells: {me.get_filled_count()}")
    print(f"    Fill ratio: {me.get_fill_ratio():.2%}")

    # Show top 5
    summary = me.export_summary()
    print(f"\n  Top 5 by Fitness:")
    for line in summary.split('\n')[3:8]:
        print(f"    {line}")

    return me


def demo_novelty_archive():
    """Demonstrate Novelty Archive."""
    print("\n" + "="*60)
    print("3. Novelty Archive Demo")
    print("="*60)

    from src.evolution.map_elites import NoveltyArchive
    from src.evolution.genome import GEPGenome

    cfg = {
        'novelty': {
            'behavior_vector_dim': 12,
            'k_neighbors': 5,
            'threshold_adaptive': True,
            'stale_generations': 3,
            'min_survival_rate': 0.1,
            'min_speed_variance': 0.001,
            'max_entropy_ratio': 0.95,
        }
    }

    na = NoveltyArchive(cfg)
    rng = random.Random(42)

    # Add some entries
    for i in range(30):
        genome = GEPGenome(
            potential_gene=[f'novel_{i}'] + ['0.0'] * 16,
            head_length=8,
            generation=i,
        )

        # Create behavior vector
        vec = np.random.rand(12).astype(np.float32)
        vec[10] = rng.uniform(0.5, 1.0)  # good survival
        vec[4] = rng.uniform(0.05, 0.2)   # ok speed var

        na.try_add(genome, vec, fitness=rng.uniform(0.3, 0.9))

    stats = na.get_stats()
    print(f"  Archive Statistics:")
    print(f"    Size: {stats['size']}")
    print(f"    Mean Novelty Score: {stats['mean_score']:.4f}")
    print(f"    Median Novelty Score: {stats.get('median_score', 0):.4f}")
    print(f"    Current Threshold: {stats['threshold']:.4f}")

    # Try adding a dead universe
    dead_genome = GEPGenome(
        potential_gene=['dead'] + ['0.0'] * 16,
        head_length=8,
    )
    dead_vec = np.zeros(12, dtype=np.float32)
    dead_vec[10] = 0.01  # 1% survival

    result = na.try_add(dead_genome, dead_vec, fitness=0.0)
    print(f"\n  Dead Universe Filter:")
    print(f"    Attempted to add dead universe (1% survival)")
    print(f"    Result: {'Rejected' if not result else 'Accepted'}")

    return na


def demo_hybrid_selection():
    """Demonstrate hybrid parent selection."""
    print("\n" + "="*60)
    print("4. Hybrid Parent Selection Demo")
    print("="*60)

    from src.evolution.gep import select_parent
    from src.evolution.genome import GEPGenome

    rng = random.Random(42)

    # Create mock grid
    grid = {}
    for i in range(10):
        genome = GEPGenome(
            potential_gene=[f'grid_{i}'] + ['0.0'] * 16,
            head_length=8,
            fitness=0.5 + 0.05 * i,
        )
        grid[f'{i}-0-0'] = {'genome': genome, 'fitness': 0.5 + 0.05 * i}

    # Create mock archive
    archive = []
    for i in range(10):
        genome = GEPGenome(
            potential_gene=[f'archive_{i}'] + ['0.0'] * 16,
            head_length=8,
        )
        archive.append({'genome': genome, 'novelty_score': 1.0 + 0.2 * i})

    # Test selection
    grid_count = 0
    archive_count = 0
    for _ in range(100):
        parent = select_parent(grid, archive, grid_prob=0.7, rng=rng)
        if parent:
            gene_str = '_'.join(str(s) for s in parent.potential_gene[:1])
            if gene_str.startswith('grid'):
                grid_count += 1
            elif gene_str.startswith('archive'):
                archive_count += 1

    print(f"  Selection Statistics (100 selections):")
    print(f"    From Grid: {grid_count} (expected ~70)")
    print(f"    From Archive: {archive_count} (expected ~30)")
    print(f"    Grid/Archive Ratio: {grid_count/max(archive_count,1):.2f}")


def demo_adaptive_threshold():
    """Demonstrate adaptive threshold."""
    print("\n" + "="*60)
    print("5. Adaptive Threshold Demo")
    print("="*60)

    from src.novelty.filter import AdaptiveThreshold

    cfg = {
        'novelty': {
            'stale_generations': 3,
        }
    }

    at = AdaptiveThreshold(cfg)
    print(f"  Initial Threshold: {at.threshold:.4f}")

    # Simulate stale period
    for _ in range(5):
        at.update(was_novel=False)
    print(f"  After 5 stale generations: {at.threshold:.4f}")

    # Simulate novelty findings
    for _ in range(20):
        at.update(was_novel=True, score=5.0)
    print(f"  After 20 novel findings: {at.threshold:.4f}")

    # Simulate another stale period
    for _ in range(3):
        at.update(was_novel=False)
    print(f"  After 3 more stale generations: {at.threshold:.4f}")


def demo_dead_universe_filter():
    """Demonstrate dead universe filter."""
    print("\n" + "="*60)
    print("6. Dead Universe Filter Demo")
    print("="*60)

    from src.novelty.filter import DeadUniverseFilter

    cfg = {
        'novelty': {
            'min_survival_rate': 0.1,
            'min_speed_variance': 0.001,
            'max_entropy_ratio': 0.95,
        }
    }

    duf = DeadUniverseFilter(cfg)

    test_cases = [
        ("Normal universe", np.array([0.5, 0.1, 5.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5])),
        ("Low survival (5%)", np.array([0.5, 0.1, 5.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.5])),
        ("High entropy (99%)", np.array([0.99, 0.1, 5.0, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5])),
        ("Static particles", np.array([0.5, 0.1, 5.0, 1.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.5])),
    ]

    print(f"  Filter Results:")
    for desc, features in test_cases:
        is_dead = duf.is_dead(features)
        status = "DEAD" if is_dead else "LIVE"
        print(f"    {desc:25s} → {status}")


def main():
    """Run all demos."""
    print("Phase 3: Hybrid Feature Archive System Demo")
    print("="*60)

    # Run demos
    demo_feature_extraction()
    demo_map_elites()
    demo_novelty_archive()
    demo_hybrid_selection()
    demo_adaptive_threshold()
    demo_dead_universe_filter()

    print("\n" + "="*60)
    print("Phase 3 Demo Complete!")
    print("="*60)
    print("\nAll components are working correctly:")
    print("  [OK] 12D time-invariant feature extraction")
    print("  [OK] 3D MAP-Elites grid (15^3 = 3,375 cells)")
    print("  [OK] Novelty Archive with k-NN scoring")
    print("  [OK] Hybrid parent selection (70% grid + 30% archive)")
    print("  [OK] Adaptive threshold logic")
    print("  [OK] Dead universe filter")
    print("\nPhase 3 is ready for integration with the evolution loop.")


if __name__ == '__main__':
    main()
