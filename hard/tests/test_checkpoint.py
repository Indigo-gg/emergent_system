"""
Checkpoint module tests.
Run with: python -m pytest tests/test_checkpoint.py -v
"""

import os
import sys
import tempfile
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_save_load_checkpoint():
    """Checkpoint saves and loads correctly."""
    from src.storage.checkpoint import save_checkpoint, load_checkpoint
    from src.evolution.map_elites import MAPElites, NoveltyArchive
    from src.evolution.genome import GEPGenome

    cfg = {
        'map_elites': {'resolution_per_dim': 15},
        'novelty': {
            'k_neighbors': 5,
            'threshold_adaptive': True,
            'stale_generations': 3,
            'min_survival_rate': 0.1,
            'min_speed_variance': 0.001,
            'max_entropy_ratio': 0.95,
        },
    }

    # Create mock state
    me = MAPElites(cfg)
    na = NoveltyArchive(cfg)

    # Add some entries
    genome = GEPGenome(
        potential_gene=['sin', 'dist'] + ['0.0'] * 15,
        head_length=8,
        generation=42,
        random_seed=123,
    )
    me.try_archive(genome, (0.5, 5.0, 0.1), 0.8)

    population = [genome]
    rng_state = {
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state(),
    }

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name

    try:
        # Save
        save_checkpoint(path, 42, me, na, population, rng_state, cfg)
        assert os.path.exists(path)

        # Load
        checkpoint = load_checkpoint(path)
        assert checkpoint['generation'] == 42
        assert 'map_elites_grid' in checkpoint
        assert 'novelty_archive' in checkpoint
        assert 'population' in checkpoint
        assert 'rng_state' in checkpoint

        print(f"PASS: test_save_load_checkpoint → gen={checkpoint['generation']}")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_restore_from_checkpoint():
    """Checkpoint restores state correctly."""
    from src.storage.checkpoint import save_checkpoint, load_checkpoint, restore_from_checkpoint
    from src.evolution.map_elites import MAPElites, NoveltyArchive
    from src.evolution.genome import GEPGenome

    cfg = {
        'map_elites': {'resolution_per_dim': 15},
        'novelty': {
            'k_neighbors': 5,
            'threshold_adaptive': True,
            'stale_generations': 3,
            'min_survival_rate': 0.1,
            'min_speed_variance': 0.001,
            'max_entropy_ratio': 0.95,
        },
    }

    # Create and populate
    me1 = MAPElites(cfg)
    na1 = NoveltyArchive(cfg)

    for i in range(5):
        genome = GEPGenome(
            potential_gene=[f'g{i}'] + ['0.0'] * 16,
            head_length=8,
            generation=i,
            random_seed=i * 10,
        )
        me1.try_archive(genome, (0.1 * i, float(i), 0.01 * i), 0.5 + 0.1 * i)

    rng_state = {'python_state': random.getstate(), 'numpy_state': np.random.get_state()}

    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        path = f.name

    try:
        save_checkpoint(path, 100, me1, na1, [], rng_state, cfg)

        # Restore to new instances
        me2 = MAPElites(cfg)
        na2 = NoveltyArchive(cfg)

        checkpoint = load_checkpoint(path)
        restore_from_checkpoint(checkpoint, me2, na2)

        assert me2.get_filled_count() == me1.get_filled_count()
        print(f"PASS: test_restore_from_checkpoint → grid={me2.get_filled_count()}")
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_list_checkpoints():
    """List checkpoints in directory."""
    from src.storage.checkpoint import save_checkpoint, list_checkpoints
    from src.evolution.map_elites import MAPElites, NoveltyArchive

    cfg = {
        'map_elites': {'resolution_per_dim': 15},
        'novelty': {'k_neighbors': 5, 'threshold_adaptive': True, 'stale_generations': 3,
                    'min_survival_rate': 0.1, 'min_speed_variance': 0.001, 'max_entropy_ratio': 0.95},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        me = MAPElites(cfg)
        na = NoveltyArchive(cfg)
        rng_state = {'python_state': random.getstate(), 'numpy_state': np.random.get_state()}

        # Save multiple checkpoints
        for gen in [0, 50, 100]:
            path = os.path.join(tmpdir, f'checkpoint_{gen:06d}.pkl')
            save_checkpoint(path, gen, me, na, [], rng_state, cfg)

        # List
        checkpoints = list_checkpoints(tmpdir)
        assert len(checkpoints) == 3
        assert checkpoints[0][1] == 0  # First generation
        assert checkpoints[-1][1] == 100  # Last generation

        print(f"PASS: test_list_checkpoints → {len(checkpoints)} checkpoints")


def test_cleanup_old_checkpoints():
    """Old checkpoints are cleaned up."""
    from src.storage.checkpoint import save_checkpoint, list_checkpoints, cleanup_old_checkpoints
    from src.evolution.map_elites import MAPElites, NoveltyArchive

    cfg = {
        'map_elites': {'resolution_per_dim': 15},
        'novelty': {'k_neighbors': 5, 'threshold_adaptive': True, 'stale_generations': 3,
                    'min_survival_rate': 0.1, 'min_speed_variance': 0.001, 'max_entropy_ratio': 0.95},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        me = MAPElites(cfg)
        na = NoveltyArchive(cfg)
        rng_state = {'python_state': random.getstate(), 'numpy_state': np.random.get_state()}

        # Save 10 checkpoints
        for gen in range(0, 100, 10):
            path = os.path.join(tmpdir, f'checkpoint_{gen:06d}.pkl')
            save_checkpoint(path, gen, me, na, [], rng_state, cfg)

        assert len(list_checkpoints(tmpdir)) == 10

        # Keep only 3
        cleanup_old_checkpoints(tmpdir, max_keep=3)
        remaining = list_checkpoints(tmpdir)

        assert len(remaining) == 3
        assert remaining[0][1] == 70  # Oldest kept
        assert remaining[-1][1] == 90  # Newest

        print(f"PASS: test_cleanup_old_checkpoints → kept={len(remaining)}")


if __name__ == '__main__':
    test_save_load_checkpoint()
    test_restore_from_checkpoint()
    test_list_checkpoints()
    test_cleanup_old_checkpoints()
    print("\nAll checkpoint tests passed!")
