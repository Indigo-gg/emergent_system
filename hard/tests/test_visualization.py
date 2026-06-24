"""
Visualization module tests.
Run with: python -m pytest tests/test_visualization.py -v
"""

import os
import sys
import tempfile
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg():
    return {
        'experiment': {'name': 'test_viz'},
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


def _populate_grid_and_archive(cfg, n=20):
    """Create populated MAP-Elites and Novelty Archive."""
    from src.evolution.map_elites import MAPElites, NoveltyArchive
    from src.evolution.genome import GEPGenome

    me = MAPElites(cfg)
    na = NoveltyArchive(cfg)
    rng = random.Random(42)

    for i in range(n):
        genome = GEPGenome(
            potential_gene=[f'sin', 'dist', f'{rng.uniform(-2, 2):.2f}'] + ['0.0'] * 14,
            head_length=8,
            generation=i,
            random_seed=rng.randint(0, 10000),
        )

        features_3d = (rng.uniform(0, 1), rng.uniform(0, 50), rng.uniform(0, 1))
        fitness = rng.uniform(0.3, 0.95)

        me.try_archive(genome, features_3d, fitness)

        features_12d = np.random.rand(12).astype(np.float32)
        features_12d[10] = rng.uniform(0.5, 1.0)
        features_12d[4] = rng.uniform(0.05, 0.2)
        na.try_add(genome, features_12d, fitness)

    return me, na


def test_visualize_map_elites_3d():
    """3D MAP-Elites visualization renders correctly."""
    from src.visualization.visualizer import visualize_map_elites_3d

    cfg = _make_cfg()
    me, _ = _populate_grid_and_archive(cfg)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name

    try:
        result = visualize_map_elites_3d(me, cfg, output_path)
        assert result is not None
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
        print(f"PASS: test_visualize_map_elites_3d → {result}")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_visualize_map_elites_slices():
    """2D slice visualization renders correctly."""
    from src.visualization.visualizer import visualize_map_elites_slices

    cfg = _make_cfg()
    me, _ = _populate_grid_and_archive(cfg)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name

    try:
        result = visualize_map_elites_slices(me, cfg, output_path)
        assert result is not None
        assert os.path.exists(result)
        print(f"PASS: test_visualize_map_elites_slices → {result}")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_export_formula_collection():
    """Formula collection exports correctly."""
    from src.visualization.visualizer import export_formula_collection

    cfg = _make_cfg()
    me, na = _populate_grid_and_archive(cfg)

    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        output_path = f.name

    try:
        result = export_formula_collection(me, na, output_path)
        assert os.path.exists(result)

        with open(result, 'r') as f:
            content = f.read()

        assert 'FORMULA COLLECTION' in content
        assert 'MAP-Elites Grid' in content
        assert 'Novelty Archive' in content
        print(f"PASS: test_export_formula_collection → {len(content)} chars")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_browse_novelty_archive():
    """Browse novelty archive returns sorted entries."""
    from src.visualization.visualizer import browse_novelty_archive

    cfg = _make_cfg()
    _, na = _populate_grid_and_archive(cfg)

    entries = browse_novelty_archive(na, top_n=5)
    assert len(entries) <= 5

    # Should be sorted by novelty score (descending)
    if len(entries) > 1:
        for i in range(len(entries) - 1):
            assert entries[i].get('novelty_score', 0) >= entries[i+1].get('novelty_score', 0)

    print(f"PASS: test_browse_novelty_archive → {len(entries)} entries")


def test_generate_report():
    """Report generation creates all files."""
    from src.visualization.visualizer import generate_report

    cfg = _make_cfg()
    me, na = _populate_grid_and_archive(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_report(me, na, cfg, tmpdir)

        assert os.path.isdir(result)

        # Check files exist
        expected_files = ['config_summary.txt', 'map_elites_3d.png',
                          'map_elites_slices.png', 'formula_collection.txt',
                          'statistics.txt', 'top_formulas.txt']

        for fname in expected_files:
            fpath = os.path.join(result, fname)
            assert os.path.exists(fpath), f"Missing: {fname}"

        print(f"PASS: test_generate_report → {result} ({len(expected_files)} files)")


def test_empty_grid_visualization():
    """Empty grid visualization returns None."""
    from src.visualization.visualizer import visualize_map_elites_3d
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)

    result = visualize_map_elites_3d(me, cfg, '/tmp/test.png')
    assert result is None
    print("PASS: test_empty_grid_visualization")


if __name__ == '__main__':
    test_visualize_map_elites_3d()
    test_visualize_map_elites_slices()
    test_export_formula_collection()
    test_browse_novelty_archive()
    test_generate_report()
    test_empty_grid_visualization()
    print("\nAll visualization tests passed!")
