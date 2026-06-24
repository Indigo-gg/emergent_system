"""
Rendering module tests.
Run with: python -m pytest tests/test_rendering.py -v
"""

import os
import sys
import numpy as np
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg():
    return {
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
        'rendering': {'resolution': [64, 64]},
    }


def test_render_density_heatmap():
    """Density heatmap renders without error."""
    from src.rendering.renderer import render_density_heatmap

    cfg = _make_cfg()
    pos_x = np.random.uniform(0, 100, 1000)
    pos_y = np.random.uniform(0, 100, 1000)

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name

    try:
        result = render_density_heatmap(pos_x, pos_y, cfg, output_path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
        print(f"PASS: test_render_density_heatmap → {result}")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_render_trajectory():
    """Trajectory overlay renders without error."""
    from src.rendering.renderer import render_trajectory

    cfg = _make_cfg()

    # Create mock position history
    history = []
    for i in range(50):
        px = np.random.uniform(0, 100, 100)
        py = np.random.uniform(0, 100, 100)
        history.append((px, py))

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name

    try:
        result = render_trajectory(history, cfg, output_path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
        print(f"PASS: test_render_trajectory → {result}")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_render_feature_curves():
    """Feature curves render without error."""
    from src.rendering.renderer import render_feature_curves

    cfg = _make_cfg()

    # Create mock feature timeseries
    timeseries = {
        'speed_variance': [(i, 0.1 + 0.01 * i) for i in range(100)],
        'entropy': [(i, 0.5 + 0.005 * i) for i in range(100)],
        'angular_momentum': [(i, float(i - 50)) for i in range(100)],
    }

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        output_path = f.name

    try:
        result = render_feature_curves(timeseries, cfg, output_path)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
        print(f"PASS: test_render_feature_curves → {result}")
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def test_render_novelty_package():
    """Novelty package renders all components."""
    from src.rendering.renderer import render_novelty_package
    from src.evolution.genome import GEPGenome

    cfg = _make_cfg()

    # Create mock genome
    genome = GEPGenome(
        potential_gene=['sin', 'dist', '0.5'] + ['0.0'] * 14,
        head_length=8,
        generation=42,
        random_seed=123,
    )

    # Create mock data
    features_12d = np.array([0.5, 0.1, 5.0, 1.0, 0.1, 0.05, 0.03, 0.02, 0.5, 0.01, 0.9, 0.7])
    position_history = [(np.random.uniform(0, 100, 50), np.random.uniform(0, 100, 50)) for _ in range(20)]
    feature_timeseries = {'speed_variance': [(i, 0.1) for i in range(20)]}

    with tempfile.TemporaryDirectory() as tmpdir:
        result = render_novelty_package(genome, features_12d, position_history, feature_timeseries, cfg, tmpdir)

        assert 'trajectory_path' in result
        assert 'curve_path' in result
        assert 'summary' in result
        assert os.path.exists(result['trajectory_path'])
        assert 'Generation: 42' in result['summary']
        print(f"PASS: test_render_novelty_package → {list(result.keys())}")


if __name__ == '__main__':
    test_render_density_heatmap()
    test_render_trajectory()
    test_render_feature_curves()
    test_render_novelty_package()
    print("\nAll rendering tests passed!")
