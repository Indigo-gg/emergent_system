"""
Novelty filter tests.
Run with: python -m pytest tests/test_novelty.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg():
    return {
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


def test_novelty_score_inf_for_small_archive():
    """Novelty score is inf when archive has fewer than k entries."""
    from src.novelty.filter import novelty_score

    vec = np.random.rand(12).astype(np.float32)
    archive = np.random.rand(3, 12).astype(np.float32)

    score = novelty_score(vec, archive, k=5)
    assert score == float('inf')
    print("PASS: test_novelty_score_inf_for_small_archive")


def test_novelty_score_computation():
    """Novelty score is average distance to k nearest neighbors."""
    from src.novelty.filter import novelty_score

    # Archive: 10 points at origin
    archive = np.zeros((10, 12), dtype=np.float32)

    # New point far from origin → high novelty
    vec_far = np.ones(12, dtype=np.float32) * 10.0
    score_far = novelty_score(vec_far, archive, k=5)
    assert score_far > 10.0

    # New point at origin → zero novelty
    vec_near = np.zeros(12, dtype=np.float32)
    score_near = novelty_score(vec_near, archive, k=5)
    assert score_near < 0.01

    print(f"PASS: test_novelty_score_computation (far={score_far:.2f}, near={score_near:.4f})")


def test_adaptive_threshold_starts_low():
    """Adaptive threshold starts low to encourage exploration."""
    from src.novelty.filter import AdaptiveThreshold

    cfg = _make_cfg()
    at = AdaptiveThreshold(cfg)
    assert at.threshold < 1.0
    print(f"PASS: test_adaptive_threshold_starts_low → {at.threshold}")


def test_adaptive_threshold_raises_on_novelty():
    """Threshold increases when novelty is found."""
    from src.novelty.filter import AdaptiveThreshold

    cfg = _make_cfg()
    at = AdaptiveThreshold(cfg)
    initial = at.threshold

    # Simulate many novel findings
    for _ in range(20):
        at.update(was_novel=True, score=5.0)

    assert at.threshold >= initial
    print(f"PASS: test_adaptive_threshold_raises_on_novelty ({initial:.3f} → {at.threshold:.3f})")


def test_adaptive_threshold_drops_on_staleness():
    """Threshold decreases after stale_generations without novelty."""
    from src.novelty.filter import AdaptiveThreshold

    cfg = _make_cfg()
    at = AdaptiveThreshold(cfg)
    initial = at.threshold

    # Simulate stale period
    for _ in range(5):
        at.update(was_novel=False)

    assert at.threshold < initial
    print(f"PASS: test_adaptive_threshold_drops_on_staleness ({initial:.3f} → {at.threshold:.3f})")


def test_dead_universe_filter_low_survival():
    """Dead universe: survival rate < 0.1."""
    from src.novelty.filter import DeadUniverseFilter

    cfg = _make_cfg()
    duf = DeadUniverseFilter(cfg)

    features = np.zeros(12, dtype=np.float32)
    features[10] = 0.05  # 5% survival
    assert duf.is_dead(features) == True

    features[10] = 0.5  # 50% survival
    features[0] = 0.5   # entropy
    features[4] = 0.1   # speed var
    assert duf.is_dead(features) == False
    print("PASS: test_dead_universe_filter_low_survival")


def test_dead_universe_filter_high_entropy():
    """Dead universe: entropy > 95% max."""
    from src.novelty.filter import DeadUniverseFilter

    cfg = _make_cfg()
    duf = DeadUniverseFilter(cfg)

    features = np.zeros(12, dtype=np.float32)
    features[0] = 0.99  # very high entropy
    features[10] = 0.5  # ok survival
    features[4] = 0.1   # ok speed var
    assert duf.is_dead(features) == True
    print("PASS: test_dead_universe_filter_high_entropy")


def test_dead_universe_filter_static():
    """Dead universe: all particles static."""
    from src.novelty.filter import DeadUniverseFilter

    cfg = _make_cfg()
    duf = DeadUniverseFilter(cfg)

    features = np.zeros(12, dtype=np.float32)
    features[10] = 0.5   # ok survival
    features[0] = 0.5    # ok entropy
    features[4] = 0.0001 # very low speed var
    assert duf.is_dead(features) == True
    print("PASS: test_dead_universe_filter_static")


def test_archive_stats():
    """Archive statistics computation."""
    from src.novelty.filter import compute_novelty_archive_stats

    archive = [
        {'novelty_score': 5.0},
        {'novelty_score': 3.0},
        {'novelty_score': 1.0},
    ]
    stats = compute_novelty_archive_stats(archive)
    assert stats['size'] == 3
    assert abs(stats['mean_score'] - 3.0) < 0.01
    assert abs(stats['median_score'] - 3.0) < 0.01
    print("PASS: test_archive_stats")


if __name__ == '__main__':
    test_novelty_score_inf_for_small_archive()
    test_novelty_score_computation()
    test_adaptive_threshold_starts_low()
    test_adaptive_threshold_raises_on_novelty()
    test_adaptive_threshold_drops_on_staleness()
    test_dead_universe_filter_low_survival()
    test_dead_universe_filter_high_entropy()
    test_dead_universe_filter_static()
    test_archive_stats()
    print("\nAll novelty tests passed!")
