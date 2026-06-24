"""
Feature extraction tests.
Run with: python -m pytest tests/test_features.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg():
    return {
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
        'novelty': {'sample_interval': 100},
    }


def test_feature_extractor_init():
    """FeatureExtractor initializes correctly."""
    from src.evolution.features import FeatureExtractor

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)
    assert len(fe.samples) == 0
    assert fe.sample_interval == 100
    print("PASS: test_feature_extractor_init")


def test_feature_extractor_reset():
    """Reset clears samples."""
    from src.evolution.features import FeatureExtractor, SampleSnapshot

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)
    fe.samples.append(SampleSnapshot(spatial_entropy=0.5))
    assert len(fe.samples) == 1

    fe.reset()
    assert len(fe.samples) == 0
    print("PASS: test_feature_extractor_reset")


def test_compute_features_empty():
    """Empty samples returns zero vector."""
    from src.evolution.features import FeatureExtractor

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)
    features = fe.compute_features()

    assert features.shape == (12,)
    assert np.all(features == 0.0)
    print("PASS: test_compute_features_empty")


def test_compute_features_with_samples():
    """Features computed from sample snapshots."""
    from src.evolution.features import FeatureExtractor, SampleSnapshot

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)

    # Add some mock samples
    for i in range(20):
        fe.samples.append(SampleSnapshot(
            spatial_entropy=0.5 + 0.01 * i,
            island_count=3 + i % 5,
            speed_variance=0.1 + 0.005 * i,
            angular_momentum=float(i - 10),
            density_laplacian_var=0.01 * (i + 1),
            alive_count=800 + i * 10,
            total_count=1000,
        ))

    features = fe.compute_features()

    assert features.shape == (12,)
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))

    # Check some properties
    assert 0.0 <= features[0] <= 1.0  # entropy mean
    assert features[2] > 0  # islands mean > 0
    assert features[4] > 0  # speed variance mean > 0
    assert 0.0 <= features[10] <= 1.0  # survival rate

    print(f"PASS: test_compute_features_with_samples → {features}")


def test_get_3d_features():
    """3D features are first 3 discriminating dimensions."""
    from src.evolution.features import FeatureExtractor, SampleSnapshot

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)

    for i in range(10):
        fe.samples.append(SampleSnapshot(
            spatial_entropy=0.6,
            island_count=5,
            speed_variance=0.2,
            angular_momentum=0.0,
            density_laplacian_var=0.01,
            alive_count=900,
            total_count=1000,
        ))

    feat3d = fe.get_3d_features()
    assert len(feat3d) == 3
    assert abs(feat3d[0] - 0.6) < 0.001  # entropy mean (float32 precision)
    assert abs(feat3d[1] - 5.0) < 0.001  # islands mean
    print(f"PASS: test_get_3d_features → {feat3d}")


def test_spatial_entropy_uniform():
    """Uniform distribution has high entropy."""
    from src.evolution.features import FeatureExtractor

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)

    # Uniform distribution
    px = np.random.uniform(0, 100, 10000)
    py = np.random.uniform(0, 100, 10000)
    entropy = fe._spatial_entropy(px, py, 100.0, 100.0)

    assert entropy > 0.9, f"Expected high entropy for uniform, got {entropy}"
    print(f"PASS: test_spatial_entropy_uniform → {entropy:.4f}")


def test_spatial_entropy_clustered():
    """Clustered distribution has low entropy."""
    from src.evolution.features import FeatureExtractor

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)

    # All at same point
    px = np.full(1000, 50.0)
    py = np.full(1000, 50.0)
    entropy = fe._spatial_entropy(px, py, 100.0, 100.0)

    assert entropy < 0.1, f"Expected low entropy for clustered, got {entropy}"
    print(f"PASS: test_spatial_entropy_clustered → {entropy:.4f}")


def test_fft_top3():
    """FFT extracts top 3 amplitudes."""
    from src.evolution.features import FeatureExtractor

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)

    # Create a signal with known frequencies
    t = np.linspace(0, 10, 200)
    signal = 3.0 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz, amplitude 3
    signal += 1.5 * np.sin(2 * np.pi * 5.0 * t)  # 5 Hz, amplitude 1.5

    top3 = fe._fft_top3(signal)
    assert len(top3) == 3
    assert top3[0] > top3[1]  # strongest > second strongest
    print(f"PASS: test_fft_top3 → {[f'{a:.3f}' for a in top3]}")


def test_skewness():
    """Skewness computation."""
    from src.evolution.features import FeatureExtractor

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)

    # Symmetric distribution → skewness ≈ 0
    symmetric = np.random.normal(0, 1, 1000)
    skew = fe._skewness(symmetric)
    assert abs(skew) < 0.5, f"Expected ~0 skewness for normal, got {skew}"

    # Right-skewed → positive skewness
    right_skewed = np.random.exponential(1, 1000)
    skew_right = fe._skewness(right_skewed)
    assert skew_right > 0, f"Expected positive skewness, got {skew_right}"

    print(f"PASS: test_skewness (symmetric={skew:.3f}, right={skew_right:.3f})")


def test_autocorrelation():
    """Autocorrelation detects periodicity."""
    from src.evolution.features import FeatureExtractor

    cfg = _make_cfg()
    fe = FeatureExtractor(cfg)

    # Periodic signal → high autocorrelation
    t = np.arange(100)
    periodic = np.sin(2 * np.pi * t / 20)  # period 20
    ac = fe._autocorrelation(periodic, lag=20)
    assert ac > 0.8, f"Expected high autocorrelation for periodic, got {ac}"

    # Random signal → low autocorrelation
    random_signal = np.random.normal(0, 1, 100)
    ac_random = fe._autocorrelation(random_signal, lag=10)
    assert abs(ac_random) < 0.5, f"Expected low autocorrelation for random, got {ac_random}"

    print(f"PASS: test_autocorrelation (periodic={ac:.3f}, random={ac_random:.3f})")


def test_sample_with_mock_particles():
    """Sampling from mock particle system works."""
    import taichi as ti
    ti.init(arch=ti.cpu, debug=True)

    from src.simulation.particles import ParticleSystem
    from src.evolution.features import FeatureExtractor

    cfg_full = {
        'simulation': {'num_particles': 500, 'particle_state_dim': 4},
        'world': {'width': 100.0, 'height': 100.0, 'cell_size': 2.0},
        'novelty': {'sample_interval': 100},
    }

    particles = ParticleSystem(cfg_full)
    particles.initialize(42)

    fe = FeatureExtractor(cfg_full)
    fe.sample(particles, cfg_full)

    assert len(fe.samples) == 1
    assert fe.samples[0].alive_count == 500
    assert fe.samples[0].total_count == 500
    print(f"PASS: test_sample_with_mock_particles")


if __name__ == '__main__':
    test_feature_extractor_init()
    test_feature_extractor_reset()
    test_compute_features_empty()
    test_compute_features_with_samples()
    test_get_3d_features()
    test_spatial_entropy_uniform()
    test_spatial_entropy_clustered()
    test_fft_top3()
    test_skewness()
    test_autocorrelation()
    test_sample_with_mock_particles()
    print("\nAll feature tests passed!")
