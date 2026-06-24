"""
Monitoring module tests.
Run with: python -m pytest tests/test_monitoring.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_check_gpu_temp():
    """GPU temperature check returns valid result."""
    from src.monitoring.monitor import check_gpu_temp

    result = check_gpu_temp()
    assert 'temp' in result
    assert 'ok' in result
    assert 'throttle' in result
    print(f"PASS: test_check_gpu_temp → temp={result['temp']}°C, ok={result['ok']}")


def test_check_disk_space():
    """Disk space check returns valid result."""
    from src.monitoring.monitor import check_disk_space

    result = check_disk_space()
    assert 'free_gb' in result
    assert 'ok' in result
    assert 'warning' in result
    print(f"PASS: test_check_disk_space → free={result['free_gb']}GB, ok={result['ok']}")


def test_check_staleness_not_stale():
    """Not stale when below limit."""
    from src.monitoring.monitor import check_staleness

    result = check_staleness(5, stale_limit=10)
    assert result['stale'] == False
    assert result['count'] == 5
    print(f"PASS: test_check_staleness_not_stale → count={result['count']}")


def test_check_staleness_stale():
    """Stale when at or above limit."""
    from src.monitoring.monitor import check_staleness

    result = check_staleness(10, stale_limit=10)
    assert result['stale'] == True
    assert result['action'] is not None
    print(f"PASS: test_check_staleness_stale → action={result['action']}")


def test_evolution_monitor_init():
    """EvolutionMonitor initializes correctly."""
    from src.monitoring.monitor import EvolutionMonitor

    cfg = {
        'monitoring': {
            'gpu_temp_threshold': 85,
            'disk_min_gb': 5,
            'stale_generations': 10,
        }
    }

    monitor = EvolutionMonitor(cfg)
    assert monitor.gpu_threshold == 85
    assert monitor.disk_min_gb == 5
    assert monitor.stale_limit == 10
    assert monitor.novelty_streak == 0
    print("PASS: test_evolution_monitor_init")


def test_evolution_monitor_novelty_update():
    """Novelty streak updates correctly."""
    from src.monitoring.monitor import EvolutionMonitor

    cfg = {'monitoring': {'stale_generations': 5}}
    monitor = EvolutionMonitor(cfg)

    # No novelty
    monitor.update_novelty(False)
    monitor.update_novelty(False)
    monitor.update_novelty(False)
    assert monitor.novelty_streak == 3

    # Found novelty
    monitor.update_novelty(True)
    assert monitor.novelty_streak == 0

    print("PASS: test_evolution_monitor_novelty_update")


def test_evolution_monitor_check_all():
    """Check all returns valid results."""
    from src.monitoring.monitor import EvolutionMonitor

    cfg = {
        'monitoring': {
            'gpu_temp_threshold': 85,
            'disk_min_gb': 5,
            'stale_generations': 10,
        }
    }

    monitor = EvolutionMonitor(cfg)
    results = monitor.check_all()

    assert 'gpu' in results
    assert 'disk' in results
    assert 'staleness' in results
    assert 'alerts' in results
    assert 'ok' in results

    print(f"PASS: test_evolution_monitor_check_all → ok={results['ok']}, alerts={len(results['alerts'])}")


def test_log_generation_stats():
    """Generation stats logging works."""
    from src.monitoring.monitor import log_generation_stats
    import logging

    # Create a string handler to capture log output
    handler = logging.StreamHandler()
    logger = logging.getLogger('test_monitoring')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    stats = {
        'best_fitness': 0.85,
        'avg_fitness': 0.65,
        'novel_count': 2,
        'dead_count': 3,
        'grid_filled': 100,
        'archive_size': 50,
        'elapsed_seconds': 3600,
    }

    # Should not raise
    log_generation_stats(42, stats, logger)
    print("PASS: test_log_generation_stats")


if __name__ == '__main__':
    test_check_gpu_temp()
    test_check_disk_space()
    test_check_staleness_not_stale()
    test_check_staleness_stale()
    test_evolution_monitor_init()
    test_evolution_monitor_novelty_update()
    test_evolution_monitor_check_all()
    test_log_generation_stats()
    print("\nAll monitoring tests passed!")
