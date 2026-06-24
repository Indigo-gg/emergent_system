"""
MAP-Elites and Novelty Archive tests.
Run with: python -m pytest tests/test_map_elites.py -v
"""

import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg():
    return {
        'map_elites': {'resolution_per_dim': 15},
        'novelty': {
            'behavior_vector_dim': 12,
            'k_neighbors': 5,
            'threshold_adaptive': True,
            'stale_generations': 3,
            'min_survival_rate': 0.1,
            'min_speed_variance': 0.001,
            'max_entropy_ratio': 0.95,
        },
    }


def _make_genome(formula='dist', gen=0):
    from src.evolution.genome import GEPGenome
    return GEPGenome(
        potential_gene=[formula] + ['0.0'] * 16,
        head_length=8,
        generation=gen,
    )


# ── MAP-Elites Tests ──

def test_map_elites_init():
    """MAP-Elites initializes with empty grid."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)
    assert me.get_filled_count() == 0
    assert me.get_total_cells() == 3375  # 15^3
    assert me.resolution == 15
    print("PASS: test_map_elites_init")


def test_map_elites_archive_first():
    """First genome in a cell is always archived."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)
    genome = _make_genome()

    result = me.try_archive(genome, (0.5, 5.0, 0.1), 0.8)
    assert result == True
    assert me.get_filled_count() == 1
    print("PASS: test_map_elites_archive_first")


def test_map_elites_replace_if_better():
    """Better genome replaces worse in same cell."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)

    g1 = _make_genome('dist')
    g2 = _make_genome('density')

    me.try_archive(g1, (0.5, 5.0, 0.1), 0.5)
    result = me.try_archive(g2, (0.5, 5.0, 0.1), 0.8)

    assert result == True
    assert me.get_filled_count() == 1  # still 1 cell
    print("PASS: test_map_elites_replace_if_better")


def test_map_elites_keep_if_worse():
    """Worse genome does NOT replace better in same cell."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)

    g1 = _make_genome('dist')
    g2 = _make_genome('density')

    me.try_archive(g1, (0.5, 5.0, 0.1), 0.8)
    result = me.try_archive(g2, (0.5, 5.0, 0.1), 0.5)

    assert result == False
    assert me.get_filled_count() == 1
    print("PASS: test_map_elites_keep_if_worse")


def test_map_elites_different_cells():
    """Different features go to different cells."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)

    me.try_archive(_make_genome('a'), (0.1, 1.0, 0.01), 0.5)
    me.try_archive(_make_genome('b'), (0.9, 10.0, 0.5), 0.5)
    me.try_archive(_make_genome('c'), (0.5, 5.0, 0.1), 0.5)

    assert me.get_filled_count() == 3
    print("PASS: test_map_elites_different_cells")


def test_map_elites_random_non_empty():
    """Random selection from non-empty grid."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)

    for i in range(10):
        me.try_archive(_make_genome(f'g{i}'), (0.1 * i, float(i), 0.01 * i), 0.5 + 0.01 * i)

    rng = random.Random(42)
    for _ in range(20):
        cell = me.random_non_empty_cell(rng)
        assert cell is not None
        assert 'genome' in cell

    print("PASS: test_map_elites_random_non_empty")


def test_map_elites_random_empty():
    """Random selection from empty grid returns None."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)
    cell = me.random_non_empty_cell()
    assert cell is None
    print("PASS: test_map_elites_random_empty")


def test_map_elites_fill_ratio():
    """Fill ratio computation."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)
    assert me.get_fill_ratio() == 0.0

    me.try_archive(_make_genome(), (0.5, 5.0, 0.1), 0.5)
    assert me.get_fill_ratio() > 0.0
    print(f"PASS: test_map_elites_fill_ratio → {me.get_fill_ratio():.6f}")


def test_map_elites_export_summary():
    """Export summary produces readable text."""
    from src.evolution.map_elites import MAPElites

    cfg = _make_cfg()
    me = MAPElites(cfg)

    for i in range(5):
        me.try_archive(_make_genome(f'g{i}'), (0.2 * i, float(i+1), 0.05 * i), 0.3 + 0.1 * i)

    summary = me.export_summary()
    assert isinstance(summary, str)
    assert 'MAP-Elites' in summary
    print(f"PASS: test_map_elites_export_summary\n{summary}")


# ── Novelty Archive Tests ──

def test_novelty_archive_init():
    """Archive initializes empty."""
    from src.evolution.map_elites import NoveltyArchive

    cfg = _make_cfg()
    na = NoveltyArchive(cfg)
    assert na.size() == 0
    print("PASS: test_novelty_archive_init")


def test_novelty_archive_first_entry():
    """First entry is always novel (inf score)."""
    from src.evolution.map_elites import NoveltyArchive

    cfg = _make_cfg()
    na = NoveltyArchive(cfg)

    vec = np.zeros(12, dtype=np.float32)
    vec[0] = 0.5
    vec[10] = 0.9  # good survival
    vec[4] = 0.1   # ok speed var

    result = na.try_add(_make_genome(), vec, fitness=0.5)
    assert result == True
    assert na.size() == 1
    print("PASS: test_novelty_archive_first_entry")


def test_novelty_archive_rejects_dead():
    """Dead universe results are rejected."""
    from src.evolution.map_elites import NoveltyArchive

    cfg = _make_cfg()
    na = NoveltyArchive(cfg)

    # Dead: 5% survival
    vec = np.zeros(12, dtype=np.float32)
    vec[10] = 0.05

    result = na.try_add(_make_genome(), vec, fitness=0.0)
    assert result == False
    assert na.size() == 0
    print("PASS: test_novelty_archive_rejects_dead")


def test_novelty_archive_adds_novel():
    """Novel behaviors are added to archive."""
    from src.evolution.map_elites import NoveltyArchive

    cfg = _make_cfg()
    na = NoveltyArchive(cfg)

    # Add several distinct behaviors
    for i in range(20):
        vec = np.zeros(12, dtype=np.float32)
        vec[0] = float(i) / 20  # different entropy
        vec[10] = 0.9  # good survival
        vec[4] = 0.1   # ok speed var

        na.try_add(_make_genome(f'g{i}', gen=i), vec, fitness=0.5)

    assert na.size() > 0
    print(f"PASS: test_novelty_archive_adds_novel → size={na.size()}")


def test_novelty_archive_random_entry():
    """Random selection from archive."""
    from src.evolution.map_elites import NoveltyArchive

    cfg = _make_cfg()
    na = NoveltyArchive(cfg)

    for i in range(10):
        vec = np.random.rand(12).astype(np.float32)
        vec[10] = 0.9  # survival
        vec[4] = 0.1   # speed var
        na.try_add(_make_genome(f'g{i}'), vec, fitness=0.5)

    if na.size() > 0:
        entry = na.random_entry()
        assert entry is not None
    print("PASS: test_novelty_archive_random_entry")


def test_novelty_archive_stats():
    """Archive statistics."""
    from src.evolution.map_elites import NoveltyArchive

    cfg = _make_cfg()
    na = NoveltyArchive(cfg)

    stats = na.get_stats()
    assert stats['size'] == 0
    assert stats['threshold'] > 0
    print(f"PASS: test_novelty_archive_stats → {stats}")


def test_novelty_archive_formulas():
    """Get all formulas from archive."""
    from src.evolution.map_elites import NoveltyArchive

    cfg = _make_cfg()
    na = NoveltyArchive(cfg)

    for i in range(5):
        vec = np.random.rand(12).astype(np.float32)
        vec[10] = 0.9
        vec[4] = 0.1
        na.try_add(_make_genome(f'g{i}'), vec, fitness=0.5)

    formulas = na.get_all_formulas()
    assert len(formulas) == na.size()
    print(f"PASS: test_novelty_archive_formulas → {len(formulas)} entries")


if __name__ == '__main__':
    test_map_elites_init()
    test_map_elites_archive_first()
    test_map_elites_replace_if_better()
    test_map_elites_keep_if_worse()
    test_map_elites_different_cells()
    test_map_elites_random_non_empty()
    test_map_elites_random_empty()
    test_map_elites_fill_ratio()
    test_map_elites_export_summary()
    test_novelty_archive_init()
    test_novelty_archive_first_entry()
    test_novelty_archive_rejects_dead()
    test_novelty_archive_adds_novel()
    test_novelty_archive_random_entry()
    test_novelty_archive_stats()
    test_novelty_archive_formulas()
    print("\nAll MAP-Elites tests passed!")
