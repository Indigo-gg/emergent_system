"""
SQLite storage layer tests.
Run with: python -m pytest tests/test_db.py -v
"""

import os
import sys
import tempfile
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_table_creation():
    """Tables are created on init."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = ExperimentDB(db_path)

        # Check tables exist
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert 'evolution_log' in tables, "evolution_log table missing"
        assert 'grid_cells' in tables, "grid_cells table missing"
        assert 'novelty_archive' in tables, "novelty_archive table missing"

        db.close()
    print("PASS: test_table_creation")


def test_wal_mode():
    """Database uses WAL journal mode."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = ExperimentDB(db_path)

        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == 'wal', f"Expected WAL mode, got {mode}"

        db.close()
    print("PASS: test_wal_mode")


def test_log_generation():
    """Can write and read evolution log entries."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = ExperimentDB(db_path)

        db.log_generation(
            gen=1, best_fit=0.85, avg_fit=0.65,
            novel=2, dead=1, vlm=0, grid=10, archive=5, elapsed=1.23
        )
        db.log_generation(
            gen=2, best_fit=0.90, avg_fit=0.70,
            novel=3, dead=0, vlm=1, grid=15, archive=8, elapsed=2.45
        )

        cursor = db.conn.execute(
            "SELECT generation, best_fitness, avg_fitness FROM evolution_log ORDER BY generation"
        )
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, 0.85, 0.65)
        assert rows[1] == (2, 0.90, 0.70)

        db.close()
    print("PASS: test_log_generation")


def test_log_generation_upsert():
    """INSERT OR REPLACE updates existing generation."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = ExperimentDB(db_path)

        db.log_generation(gen=1, best_fit=0.5, avg_fit=0.3)
        db.log_generation(gen=1, best_fit=0.8, avg_fit=0.6)  # update

        cursor = db.conn.execute(
            "SELECT best_fitness, avg_fitness FROM evolution_log WHERE generation=1"
        )
        row = cursor.fetchone()
        assert row == (0.8, 0.6), f"Expected (0.8, 0.6), got {row}"

        db.close()
    print("PASS: test_log_generation_upsert")


def test_grid_cells_schema():
    """grid_cells table has correct columns."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = ExperimentDB(db_path)

        cursor = db.conn.execute("PRAGMA table_info(grid_cells)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            'grid_key', 'generation', 'fitness',
            'f_entropy_mean', 'f_islands_mean', 'f_fft_amp_1',
            'features_12d', 'potential_formula', 'state_formula',
            'sense_formula', 'random_seed', 'screenshot_path',
            'vlm_name', 'vlm_judgment', 'vlm_score', 'created_at',
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"

        db.close()
    print("PASS: test_grid_cells_schema")


def test_novelty_archive_schema():
    """novelty_archive table has correct columns."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')
        db = ExperimentDB(db_path)

        cursor = db.conn.execute("PRAGMA table_info(novelty_archive)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            'id', 'generation', 'fitness', 'novelty_score',
            'features_12d', 'potential_formula', 'state_formula',
            'sense_formula', 'random_seed', 'screenshot_path',
            'gif_path', 'vlm_name', 'vlm_judgment', 'vlm_score',
            'created_at',
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"

        db.close()
    print("PASS: test_novelty_archive_schema")


def test_close_reopen():
    """Data persists after close and reopen."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test.db')

        # Write
        db = ExperimentDB(db_path)
        db.log_generation(gen=42, best_fit=0.99, avg_fit=0.88)
        db.close()

        # Reopen and read
        db2 = ExperimentDB(db_path)
        cursor = db2.conn.execute(
            "SELECT best_fitness FROM evolution_log WHERE generation=42"
        )
        row = cursor.fetchone()
        assert row is not None and row[0] == 0.99
        db2.close()
    print("PASS: test_close_reopen")


def test_directory_auto_create():
    """DB creates parent directories automatically."""
    from src.storage.db import ExperimentDB

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'subdir', 'deep', 'test.db')
        db = ExperimentDB(db_path)
        assert os.path.exists(db_path)
        db.close()
    print("PASS: test_directory_auto_create")


if __name__ == '__main__':
    test_table_creation()
    test_wal_mode()
    test_log_generation()
    test_log_generation_upsert()
    test_grid_cells_schema()
    test_novelty_archive_schema()
    test_close_reopen()
    test_directory_auto_create()
    print("\nAll DB tests passed!")
