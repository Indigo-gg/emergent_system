"""
SQLite storage layer with WAL mode and single-writer pattern.
"""

import sqlite3
import os
import json
from datetime import datetime


class ExperimentDB:
    """SQLite database for experiment data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                generation INTEGER PRIMARY KEY,
                best_fitness REAL,
                avg_fitness REAL,
                novel_count INTEGER,
                dead_count INTEGER,
                vlm_calls INTEGER,
                grid_filled INTEGER,
                archive_size INTEGER,
                elapsed_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS grid_cells (
                grid_key TEXT PRIMARY KEY,
                generation INTEGER,
                fitness REAL,
                f_entropy_mean REAL,
                f_islands_mean REAL,
                f_fft_amp_1 REAL,
                features_12d TEXT,
                potential_formula TEXT,
                state_formula TEXT,
                sense_formula TEXT,
                random_seed INTEGER,
                screenshot_path TEXT,
                vlm_name TEXT,
                vlm_judgment TEXT,
                vlm_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS novelty_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                generation INTEGER,
                fitness REAL,
                novelty_score REAL,
                features_12d TEXT,
                potential_formula TEXT,
                state_formula TEXT,
                sense_formula TEXT,
                random_seed INTEGER,
                screenshot_path TEXT,
                gif_path TEXT,
                vlm_name TEXT,
                vlm_judgment TEXT,
                vlm_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    def log_generation(self, gen: int, best_fit: float, avg_fit: float,
                       novel: int = 0, dead: int = 0, vlm: int = 0,
                       grid: int = 0, archive: int = 0, elapsed: float = 0.0):
        """Log one generation's stats."""
        self.conn.execute(
            """INSERT OR REPLACE INTO evolution_log
               (generation, best_fitness, avg_fitness, novel_count, dead_count,
                vlm_calls, grid_filled, archive_size, elapsed_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (gen, best_fit, avg_fit, novel, dead, vlm, grid, archive, elapsed)
        )
        self.conn.commit()

    def get_evolution_log(self) -> list:
        """Retrieve all evolution log entries ordered by generation."""
        cursor = self.conn.execute(
            """SELECT generation, best_fitness, avg_fitness, novel_count,
                      dead_count, vlm_calls, grid_filled, archive_size,
                      elapsed_seconds
               FROM evolution_log ORDER BY generation"""
        )
        return cursor.fetchall()

    def insert_grid_cell(self, key: str, gen: int, fitness: float,
                         features_3d: tuple, features_12d: list,
                         formula: str, seed: int):
        """Insert or replace a MAP-Elites grid cell."""
        self.conn.execute(
            """INSERT OR REPLACE INTO grid_cells
               (grid_key, generation, fitness, f_entropy_mean,
                f_islands_mean, f_fft_amp_1, features_12d,
                potential_formula, random_seed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, gen, fitness,
             features_3d[0] if len(features_3d) > 0 else 0,
             features_3d[1] if len(features_3d) > 1 else 0,
             features_3d[2] if len(features_3d) > 2 else 0,
             json.dumps(features_12d.tolist() if hasattr(features_12d, 'tolist') else features_12d) if features_12d is not None else None,
             formula, seed)
        )
        self.conn.commit()

    def insert_novelty_entry(self, gen: int, fitness: float,
                             novelty_score: float, features_12d: list,
                             formula: str, seed: int):
        """Insert a novelty archive entry."""
        self.conn.execute(
            """INSERT INTO novelty_archive
               (generation, fitness, novelty_score, features_12d,
                potential_formula, random_seed)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (gen, fitness, novelty_score,
             json.dumps(features_12d.tolist() if hasattr(features_12d, 'tolist') else features_12d) if features_12d is not None else None,
             formula, seed)
        )
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()
