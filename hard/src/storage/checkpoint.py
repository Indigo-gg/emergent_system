"""
Checkpoint: Save and restore evolution state.

Provides:
- Full checkpoint save (generation, grid, archive, population, RNG state)
- Checkpoint load and restore
- Auto-save management
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger('hard-mode')


def save_checkpoint(path: str, generation: int,
                    map_elites, novelty_archive,
                    population: list, rng_state: dict,
                    cfg: dict = None) -> str:
    """
    Save full evolution checkpoint.

    Args:
        path: Output file path
        generation: Current generation number
        map_elites: MAPElites instance
        novelty_archive: NoveltyArchive instance
        population: List of current GEPGenome objects
        rng_state: Dict with RNG state (random module state, numpy state)
        cfg: Optional configuration dict

    Returns:
        Path to saved checkpoint
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    # Prepare checkpoint data
    checkpoint = {
        'version': 1,
        'timestamp': datetime.now().isoformat(),
        'generation': generation,
        'cfg': cfg,
        'rng_state': rng_state,
    }

    # Save MAP-Elites grid
    grid_data = {}
    for key, cell in map_elites.grid.items():
        grid_data[str(key)] = {
            'fitness': cell['fitness'],
            'features_3d': cell['features_3d'],
            'features_12d': cell.get('features_12d', []),
            'formula': cell.get('formula', ''),
            'seed': cell.get('seed', 0),
            'generation': cell.get('generation', 0),
            'genome_gene': cell['genome'].potential_gene if hasattr(cell['genome'], 'potential_gene') else [],
            'genome_head_length': cell['genome'].head_length if hasattr(cell['genome'], 'head_length') else 8,
        }
    checkpoint['map_elites_grid'] = grid_data

    # Save Novelty Archive
    archive_data = []
    for entry in novelty_archive.archive:
        archive_data.append({
            'behavior_vector': entry.get('behavior_vector', []),
            'novelty_score': entry.get('novelty_score', 0),
            'fitness': entry.get('fitness', 0),
            'seed': entry.get('seed', 0),
            'generation': entry.get('generation', 0),
            'formula': entry.get('formula', ''),
            'genome_gene': entry['genome'].potential_gene if hasattr(entry.get('genome'), 'potential_gene') else [],
            'genome_head_length': entry['genome'].head_length if hasattr(entry.get('genome'), 'head_length') else 8,
        })
    checkpoint['novelty_archive'] = archive_data

    # Save population
    pop_data = []
    for genome in population:
        pop_data.append({
            'potential_gene': genome.potential_gene,
            'state_gene': genome.state_gene if hasattr(genome, 'state_gene') else [],
            'sense_gene': genome.sense_gene if hasattr(genome, 'sense_gene') else [],
            'head_length': genome.head_length,
            'generation': genome.generation,
            'random_seed': genome.random_seed,
            'fitness': genome.fitness,
        })
    checkpoint['population'] = pop_data

    # Save adaptive threshold state
    if hasattr(novelty_archive, 'threshold_manager'):
        tm = novelty_archive.threshold_manager
        checkpoint['adaptive_threshold'] = {
            'threshold': tm.threshold,
            'stale_count': tm.stale_count,
            'history': tm.history[-100:],  # Keep last 100
        }

    # Write to file
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)

    logger.info(f"Checkpoint saved: gen={generation}, path={path}")
    return path


def load_checkpoint(path: str) -> dict:
    """
    Load evolution checkpoint.

    Args:
        path: Checkpoint file path

    Returns:
        Checkpoint dict with all state
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    logger.info(f"Checkpoint loaded: gen={checkpoint['generation']}, path={path}")
    return checkpoint


def restore_from_checkpoint(checkpoint: dict, map_elites, novelty_archive):
    """
    Restore evolution state from checkpoint.

    Args:
        checkpoint: Checkpoint dict from load_checkpoint
        map_elites: MAPElites instance to restore
        novelty_archive: NoveltyArchive instance to restore
    """
    from src.evolution.genome import GEPGenome

    # Restore MAP-Elites grid
    map_elites.grid = {}
    for key_str, cell_data in checkpoint.get('map_elites_grid', {}).items():
        # Reconstruct genome
        genome = GEPGenome(
            potential_gene=cell_data.get('genome_gene', []),
            head_length=cell_data.get('genome_head_length', 8),
            generation=cell_data.get('generation', 0),
            random_seed=cell_data.get('seed', 0),
        )
        genome.fitness = cell_data.get('fitness', 0)

        # Parse key
        try:
            key = tuple(int(x) for x in key_str.strip('()').split(','))
        except:
            key = key_str

        map_elites.grid[key] = {
            'genome': genome,
            'fitness': cell_data.get('fitness', 0),
            'features_3d': cell_data.get('features_3d', (0, 0, 0)),
            'features_12d': cell_data.get('features_12d', []),
            'formula': cell_data.get('formula', ''),
            'seed': cell_data.get('seed', 0),
            'generation': cell_data.get('generation', 0),
        }

    # Restore Novelty Archive
    novelty_archive.archive = []
    for entry_data in checkpoint.get('novelty_archive', []):
        genome = GEPGenome(
            potential_gene=entry_data.get('genome_gene', []),
            head_length=entry_data.get('genome_head_length', 8),
            generation=entry_data.get('generation', 0),
            random_seed=entry_data.get('seed', 0),
        )
        novelty_archive.archive.append({
            'genome': genome,
            'behavior_vector': entry_data.get('behavior_vector', []),
            'novelty_score': entry_data.get('novelty_score', 0),
            'fitness': entry_data.get('fitness', 0),
            'seed': entry_data.get('seed', 0),
            'generation': entry_data.get('generation', 0),
            'formula': entry_data.get('formula', ''),
        })

    # Restore adaptive threshold
    if 'adaptive_threshold' in checkpoint and hasattr(novelty_archive, 'threshold_manager'):
        tm = novelty_archive.threshold_manager
        at_data = checkpoint['adaptive_threshold']
        tm.threshold = at_data.get('threshold', tm.threshold)
        tm.stale_count = at_data.get('stale_count', 0)
        tm.history = at_data.get('history', [])

    logger.info(f"State restored: grid={len(map_elites.grid)}, archive={len(novelty_archive.archive)}")


def list_checkpoints(checkpoint_dir: str) -> list:
    """
    List available checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        List of (path, generation, timestamp) tuples
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_') and f.endswith('.pkl'):
            path = os.path.join(checkpoint_dir, f)
            try:
                with open(path, 'rb') as fh:
                    data = pickle.load(fh)
                checkpoints.append((
                    path,
                    data.get('generation', 0),
                    data.get('timestamp', ''),
                ))
            except:
                continue

    # Sort by generation
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get path to latest checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    if checkpoints:
        return checkpoints[-1][0]
    return None


def cleanup_old_checkpoints(checkpoint_dir: str, max_keep: int = 5):
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        max_keep: Maximum number of checkpoints to keep
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    if len(checkpoints) <= max_keep:
        return

    # Remove oldest
    to_remove = checkpoints[:-max_keep]
    for path, gen, _ in to_remove:
        try:
            os.remove(path)
            logger.info(f"Removed old checkpoint: gen={gen}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {path}: {e}")
