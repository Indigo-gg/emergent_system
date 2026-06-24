"""
Visualizer: Analysis and visualization tools for evolution results.

Provides:
- 3D MAP-Elites grid visualization
- Evolution curve plotting
- Formula collection export
- Report generation
- Novelty Archive browsing
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger('hard-mode')


def visualize_map_elites_3d(map_elites, cfg: dict,
                            output_path: str = None) -> str:
    """
    Create 3D scatter plot of MAP-Elites grid.

    Points are colored by fitness, sized by generation.

    Args:
        map_elites: MAPElites instance
        cfg: Configuration dict
        output_path: Output file path

    Returns:
        Path to saved image
    """
    if not map_elites.grid:
        logger.warning("Empty grid, skipping visualization")
        return None

    # Extract data
    entropies = []
    islands = []
    fft_amps = []
    fitnesses = []
    generations = []

    for key, cell in map_elites.grid.items():
        feat = cell.get('features_3d', (0, 0, 0))
        entropies.append(feat[0])
        islands.append(feat[1])
        fft_amps.append(feat[2])
        fitnesses.append(cell.get('fitness', 0))
        generations.append(cell.get('generation', 0))

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Color by fitness, size by generation
    fitnesses = np.array(fitnesses)
    generations = np.array(generations)

    # Normalize for colormap
    norm_fitness = (fitnesses - fitnesses.min()) / max(fitnesses.max() - fitnesses.min(), 0.001)
    colors = plt.cm.viridis(norm_fitness)

    # Size: older = smaller, newer = larger
    max_gen = max(generations.max(), 1)
    sizes = 20 + 80 * (generations / max_gen)

    scatter = ax.scatter(entropies, islands, fft_amps,
                         c=fitnesses, cmap='viridis', s=sizes, alpha=0.7)

    ax.set_xlabel('Spatial Entropy Mean')
    ax.set_ylabel('Islands Mean')
    ax.set_zlabel('FFT Amp 1')
    ax.set_title(f'MAP-Elites Grid ({len(map_elites.grid)}/{map_elites.get_total_cells()} filled)')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Fitness')

    if output_path is None:
        output_path = 'map_elites_3d.png'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"MAP-Elites 3D visualization saved: {output_path}")
    return output_path


def visualize_map_elites_slices(map_elites, cfg: dict,
                                output_path: str = None) -> str:
    """
    Create 2D slice views of MAP-Elites grid (3 projections).

    Args:
        map_elites: MAPElites instance
        cfg: Configuration dict
        output_path: Output file path

    Returns:
        Path to saved image
    """
    if not map_elites.grid:
        return None

    # Extract data
    entropies = []
    islands = []
    fft_amps = []
    fitnesses = []

    for key, cell in map_elites.grid.items():
        feat = cell.get('features_3d', (0, 0, 0))
        entropies.append(feat[0])
        islands.append(feat[1])
        fft_amps.append(feat[2])
        fitnesses.append(cell.get('fitness', 0))

    entropies = np.array(entropies)
    islands = np.array(islands)
    fft_amps = np.array(fft_amps)
    fitnesses = np.array(fitnesses)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # XY: Entropy vs Islands
    sc1 = axes[0].scatter(entropies, islands,
                          c=fitnesses, cmap='viridis', s=30, alpha=0.7)
    axes[0].set_xlabel('Spatial Entropy Mean')
    axes[0].set_ylabel('Islands Mean')
    axes[0].set_title('Entropy x Islands')
    plt.colorbar(sc1, ax=axes[0], label='Fitness')

    # XZ: Entropy vs FFT
    sc2 = axes[1].scatter(entropies, fft_amps,
                          c=fitnesses, cmap='viridis', s=30, alpha=0.7)
    axes[1].set_xlabel('Spatial Entropy Mean')
    axes[1].set_ylabel('FFT Amp 1')
    axes[1].set_title('Entropy x FFT')
    plt.colorbar(sc2, ax=axes[1], label='Fitness')

    # YZ: Islands vs FFT
    sc3 = axes[2].scatter(islands, fft_amps,
                          c=fitnesses, cmap='viridis', s=30, alpha=0.7)
    axes[2].set_xlabel('Islands Mean')
    axes[2].set_ylabel('FFT Amp 1')
    axes[2].set_title('Islands x FFT')
    plt.colorbar(sc3, ax=axes[2], label='Fitness')

    fig.suptitle(f'MAP-Elites Grid Slices ({len(entropies)} cells)', fontsize=14)
    plt.tight_layout()

    if output_path is None:
        output_path = 'map_elites_slices.png'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_evolution_curves(db_path: str, cfg: dict,
                               output_path: str = None) -> str:
    """
    Plot evolution curves from database.

    Shows fitness, novelty count, grid fill, archive size over generations.

    Args:
        db_path: Path to SQLite database
        cfg: Configuration dict
        output_path: Output file path

    Returns:
        Path to saved image
    """
    from src.storage.db import ExperimentDB

    db = ExperimentDB(db_path)
    log = db.get_evolution_log()

    if not log:
        logger.warning("No evolution log data, skipping curves")
        return None

    # Unpack columns: generation, best_fitness, avg_fitness, novel_count,
    #                 dead_count, vlm_calls, grid_filled, archive_size, elapsed
    generations = [row[0] for row in log]
    best_fitness = [row[1] for row in log]
    avg_fitness = [row[2] for row in log]
    novel_count = [row[3] for row in log]
    grid_fill = [row[6] for row in log]
    archive_size = [row[7] for row in log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Best & Average Fitness
    axes[0, 0].plot(generations, best_fitness, label='Best', linewidth=2)
    axes[0, 0].plot(generations, avg_fitness, label='Average', linewidth=1, alpha=0.7)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Novel Count
    axes[0, 1].bar(generations, novel_count, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Novel Findings')
    axes[0, 1].set_title('Novelty Discoveries')
    axes[0, 1].grid(True, alpha=0.3)

    # Grid Fill
    axes[1, 0].plot(generations, grid_fill, linewidth=2, color='green')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Cells Filled')
    axes[1, 0].set_title('MAP-Elites Grid Fill')
    axes[1, 0].grid(True, alpha=0.3)

    # Archive Size
    axes[1, 1].plot(generations, archive_size, linewidth=2, color='red')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Archive Entries')
    axes[1, 1].set_title('Novelty Archive Size')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Evolution Progress', fontsize=14)
    plt.tight_layout()

    if output_path is None:
        output_path = 'evolution_curves.png'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Evolution curves saved: {output_path}")
    return output_path


def export_formula_collection(map_elites, novelty_archive,
                              output_path: str = None) -> str:
    """
    Export all formulas as a human-readable collection.

    Args:
        map_elites: MAPElites instance
        novelty_archive: NoveltyArchive instance
        output_path: Output file path

    Returns:
        Path to saved file
    """
    lines = [
        "=" * 80,
        "EMERGENT SYSTEM - FORMULA COLLECTION",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 80,
        "",
    ]

    # MAP-Elites Grid Formulas
    lines.append("## MAP-Elites Grid Formulas")
    lines.append(f"   Filled: {map_elites.get_filled_count()}/{map_elites.get_total_cells()} cells")
    lines.append("")

    grid_formulas = map_elites.get_all_formulas()
    grid_formulas.sort(key=lambda x: x.get('fitness', 0), reverse=True)

    for i, entry in enumerate(grid_formulas[:50]):  # Top 50
        lines.append(f"  #{i+1:3d} | Fitness: {entry.get('fitness', 0):.4f} | "
                     f"Features: {entry.get('features_3d', (0,0,0))}")
        lines.append(f"       Formula: {entry.get('formula', 'N/A')}")
        lines.append("")

    # Novelty Archive Formulas
    lines.append("## Novelty Archive Formulas")
    lines.append(f"   Entries: {novelty_archive.size()}")
    lines.append("")

    archive_formulas = novelty_archive.get_all_formulas()
    archive_formulas.sort(key=lambda x: x.get('novelty_score', 0), reverse=True)

    for i, entry in enumerate(archive_formulas[:50]):  # Top 50
        lines.append(f"  #{i+1:3d} | Novelty: {entry.get('novelty_score', 0):.4f} | "
                     f"Gen: {entry.get('generation', 0)}")
        lines.append(f"       Formula: {entry.get('formula', 'N/A')}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF COLLECTION")
    lines.append("=" * 80)

    content = '\n'.join(lines)

    if output_path is None:
        output_path = 'formula_collection.txt'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logger.info(f"Formula collection exported: {output_path}")
    return output_path


def browse_novelty_archive(novelty_archive, top_n: int = 20) -> List[Dict]:
    """
    Get top novelty entries for browsing.

    Args:
        novelty_archive: NoveltyArchive instance
        top_n: Number of top entries to return

    Returns:
        List of entry dicts sorted by novelty score
    """
    entries = novelty_archive.get_all_formulas()
    entries.sort(key=lambda x: x.get('novelty_score', 0), reverse=True)
    return entries[:top_n]


def generate_report(map_elites, novelty_archive, cfg: dict,
                    output_dir: str = None) -> str:
    """
    Generate a comprehensive experiment report.

    Includes:
    - Experiment configuration
    - MAP-Elites statistics
    - Novelty Archive statistics
    - Top formulas
    - Evolution progress

    Args:
        map_elites: MAPElites instance
        novelty_archive: NoveltyArchive instance
        cfg: Configuration dict
        output_dir: Output directory

    Returns:
        Path to report directory
    """
    if output_dir is None:
        exp_name = cfg.get('experiment', {}).get('name', 'experiment')
        output_dir = os.path.join('data', 'experiments', exp_name, 'report')

    os.makedirs(output_dir, exist_ok=True)

    # 1. Configuration summary
    config_path = os.path.join(output_dir, 'config_summary.txt')
    with open(config_path, 'w') as f:
        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("=" * 40 + "\n\n")
        for section, values in cfg.items():
            f.write(f"[{section}]\n")
            if isinstance(values, dict):
                for k, v in values.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {values}\n")
            f.write("\n")

    # 2. MAP-Elites visualization
    grid_path = os.path.join(output_dir, 'map_elites_3d.png')
    visualize_map_elites_3d(map_elites, cfg, grid_path)

    slices_path = os.path.join(output_dir, 'map_elites_slices.png')
    visualize_map_elites_slices(map_elites, cfg, slices_path)

    # 3. Formula collection
    formula_path = os.path.join(output_dir, 'formula_collection.txt')
    export_formula_collection(map_elites, novelty_archive, formula_path)

    # 4. Statistics summary
    stats_path = os.path.join(output_dir, 'statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("EVOLUTION STATISTICS\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"MAP-Elites Grid:\n")
        f.write(f"  Filled cells: {map_elites.get_filled_count()}/{map_elites.get_total_cells()}\n")
        f.write(f"  Fill ratio: {map_elites.get_fill_ratio():.2%}\n\n")

        stats = novelty_archive.get_stats()
        f.write(f"Novelty Archive:\n")
        f.write(f"  Entries: {stats.get('size', 0)}\n")
        f.write(f"  Mean novelty score: {stats.get('mean_score', 0):.4f}\n")
        f.write(f"  Current threshold: {stats.get('threshold', 0):.4f}\n")

    # 5. Top formulas summary
    top_path = os.path.join(output_dir, 'top_formulas.txt')
    with open(top_path, 'w') as f:
        f.write("TOP FORMULAS\n")
        f.write("=" * 40 + "\n\n")

        # Top by fitness
        f.write("## Top 10 by Fitness (MAP-Elites)\n\n")
        grid_formulas = map_elites.get_all_formulas()
        grid_formulas.sort(key=lambda x: x.get('fitness', 0), reverse=True)
        for i, entry in enumerate(grid_formulas[:10]):
            f.write(f"{i+1}. Fitness={entry.get('fitness', 0):.4f}\n")
            f.write(f"   Formula: {entry.get('formula', 'N/A')}\n")
            f.write(f"   Features: {entry.get('features_3d', (0,0,0))}\n\n")

        # Top by novelty
        f.write("## Top 10 by Novelty (Archive)\n\n")
        archive_formulas = novelty_archive.get_all_formulas()
        archive_formulas.sort(key=lambda x: x.get('novelty_score', 0), reverse=True)
        for i, entry in enumerate(archive_formulas[:10]):
            f.write(f"{i+1}. Novelty={entry.get('novelty_score', 0):.4f}\n")
            f.write(f"   Formula: {entry.get('formula', 'N/A')}\n")
            f.write(f"   Generation: {entry.get('generation', 0)}\n\n")

    logger.info(f"Report generated: {output_dir}")
    return output_dir
