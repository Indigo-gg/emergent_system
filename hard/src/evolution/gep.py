"""
GEP Engine: fitness evaluation and evolution loop.

Handles:
- Fitness evaluation (run simulation + extract features)
- Dead universe filtering
- Parsimony pressure
- Integration with simulation step
"""

import math
import random
import logging
import numpy as np

from src.evolution.genome import (
    GEPGenome, random_genome, decode_gene, FUNCTION_ARITY
)
from src.simulation.potential import Node, Const, compile_potential


logger = logging.getLogger('hard-mode')


# ── Fitness Evaluation ──

def evaluate_fitness(genome: GEPGenome, sim_components: dict, cfg: dict,
                     feature_extractor=None) -> tuple:
    """
    Evaluate a genome's fitness by running simulation.

    Also collects features if a FeatureExtractor is provided,
    avoiding the need to re-run the simulation.

    Args:
        genome: The GEP genome to evaluate
        sim_components: dict with 'particles', 'spatial_hash', 'integrator', 'step'
        cfg: configuration dict
        feature_extractor: optional FeatureExtractor to collect features during run

    Returns:
        (fitness, features_12d, features_3d) tuple
        fitness: float score (0.0 if invalid)
        features_12d: 12D feature vector (or None)
        features_3d: 3D MAP-Elites features (or None)
    """
    particles = sim_components['particles']
    sim_step = sim_components['step']

    # 1. Compile genome to bytecode
    try:
        genome.compile(cfg['gep']['bytecode_length'])
    except Exception as e:
        logger.warning(f"Compilation failed: {e}")
        return 0.0, None, None

    # 2. Check for NaN/Inf in bytecode constants
    consts = genome._potential_constants
    if any(math.isnan(c) or math.isinf(c) for c in consts):
        logger.debug("NaN/Inf in constants → fitness = 0")
        return 0.0, None, None

    # 3. Set the potential formula in simulation
    tree = decode_gene(genome.potential_gene, genome.head_length)
    sim_step.set_potential(tree)

    # 4. Re-initialize particles
    seed = genome.random_seed
    particles.initialize(seed)

    # 5. Run simulation
    steps = cfg['simulation'].get('steps_per_eval', 5000)
    sample_interval = cfg['novelty'].get('sample_interval', 500)

    speed_variances = []
    survival_counts = []
    n = cfg['simulation']['num_particles']

    if feature_extractor:
        feature_extractor.reset()

    for step in range(steps):
        sim_step.step(particles)

        # Sample features at intervals
        if (step + 1) % sample_interval == 0:
            vel_x = particles.vel_x.to_numpy()
            vel_y = particles.vel_y.to_numpy()
            alive = particles.alive.to_numpy()

            # Speed variance
            speeds = np.sqrt(vel_x**2 + vel_y**2)
            speed_var = np.var(speeds[alive == 1]) if np.sum(alive) > 0 else 0.0
            speed_variances.append(speed_var)

            # Survival count
            alive_count = np.sum(alive)
            survival_counts.append(alive_count)

            # Full feature extraction if requested
            if feature_extractor:
                feature_extractor.sample(particles, cfg)

    # 6. Compute fitness components
    if not survival_counts:
        return 0.0, None, None

    final_alive = survival_counts[-1]
    survival_rate = final_alive / n

    # Dead universe filter
    min_survival = cfg['novelty'].get('min_survival_rate', 0.1)
    if survival_rate < min_survival:
        logger.debug(f"Dead universe: survival_rate={survival_rate:.3f} < {min_survival}")
        return 0.0, None, None

    # Speed variance filter (all particles static)
    avg_speed_var = np.mean(speed_variances) if speed_variances else 0.0
    min_speed_var = cfg['novelty'].get('min_speed_variance', 0.001)
    if avg_speed_var < min_speed_var:
        logger.debug(f"All static: avg_speed_var={avg_speed_var:.6f} < {min_speed_var}")
        return 0.0, None, None

    # 7. Multi-objective fitness
    w1 = 0.4
    f_survival = survival_rate

    w2 = 0.3
    f_structure = min(avg_speed_var / 1.0, 1.0)

    w3 = 0.3
    if len(survival_counts) > 1:
        survival_stability = 1.0 - (np.std(survival_counts) / max(np.mean(survival_counts), 1.0))
        f_efficiency = max(0.0, survival_stability)
    else:
        f_efficiency = 0.0

    fitness = w1 * f_survival + w2 * f_structure + w3 * f_efficiency

    # 8. Parsimony pressure
    parsimony = cfg['evolution'].get('parsimony_pressure', 0.001)
    tree = decode_gene(genome.potential_gene, genome.head_length)
    effective_size = tree.size()
    fitness -= parsimony * effective_size

    # 9. NaN check
    if math.isnan(fitness) or math.isinf(fitness):
        return 0.0, None, None

    # 10. Compute features if extractor was provided
    features_12d = None
    features_3d = None
    if feature_extractor:
        features_12d = feature_extractor.compute_features()
        features_3d = feature_extractor.get_3d_features()

    return max(0.0, fitness), features_12d, features_3d


# ── Feature Extraction (basic version for Phase 2) ──

def extract_basic_features(particles, cfg: dict) -> tuple:
    """
    Extract basic 3D features for MAP-Elites grid.
    Returns (entropy_mean, islands_mean, fft_amp_1).
    """
    pos_x = particles.pos_x.to_numpy()
    pos_y = particles.pos_y.to_numpy()
    vel_x = particles.vel_x.to_numpy()
    vel_y = particles.vel_y.to_numpy()
    alive = particles.alive.to_numpy()

    n_alive = np.sum(alive)
    if n_alive == 0:
        return (0.0, 0.0, 0.0)

    # Filter to alive particles only
    px = pos_x[alive == 1]
    py = pos_y[alive == 1]
    vx = vel_x[alive == 1]
    vy = vel_y[alive == 1]

    w = cfg['world']['width']
    h = cfg['world']['height']

    # 1. Spatial entropy
    grid_size = 16
    hist, _, _ = np.histogram2d(px, py, bins=grid_size, range=[[0, w], [0, h]])
    probs = hist.flatten() / max(n_alive, 1)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(grid_size * grid_size)
    entropy_norm = entropy / max(max_entropy, 1e-10)

    # 2. Island count (connected components via simple grid)
    cell_size = cfg['world']['cell_size'] * 3
    grid_w = int(w / cell_size) + 1
    grid_h = int(h / cell_size) + 1
    occupancy = np.zeros((grid_h, grid_w), dtype=bool)
    for i in range(len(px)):
        col = min(int(px[i] / cell_size), grid_w - 1)
        row = min(int(py[i] / cell_size), grid_h - 1)
        occupancy[row, col] = True
    islands = _count_islands(occupancy)

    # 3. Speed variance FFT (simplified: just the variance)
    speeds = np.sqrt(vx**2 + vy**2)
    speed_var = np.var(speeds)

    return (entropy_norm, float(islands), speed_var)


def _count_islands(grid: np.ndarray) -> int:
    """Count connected components in a boolean grid."""
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] and not visited[r, c]:
                count += 1
                _flood_fill(grid, visited, r, c)

    return count


def _flood_fill(grid, visited, r, c):
    """Flood fill from (r, c)."""
    rows, cols = grid.shape
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
            continue
        if visited[cr, cc] or not grid[cr, cc]:
            continue
        visited[cr, cc] = True
        stack.extend([(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)])


# ── Parent Selection ──

def select_parent(map_elites_grid: dict, novelty_archive: list,
                  grid_prob: float = 0.7, rng: random.Random = None) -> GEPGenome:
    """
    Select a parent genome using hybrid strategy:
    - grid_prob probability from MAP-Elites grid (prefer high fitness)
    - (1 - grid_prob) probability from Novelty Archive (prefer diversity)
    """
    if rng is None:
        rng = random.Random()

    if rng.random() < grid_prob and map_elites_grid:
        # Select from grid (weighted by fitness)
        entries = list(map_elites_grid.values())
        if entries:
            weights = [max(e['fitness'], 0.001) for e in entries]
            total = sum(weights)
            weights = [w / total for w in weights]
            chosen = rng.choices(entries, weights=weights, k=1)[0]
            return chosen['genome']
    elif novelty_archive:
        # Select from archive (uniform)
        chosen = rng.choice(novelty_archive)
        return chosen['genome']

    # Fallback: random genome
    return None
