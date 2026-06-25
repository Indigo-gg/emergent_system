"""
Hard Mode Emergence System — Main Entry Point

Phases:
- Phase 0: Config loading + Taichi initialization + logging
- Phase 1: Basic simulation loop with hardcoded potential
- Phase 2: GEP evolution engine
- Phase 3: Hybrid feature archive (MAP-Elites + Novelty Archive)
- Phase 4: Rendering & VLM integration
- Phase 5: Reliability (checkpoint, monitoring)
"""

import argparse
import logging
import os
import sys
import time
import random
import signal

import yaml
import taichi as ti
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(cfg: dict) -> logging.Logger:
    """Setup logging."""
    log_dir = os.path.join('data', 'experiments', cfg['experiment']['name'], 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('hard-mode')
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def init_taichi(cfg: dict, logger: logging.Logger):
    """Initialize Taichi with CUDA backend."""
    ti.init(arch=ti.cuda, debug=False)
    logger.info("Taichi initialized: arch=cuda")


def run_phase1(cfg: dict, logger: logging.Logger):
    """Run Phase 1: basic simulation with hardcoded potential."""
    from src.simulation.particles import ParticleSystem
    from src.simulation.spatial_hash import SpatialHash
    from src.simulation.integrator import Integrator
    from src.simulation.step import SimulationStep
    from src.simulation.environment import EnvironmentLayer
    from src.storage.db import ExperimentDB

    n = cfg['simulation']['num_particles']
    steps = cfg['simulation']['steps_per_eval']
    seed = cfg['experiment']['seed']

    logger.info(f"Starting Phase 1: {n} particles, {steps} steps")

    # Initialize components
    particles = ParticleSystem(cfg)
    spatial_hash = SpatialHash(cfg)
    integrator = Integrator(cfg)
    sim_step = SimulationStep(spatial_hash, integrator, cfg)
    environment = EnvironmentLayer(cfg)

    # Initialize particles and environment
    particles.initialize(seed)
    environment.initialize(seed)
    logger.info("Particles + environment initialized")

    # Database
    db_path = os.path.join('data', 'experiments', cfg['experiment']['name'],
                           'archive.db')
    db = ExperimentDB(db_path)

    # Run simulation
    logger.info(f"Running {steps} steps...")
    t_start = time.time()

    for step in range(steps):
        sim_step.step(particles, environment)

        if (step + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            fps = (step + 1) / elapsed
            alive = particles.count_alive()
            logger.info(f"  step={step+1}/{steps} | fps={fps:.1f} | alive={alive}/{n}")

    elapsed = time.time() - t_start
    logger.info(f"Simulation complete: {steps} steps in {elapsed:.2f}s "
                f"({steps/elapsed:.1f} steps/s)")

    # Export final positions for rendering
    pos_x = particles.pos_x.to_numpy()
    pos_y = particles.pos_y.to_numpy()

    return pos_x, pos_y, cfg


def render_heatmap(pos_x, pos_y, cfg, output_path: str = None):
    """Render a density heatmap from particle positions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    w = cfg['world']['width']
    h = cfg['world']['height']
    res = cfg['rendering']['resolution']

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist2d(pos_x, pos_y, bins=res, range=[[0, w], [0, h]],
              cmap='inferno', density=True)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.set_title(f'Density Heatmap ({len(pos_x)} particles)')

    if output_path is None:
        output_path = os.path.join('data', 'experiments',
                                   cfg['experiment']['name'], 'heatmap.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def run_evolution(cfg: dict, logger: logging.Logger, max_generations: int = None):
    """
    Run full evolution loop with Phase 3+4+5 integration.

    Args:
        cfg: Configuration dict
        logger: Logger instance
        max_generations: Maximum generations to run (None = unlimited)
    """
    from src.simulation.particles import ParticleSystem
    from src.simulation.spatial_hash import SpatialHash
    from src.simulation.integrator import Integrator
    from src.simulation.step import SimulationStep
    from src.simulation.environment import EnvironmentLayer
    from src.evolution.genome import random_genome, decode_gene
    from src.evolution.mutation import mutate
    from src.evolution.gep import evaluate_fitness
    from src.evolution.features import FeatureExtractor
    from src.evolution.map_elites import MAPElites, NoveltyArchive
    from src.storage.db import ExperimentDB
    from src.storage.checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint, cleanup_old_checkpoints
    from src.rendering.renderer import render_novelty_package
    from src.vlm.judge import (
        vlm_judge, vlm_session, check_daily_limit,
        should_call_vlm, compress_image_for_vlm, get_vlm_cache_stats
    )
    from src.monitoring.monitor import EvolutionMonitor, log_generation_stats

    # Initialize simulation components
    particles = ParticleSystem(cfg)
    spatial_hash = SpatialHash(cfg)
    integrator = Integrator(cfg)
    sim_step = SimulationStep(spatial_hash, integrator, cfg)

    # v6: Initialize environment layer
    environment = EnvironmentLayer(cfg)

    sim_components = {
        'particles': particles,
        'spatial_hash': spatial_hash,
        'integrator': integrator,
        'step': sim_step,
        'environment': environment,
    }

    # Initialize Phase 3 components
    fe = FeatureExtractor(cfg)
    me = MAPElites(cfg)
    na = NoveltyArchive(cfg)

    # Initialize Phase 5 components
    monitor = EvolutionMonitor(cfg)
    db_path = os.path.join('data', 'experiments', cfg['experiment']['name'], 'archive.db')
    db = ExperimentDB(db_path)

    # Checkpoint setup
    checkpoint_dir = os.path.join('data', 'experiments', cfg['experiment']['name'], 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_interval = cfg.get('checkpoint', {}).get('interval', 1000)
    max_keep = cfg.get('checkpoint', {}).get('max_keep', 5)

    # Output directories
    exp_dir = os.path.join('data', 'experiments', cfg['experiment']['name'])
    formula_dir = os.path.join(exp_dir, 'formulas')
    screenshot_dir = os.path.join(exp_dir, 'screenshots')
    os.makedirs(formula_dir, exist_ok=True)
    os.makedirs(screenshot_dir, exist_ok=True)

    # RNG setup
    seed = cfg['experiment']['seed']
    rng = random.Random(seed)

    # Try to restore from checkpoint
    start_gen = 0
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        try:
            checkpoint = load_checkpoint(latest_ckpt)
            from src.storage.checkpoint import restore_from_checkpoint
            restore_from_checkpoint(checkpoint, me, na)
            start_gen = checkpoint['generation'] + 1
            logger.info(f"Restored from checkpoint: gen={start_gen}")
        except Exception as e:
            logger.warning(f"Failed to restore checkpoint: {e}")

    # Signal handler for graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info("Shutdown requested (SIGINT)")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)

    # Evolution parameters
    population_size = cfg['evolution']['population_size']
    grid_selection_prob = cfg.get('novelty', {}).get('grid_selection_prob', 0.7)

    # Initialize population
    population = []
    for _ in range(population_size):
        genome = random_genome(cfg, rng)
        population.append(genome)

    logger.info(f"Starting evolution: pop_size={population_size}, start_gen={start_gen}")
    if max_generations:
        logger.info(f"Max generations: {max_generations}")

    t_start = time.time()
    generations_without_novelty = 0

    # Main evolution loop
    gen = start_gen
    judged_features = []  # 12D features already sent to VLM (persists across gens)
    vlm_gen_cap = cfg.get('vlm', {}).get('max_per_generation', 3)

    while not shutdown_requested:
        if max_generations and gen >= max_generations:
            break

        gen_start = time.time()
        novel_count = 0
        dead_count = 0
        fitnesses = []
        novelty_packages = []  # collect for batched VLM calls
        vlm_calls_this_gen = 0  # per-gen VLM cap counter

        # Evaluate each genome in population
        genome_features = {}  # genome_id -> features_15d (for periodic screenshot)
        for i, genome in enumerate(population):
            # Evaluate fitness + extract features in single simulation pass
            fitness, features_15d, features_3d = evaluate_fitness(
                genome, sim_components, cfg, feature_extractor=fe,
                environment=environment
            )

            if fitness <= 0:
                dead_count += 1
                continue

            # Store features for periodic screenshot lookup
            genome_features[genome.get_id()] = features_15d

            # Archive to MAP-Elites
            was_archived = me.try_archive(genome, features_3d, fitness, features_15d, genome.random_seed)
            if was_archived:
                # Persist grid cell to DB
                grid_key = me._discretize(features_3d) if features_3d else (0, 0, 0)
                formula = genome.to_formula() if hasattr(genome, 'to_formula') else ''
                db.insert_grid_cell(
                    key=str(grid_key), gen=gen, fitness=fitness,
                    features_3d=features_3d or (0.0, 0.0, 0.0),
                    features_12d=features_15d,
                    formula=formula, seed=genome.random_seed,
                )

            # Try adding to Novelty Archive
            was_novel = na.try_add(genome, features_15d, fitness, genome.random_seed)
            if was_novel:
                # Persist novelty entry to DB
                formula = genome.to_formula() if hasattr(genome, 'to_formula') else ''
                novelty_score_val = na.archive[-1]['novelty_score'] if na.archive else 0.0
                db.insert_novelty_entry(
                    gen=gen, fitness=fitness,
                    novelty_score=novelty_score_val,
                    features_12d=features_15d,
                    formula=formula, seed=genome.random_seed,
                )
                novel_count += 1
                generations_without_novelty = 0

                # Two-stage filter: only VLM-worthy patterns get rendered
                vlm_worthy = (
                    check_daily_limit(cfg)
                    and vlm_calls_this_gen < vlm_gen_cap
                    and should_call_vlm(features_15d, judged_features, novelty_score=novelty_score_val)
                )

                if vlm_worthy:
                    try:
                        position_history = []
                        genome.compile(cfg['gep']['bytecode_length'])
                        genome.compile_chemotaxis(cfg['gep']['bytecode_length'])
                        tree = decode_gene(genome.potential_gene, genome.head_length)
                        sim_step.set_potential(tree)
                        if genome.chemotaxis_gene:
                            from src.evolution.genome import decode_chemotaxis_gene
                            chemo_tree = decode_chemotaxis_gene(genome.chemotaxis_gene, genome.head_length)
                            sim_step.set_chemotaxis(chemo_tree)
                        particles.initialize(genome.random_seed)
                        environment.initialize(genome.random_seed)

                        feature_ts = {'speed_variance': [], 'survival_count': [], 'avg_energy': []}
                        for step in range(500):
                            sim_step.step(particles, environment)
                            if (step + 1) % 10 == 0:
                                px = particles.pos_x.to_numpy()
                                py = particles.pos_y.to_numpy()
                                vx = particles.vel_x.to_numpy()
                                vy = particles.vel_y.to_numpy()
                                alive = particles.alive.to_numpy()
                                energy = particles.energy.to_numpy()
                                position_history.append((px[alive == 1], py[alive == 1]))

                                # Collect feature time series
                                speeds = np.sqrt(vx[alive == 1]**2 + vy[alive == 1]**2)
                                feature_ts['speed_variance'].append((step, float(np.var(speeds))))
                                feature_ts['survival_count'].append((step, int(np.sum(alive))))
                                feature_ts['avg_energy'].append((step, float(np.mean(energy[alive == 1])) if np.sum(alive) > 0 else 0.0))

                        # Extract environment field snapshots for visualization
                        nutrient_np = environment.nutrient_field.to_numpy()
                        waste_np = environment.waste_field.to_numpy()

                        package = render_novelty_package(
                            genome, features_15d, position_history,
                            feature_ts, cfg, screenshot_dir,
                            nutrient_field=nutrient_np,
                            waste_field=waste_np,
                        )

                        # Compress image for VLM to reduce token consumption
                        compressed_path = compress_image_for_vlm(
                            package['trajectory_path'], max_size=128
                        )

                        novelty_packages.append({
                            'trajectory_path': compressed_path,
                            'summary': package['summary'],
                            'formula': genome.to_formula(),
                        })

                        # Track judged features for future filtering
                        if features_15d is not None:
                            judged_features.append(features_15d.tolist()
                                                    if hasattr(features_15d, 'tolist')
                                                    else list(features_15d))
                            # Keep last 200 entries to avoid unbounded growth
                            if len(judged_features) > 200:
                                judged_features = judged_features[-200:]

                        vlm_calls_this_gen += 1

                    except Exception as e:
                        logger.warning(f"Novelty rendering failed: {e}")

            fitnesses.append(fitness)

            # Save formula
            formula = genome.to_formula()
            formula_path = os.path.join(formula_dir, f'gen_{gen:06d}_{genome.get_id()[:8]}.txt')
            with open(formula_path, 'w') as f:
                f.write(f"Generation: {gen}\n")
                f.write(f"Fitness: {fitness:.6f}\n")
                f.write(f"Seed: {genome.random_seed}\n")
                f.write(f"Formula: {formula}\n")

        # ── Batched VLM evaluation (time-share VRAM) ──
        if novelty_packages:
            provider = cfg.get('vlm', {}).get('provider', 'local')
            if provider == 'local':
                # Pause Taichi → run VLM → resume Taichi
                with vlm_session(cfg):
                    for pkg in novelty_packages:
                        try:
                            vlm_response = vlm_judge(
                                pkg['trajectory_path'],
                                pkg['summary'],
                                cfg
                            )
                            logger.info(f"VLM [{pkg['formula'][:40]}...]: {vlm_response[:100]}...")
                        except Exception as e:
                            logger.warning(f"VLM judgment failed: {e}")

                # Reinitialize simulation components after Taichi reset
                particles = ParticleSystem(cfg)
                spatial_hash = SpatialHash(cfg)
                integrator = Integrator(cfg)
                sim_step = SimulationStep(spatial_hash, integrator, cfg)
                environment = EnvironmentLayer(cfg)
                sim_components = {
                    'particles': particles,
                    'spatial_hash': spatial_hash,
                    'integrator': integrator,
                    'step': sim_step,
                    'environment': environment,
                }
                logger.info("Simulation + environment components reinitialized after VLM session")
            else:
                # Ollama: no VRAM conflict, call directly
                for pkg in novelty_packages:
                    try:
                        vlm_response = vlm_judge(
                            pkg['trajectory_path'],
                            pkg['summary'],
                            cfg
                        )
                        logger.info(f"VLM [{pkg['formula'][:40]}...]: {vlm_response[:100]}...")
                    except Exception as e:
                        logger.warning(f"VLM judgment failed: {e}")

        # Update novelty streak
        monitor.update_novelty(novel_count > 0)
        if novel_count == 0:
            generations_without_novelty += 1

        # Periodic screenshot saving (every 10 generations for best genome)
        if (gen + 1) % 10 == 0 and population:
            try:
                # Find best genome from this generation
                best_genome = None
                best_fitness = -1
                for g in population:
                    if hasattr(g, 'fitness') and g.fitness > best_fitness:
                        best_fitness = g.fitness
                        best_genome = g

                if best_genome and best_fitness > 0:
                    # Lookup this genome's features (not from loop variable)
                    best_features = genome_features.get(best_genome.get_id(), np.zeros(15))

                    # Run simulation to get position history
                    best_genome.compile(cfg['gep']['bytecode_length'])
                    best_genome.compile_chemotaxis(cfg['gep']['bytecode_length'])
                    tree = decode_gene(best_genome.potential_gene, best_genome.head_length)
                    sim_step.set_potential(tree)
                    if best_genome.chemotaxis_gene:
                        from src.evolution.genome import decode_chemotaxis_gene
                        chemo_tree = decode_chemotaxis_gene(best_genome.chemotaxis_gene, best_genome.head_length)
                        sim_step.set_chemotaxis(chemo_tree)
                    particles.initialize(best_genome.random_seed)
                    environment.initialize(best_genome.random_seed)

                    position_history = []
                    feature_ts = {'speed_variance': [], 'survival_count': [], 'avg_energy': []}
                    for step in range(500):
                        sim_step.step(particles, environment)
                        if (step + 1) % 10 == 0:
                            px = particles.pos_x.to_numpy()
                            py = particles.pos_y.to_numpy()
                            vx = particles.vel_x.to_numpy()
                            vy = particles.vel_y.to_numpy()
                            alive = particles.alive.to_numpy()
                            energy = particles.energy.to_numpy()
                            position_history.append((px[alive == 1], py[alive == 1]))

                            speeds = np.sqrt(vx[alive == 1]**2 + vy[alive == 1]**2)
                            feature_ts['speed_variance'].append((step, float(np.var(speeds))))
                            feature_ts['survival_count'].append((step, int(np.sum(alive))))
                            feature_ts['avg_energy'].append((step, float(np.mean(energy[alive == 1])) if np.sum(alive) > 0 else 0.0))

                    # Extract environment field snapshots for visualization
                    nutrient_np = environment.nutrient_field.to_numpy()
                    waste_np = environment.waste_field.to_numpy()

                    # Render and save screenshot
                    package = render_novelty_package(
                        best_genome, best_features,
                        position_history, feature_ts, cfg, screenshot_dir,
                        nutrient_field=nutrient_np,
                        waste_field=waste_np,
                    )
                    logger.info(f"Periodic screenshot saved: gen={gen}, fitness={best_fitness:.4f}")
            except Exception as e:
                logger.warning(f"Periodic screenshot failed: {e}")

        # Generate next population
        new_population = []
        for _ in range(population_size):
            # Select parent
            if rng.random() < grid_selection_prob and me.grid:
                cell = me.random_non_empty_cell(rng)
                parent = cell['genome'] if cell else population[0]
            elif na.archive:
                entry = na.random_entry(rng)
                parent = entry['genome'] if entry else population[0]
            else:
                parent = population[0]

            # Mutate
            child = mutate(parent, cfg, rng)
            new_population.append(child)

        population = new_population

        # Log statistics
        gen_elapsed = time.time() - gen_start
        total_elapsed = time.time() - t_start
        vlm_stats = get_vlm_cache_stats()
        stats = {
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'novel_count': novel_count,
            'dead_count': dead_count,
            'grid_filled': me.get_filled_count(),
            'archive_size': na.size(),
            'elapsed_seconds': total_elapsed,
            'vlm_calls': vlm_calls_this_gen,
            'vlm_cached': vlm_stats['cached_results'],
        }
        log_generation_stats(gen, stats, logger)

        # Save to database
        db.log_generation(
            gen=gen,
            best_fit=stats['best_fitness'],
            avg_fit=stats['avg_fitness'],
            novel=stats['novel_count'],
            dead=stats['dead_count'],
            vlm=0,
            grid=stats['grid_filled'],
            archive=stats['archive_size'],
            elapsed=stats['elapsed_seconds'],
        )

        # Checkpoint
        if (gen + 1) % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{gen:06d}.pkl')
            rng_state = {
                'python_state': random.getstate(),
                'numpy_state': np.random.get_state(),
            }
            save_checkpoint(ckpt_path, gen, me, na, population, rng_state, cfg)
            cleanup_old_checkpoints(checkpoint_dir, max_keep)

        # Periodic visualization export (every 100 generations)
        if (gen + 1) % 100 == 0:
            try:
                from src.visualization.visualizer import (
                    visualize_evolution_curves, visualize_map_elites_3d,
                    export_formula_collection
                )
                export_dir = os.path.join(exp_dir, f'export_gen{gen:06d}')
                os.makedirs(export_dir, exist_ok=True)

                # Evolution curves from DB
                db_path_full = os.path.join(exp_dir, 'archive.db')
                visualize_evolution_curves(
                    db_path_full, cfg,
                    os.path.join(export_dir, 'evolution_curves.png')
                )

                # MAP-Elites 3D map
                visualize_map_elites_3d(
                    me, cfg,
                    os.path.join(export_dir, 'map_elites_3d.png')
                )

                # Formula collection
                export_formula_collection(
                    me, na,
                    os.path.join(export_dir, 'formula_collection.txt')
                )

                logger.info(f"Periodic export saved: {export_dir}")
            except Exception as e:
                logger.warning(f"Periodic export failed: {e}")

        # Health checks (with population for formula depth / NaN checks)
        health = monitor.check_all(population=population, sim_components=sim_components)
        if health['alerts']:
            for alert in health['alerts']:
                logger.warning(f"ALERT: {alert}")

        gen += 1

    # Final checkpoint on exit
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{gen:06d}.pkl')
    rng_state = {
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state(),
    }
    save_checkpoint(ckpt_path, gen, me, na, population, rng_state, cfg)
    logger.info(f"Evolution stopped at gen={gen}. Final checkpoint saved.")

    # Export results
    logger.info(f"\n{'='*60}")
    logger.info("Evolution Summary:")
    logger.info(f"  Total generations: {gen}")
    logger.info(f"  Grid filled: {me.get_filled_count()}/{me.get_total_cells()}")
    logger.info(f"  Archive size: {na.size()}")
    logger.info(f"  Elapsed time: {(time.time() - t_start)/3600:.2f} hours")
    logger.info(f"{'='*60}")

    return me, na


def main():
    parser = argparse.ArgumentParser(description='Hard Mode Emergence System')
    parser.add_argument('--config', default='config/default.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--mode', default='evolution',
                        choices=['phase1', 'evolution'],
                        help='Run mode: phase1 (basic sim) or evolution (full loop)')
    parser.add_argument('--generations', type=int, default=None,
                        help='Maximum generations to run')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info(f"Config loaded: {cfg['experiment']['name']}")

    # Init Taichi
    init_taichi(cfg, logger)

    if args.mode == 'phase1':
        # Run Phase 1 only
        pos_x, pos_y, cfg = run_phase1(cfg, logger)
        path = render_heatmap(pos_x, pos_y, cfg)
        logger.info(f"Heatmap saved: {path}")
    else:
        # Run full evolution
        run_evolution(cfg, logger, args.generations)


if __name__ == '__main__':
    main()
