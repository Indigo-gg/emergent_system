"""
Hard Mode Emergence System — Main Entry Point

Phase 0: Config loading + Taichi initialization + logging
Phase 1: Basic simulation loop with hardcoded potential
"""

import argparse
import logging
import os
import sys
import time

import yaml
import taichi as ti
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
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
    logger.info(f"Taichi initialized: arch=cuda, device={ti.lang.impl.get_runtime().get_current_device()}")


def run_phase1(cfg: dict, logger: logging.Logger):
    """Run Phase 1: basic simulation with hardcoded potential."""
    from src.simulation.particles import ParticleSystem
    from src.simulation.spatial_hash import SpatialHash
    from src.simulation.integrator import Integrator
    from src.simulation.step import SimulationStep
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

    # Initialize particles
    particles.initialize(seed)
    logger.info("Particles initialized")

    # Database
    db_path = os.path.join('data', 'experiments', cfg['experiment']['name'],
                           'archive.db')
    db = ExperimentDB(db_path)

    # Run simulation
    logger.info(f"Running {steps} steps...")
    t_start = time.time()

    for step in range(steps):
        sim_step.step(particles)

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


def main():
    parser = argparse.ArgumentParser(description='Hard Mode Emergence System')
    parser.add_argument('--config', default='config/default.yaml',
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    logger = setup_logging(cfg)
    logger.info(f"Config loaded: {cfg['experiment']['name']}")

    # Init Taichi
    init_taichi(cfg, logger)

    # Run Phase 1
    pos_x, pos_y, cfg = run_phase1(cfg, logger)

    # Render heatmap
    path = render_heatmap(pos_x, pos_y, cfg)
    logger.info(f"Heatmap saved: {path}")


if __name__ == '__main__':
    main()
