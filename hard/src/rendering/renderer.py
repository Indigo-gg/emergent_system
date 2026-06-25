"""
Renderer: Visualization functions for particle simulations.

Provides:
- Density heatmap rendering
- Trajectory overlay (500 frames, time-encoded colors)
- Feature time series curves
- GIF animation generation
"""

import os
import numpy as np
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def render_density_heatmap(pos_x: np.ndarray, pos_y: np.ndarray,
                           cfg: dict, output_path: str = None) -> str:
    """
    Render a 2D density heatmap of particle positions.

    Args:
        pos_x: Particle x positions
        pos_y: Particle y positions
        cfg: Configuration dict with world.width, world.height, rendering.resolution
        output_path: Output file path (auto-generated if None)

    Returns:
        Path to saved image
    """
    w = cfg['world']['width']
    h = cfg['world']['height']
    res = cfg.get('rendering', {}).get('resolution', [256, 256])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist2d(pos_x, pos_y, bins=res, range=[[0, w], [0, h]],
              cmap='inferno', density=True)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.set_title(f'Density Heatmap ({len(pos_x)} particles)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if output_path is None:
        output_path = 'heatmap.png'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def render_trajectory(position_history: List[Tuple[np.ndarray, np.ndarray]],
                      cfg: dict, output_path: str = None,
                      max_frames: int = 500) -> str:
    """
    Render trajectory overlay with time-encoded colors.

    Overlays multiple frames of particle positions, coloring by time
    (older = blue, newer = red).

    Args:
        position_history: List of (pos_x, pos_y) tuples per frame
        cfg: Configuration dict
        output_path: Output file path
        max_frames: Maximum frames to overlay

    Returns:
        Path to saved image
    """
    w = cfg['world']['width']
    h = cfg['world']['height']

    # Subsample if too many frames
    if len(position_history) > max_frames:
        indices = np.linspace(0, len(position_history) - 1, max_frames, dtype=int)
        position_history = [position_history[i] for i in indices]

    n_frames = len(position_history)

    # Create time-encoded colormap (blue → red)
    cmap = LinearSegmentedColormap.from_list('time', ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.set_facecolor('black')

    # Plot each frame with time-based color
    for i, (px, py) in enumerate(position_history):
        color = cmap(i / max(n_frames - 1, 1))
        alpha = 0.1 + 0.4 * (i / max(n_frames - 1, 1))  # Fade older frames
        ax.scatter(px, py, c=[color], alpha=alpha, s=0.5, edgecolors='none')

    ax.set_title(f'Trajectory Overlay ({n_frames} frames)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_frames))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time (frame)')

    if output_path is None:
        output_path = 'trajectory.png'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def render_feature_curves(feature_timeseries: dict, cfg: dict,
                          output_path: str = None) -> str:
    """
    Render time series of simulation features.

    Args:
        feature_timeseries: Dict with keys like 'speed_variance', 'entropy', 'angular_momentum'
                           Each value is a list of (step, value) tuples
        cfg: Configuration dict
        output_path: Output file path

    Returns:
        Path to saved image
    """
    n_features = len(feature_timeseries)
    if n_features == 0:
        return None

    fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features), squeeze=False)

    for idx, (name, data) in enumerate(feature_timeseries.items()):
        ax = axes[idx, 0]
        if data:
            steps, values = zip(*data)
            ax.plot(steps, values, linewidth=1.5)
            ax.set_ylabel(name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.set_xlim(steps[0], steps[-1])

    axes[-1, 0].set_xlabel('Simulation Step')
    fig.suptitle('Feature Time Series', fontsize=14)
    plt.tight_layout()

    if output_path is None:
        output_path = 'feature_curves.png'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def render_gif(position_history: List[Tuple[np.ndarray, np.ndarray]],
               cfg: dict, output_path: str = None,
               fps: int = 10, max_frames: int = 200) -> str:
    """
    Render animated GIF of particle simulation.

    Args:
        position_history: List of (pos_x, pos_y) tuples per frame
        cfg: Configuration dict
        output_path: Output file path
        fps: Frames per second
        max_frames: Maximum frames to include

    Returns:
        Path to saved GIF
    """
    w = cfg['world']['width']
    h = cfg['world']['height']
    res = cfg.get('rendering', {}).get('resolution', [256, 256])

    # Subsample if too many frames
    if len(position_history) > max_frames:
        indices = np.linspace(0, len(position_history) - 1, max_frames, dtype=int)
        position_history = [position_history[i] for i in indices]

    frames = []
    for px, py in position_history:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist2d(px, py, bins=res, range=[[0, w], [0, h]],
                  cmap='inferno', density=True, vmin=0, vmax=0.01)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')
        ax.axis('off')

        fig.canvas.draw()
        # Convert to numpy array
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    if output_path is None:
        output_path = 'animation.gif'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save as GIF using PIL
    try:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in frames]
        duration = int(1000 / fps)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
    except ImportError:
        # Fallback: save as PNG if PIL not available
        if frames:
            from PIL import Image
            Image.fromarray(frames[0]).save(output_path.replace('.gif', '.png'))

    return output_path


def render_environment_overlay(nutrient_field: np.ndarray, waste_field: np.ndarray,
                               position_history: List[Tuple[np.ndarray, np.ndarray]],
                               cfg: dict, output_path: str = None) -> str:
    """
    Render environment fields with particle trajectory overlay.

    Layers:
    - Bottom: nutrient field heatmap (Greens, alpha=0.3)
    - Middle: waste field heatmap (Reds, alpha=0.3)
    - Top: particle trajectory (time-encoded colors, alpha=0.6)

    Args:
        nutrient_field: 2D numpy array of nutrient concentrations
        waste_field: 2D numpy array of waste concentrations
        position_history: List of (pos_x, pos_y) tuples per frame
        cfg: Configuration dict
        output_path: Output file path

    Returns:
        Path to saved image
    """
    w = cfg['world']['width']
    h = cfg['world']['height']
    rows, cols = nutrient_field.shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.set_facecolor('black')

    # Layer 1: Nutrient field (green)
    nut_norm = nutrient_field / max(np.max(nutrient_field), 1e-10)
    ax.imshow(nut_norm.T, origin='lower', extent=[0, w, 0, h],
              cmap='Greens', alpha=0.5, interpolation='bilinear')

    # Layer 2: Waste field (red)
    waste_norm = waste_field / max(np.max(waste_field), 1e-10)
    ax.imshow(waste_norm.T, origin='lower', extent=[0, w, 0, h],
              cmap='Reds', alpha=0.4, interpolation='bilinear')

    # Layer 3: Particle trajectory overlay
    if position_history:
        n_frames = len(position_history)
        cmap = LinearSegmentedColormap.from_list('time', ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF0000'])
        for i, (px, py) in enumerate(position_history):
            color = cmap(i / max(n_frames - 1, 1))
            alpha = 0.1 + 0.4 * (i / max(n_frames - 1, 1))
            ax.scatter(px, py, c=[color], alpha=alpha, s=0.5, edgecolors='none')

    ax.set_title('Environment + Trajectory Overlay')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if output_path is None:
        output_path = 'env_overlay.png'

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def render_novelty_package(genome, features_12d: np.ndarray,
                           position_history: List[Tuple[np.ndarray, np.ndarray]],
                           feature_timeseries: dict, cfg: dict,
                           output_dir: str,
                           nutrient_field: np.ndarray = None,
                           waste_field: np.ndarray = None) -> dict:
    """
    Render the complete novelty package for VLM judgment.

    Generates:
    1. Trajectory overlay image
    2. Feature time series curves
    3. Text summary

    Args:
        genome: The GEP genome
        features_12d: 12D feature vector
        position_history: List of (pos_x, pos_y) tuples
        feature_timeseries: Dict of feature time series
        cfg: Configuration dict
        output_dir: Output directory

    Returns:
        Dict with paths to generated files and text summary
    """
    os.makedirs(output_dir, exist_ok=True)

    gen = genome.generation if hasattr(genome, 'generation') else 0
    genome_id = genome.get_id() if hasattr(genome, 'get_id') else 'unknown'

    # 1. Trajectory overlay
    traj_path = os.path.join(output_dir, f'gen_{gen:04d}_{genome_id[:8]}_traj.png')
    render_trajectory(position_history, cfg, traj_path)

    # 2. Feature curves
    curve_path = os.path.join(output_dir, f'gen_{gen:04d}_{genome_id[:8]}_curves.png')
    render_feature_curves(feature_timeseries, cfg, curve_path)

    # 3. Environment overlay (if data provided)
    env_overlay_path = None
    if nutrient_field is not None and waste_field is not None:
        env_overlay_path = os.path.join(output_dir, f'gen_{gen:04d}_{genome_id[:8]}_env.png')
        render_environment_overlay(nutrient_field, waste_field,
                                   position_history, cfg, env_overlay_path)

    # 4. Text summary
    formula = genome.to_formula() if hasattr(genome, 'to_formula') else 'N/A'
    summary = _build_text_summary(genome, features_12d, formula, gen)

    return {
        'trajectory_path': traj_path,
        'curve_path': curve_path,
        'env_overlay_path': env_overlay_path,
        'summary': summary,
        'generation': gen,
        'genome_id': genome_id,
    }


def _build_text_summary(genome, features_12d: np.ndarray,
                        formula: str, generation: int) -> str:
    """Build text summary for VLM judgment."""
    lines = [
        "Discovered a novel emergent pattern.",
        "",
        "Potential Energy Formula:",
        f"  U = {formula}",
        "  Force computed via F = -dU/dr (symbolic differentiation)",
        "",
        "Time-Invariant Features:",
        f"  Spatial Entropy Mean: {features_12d[0]:.4f}",
        f"  Islands Mean: {features_12d[2]:.1f}",
        f"  Speed Variance Mean: {features_12d[4]:.4f}",
        f"  FFT Amp 1: {features_12d[5]:.4f}",
        f"  Angular Momentum Skew: {features_12d[8]:.4f}",
        f"  Survival Rate: {features_12d[10]:.1%}",
        f"  Autocorrelation Lag-10: {features_12d[11]:.4f}",
        "",
        f"Generation: {generation}",
        f"Random Seed: {genome.random_seed if hasattr(genome, 'random_seed') else 'N/A'}",
    ]
    return '\n'.join(lines)
