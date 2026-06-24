"""
Monitor: System health monitoring for long-running evolution.

Provides:
- GPU temperature monitoring
- GPU VRAM monitoring
- Disk space monitoring
- Evolution staleness detection
- Formula depth monitoring
- NaN ratio monitoring
- Per-generation statistics logging
"""

import os
import shutil
import logging
import subprocess
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger('hard-mode')


def check_gpu_temp(threshold: float = 85.0) -> Dict[str, Any]:
    """
    Check GPU temperature using nvidia-smi.

    Args:
        threshold: Temperature threshold in Celsius

    Returns:
        Dict with 'temp', 'ok', 'throttle' keys
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            temp = float(result.stdout.strip())
            return {
                'temp': temp,
                'ok': temp < threshold,
                'throttle': temp >= threshold,
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # Fallback: assume OK if can't detect
    return {'temp': 0, 'ok': True, 'throttle': False}


def check_gpu_vram(threshold_gb: float = 7.5) -> Dict[str, Any]:
    """
    Check GPU VRAM usage using nvidia-smi.

    Args:
        threshold_gb: VRAM usage threshold in GB

    Returns:
        Dict with 'used_gb', 'total_gb', 'ok', 'warning' keys
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            used_mb = float(parts[0].strip())
            total_mb = float(parts[1].strip())
            used_gb = used_mb / 1024
            total_gb = total_mb / 1024
            return {
                'used_gb': round(used_gb, 2),
                'total_gb': round(total_gb, 2),
                'ok': used_gb < threshold_gb,
                'warning': used_gb >= threshold_gb,
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return {'used_gb': 0, 'total_gb': 0, 'ok': True, 'warning': False}


def check_disk_space(min_gb: float = 5.0, path: str = '.') -> Dict[str, Any]:
    """
    Check available disk space.

    Args:
        min_gb: Minimum required space in GB
        path: Path to check (default: current directory)

    Returns:
        Dict with 'free_gb', 'ok', 'warning' keys
    """
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        return {
            'free_gb': round(free_gb, 2),
            'ok': free_gb >= min_gb,
            'warning': free_gb < min_gb,
        }
    except Exception:
        return {'free_gb': 0, 'ok': True, 'warning': False}


def check_staleness(generations_without_novelty: int,
                    stale_limit: int = 10) -> Dict[str, Any]:
    """
    Check if evolution has stagnated.

    Args:
        generations_without_novelty: Count of consecutive generations without novel findings
        stale_limit: Number of generations before considering stale

    Returns:
        Dict with 'stale', 'count', 'action' keys
    """
    stale = generations_without_novelty >= stale_limit
    action = None

    if stale:
        action = "Lower novelty threshold by 20%"

    return {
        'stale': stale,
        'count': generations_without_novelty,
        'limit': stale_limit,
        'action': action,
    }


def check_formula_depth(population: list, max_depth: int = 20) -> Dict[str, Any]:
    """
    Check formula tree depth for bloat detection.

    Args:
        population: List of GEPGenome objects
        max_depth: Maximum allowed tree depth

    Returns:
        Dict with 'max_depth', 'avg_depth', 'ok', 'warning' keys
    """
    if not population:
        return {'max_depth': 0, 'avg_depth': 0, 'ok': True, 'warning': False}

    depths = []
    for genome in population:
        try:
            from src.evolution.genome import decode_gene
            tree = decode_gene(genome.potential_gene, genome.head_length)
            depths.append(tree.size())
        except Exception:
            depths.append(0)

    max_d = max(depths) if depths else 0
    avg_d = sum(depths) / len(depths) if depths else 0

    return {
        'max_depth': max_d,
        'avg_depth': round(avg_d, 2),
        'ok': max_d <= max_depth,
        'warning': max_d > max_depth,
    }


def check_nan_ratio(population: list, sim_components: dict, cfg: dict,
                    sample_size: int = 5) -> Dict[str, Any]:
    """
    Check ratio of genomes producing NaN/Inf in simulation.

    Args:
        population: List of GEPGenome objects
        sim_components: Simulation components dict
        cfg: Configuration dict
        sample_size: Number of genomes to sample for NaN check

    Returns:
        Dict with 'nan_ratio', 'ok', 'warning' keys
    """
    if not population:
        return {'nan_ratio': 0.0, 'ok': True, 'warning': False}

    import random as _random
    sample = _random.sample(population, min(sample_size, len(population)))

    nan_count = 0
    for genome in sample:
        try:
            genome.compile(cfg['gep']['bytecode_length'])
            consts = genome._potential_constants
            if any(__import__('math').isnan(c) or __import__('math').isinf(c) for c in consts):
                nan_count += 1
        except Exception:
            nan_count += 1

    ratio = nan_count / len(sample) if sample else 0.0

    return {
        'nan_ratio': round(ratio, 3),
        'ok': ratio < 0.5,
        'warning': ratio >= 0.5,
    }


def log_generation_stats(gen: int, stats: Dict[str, Any],
                         logger_instance: logging.Logger = None):
    """
    Log per-generation statistics.

    Args:
        gen: Generation number
        stats: Dict with generation statistics
        logger_instance: Logger to use (default: module logger)
    """
    log = logger_instance or logger

    best_fit = stats.get('best_fitness', 0)
    avg_fit = stats.get('avg_fitness', 0)
    novel_count = stats.get('novel_count', 0)
    dead_count = stats.get('dead_count', 0)
    grid_filled = stats.get('grid_filled', 0)
    archive_size = stats.get('archive_size', 0)
    elapsed = stats.get('elapsed_seconds', 0)
    vlm_calls = stats.get('vlm_calls', 0)
    vlm_cached = stats.get('vlm_cached', 0)

    # Format elapsed time
    if elapsed > 86400:
        elapsed_str = f"{elapsed/86400:.1f}d"
    elif elapsed > 3600:
        elapsed_str = f"{elapsed/3600:.1f}h"
    elif elapsed > 60:
        elapsed_str = f"{elapsed/60:.1f}m"
    else:
        elapsed_str = f"{elapsed:.0f}s"

    vlm_str = f"vlm={vlm_calls}" if vlm_calls else ""
    if vlm_cached:
        vlm_str += f"(cache={vlm_cached})"

    log.info(
        f"gen={gen:6d} | fit={best_fit:.4f} avg={avg_fit:.4f} | "
        f"novel={novel_count} dead={dead_count} | "
        f"grid={grid_filled} archive={archive_size} | "
        f"{vlm_str} | "
        f"elapsed={elapsed_str}"
    )


class EvolutionMonitor:
    """
    Monitors evolution health and triggers alerts.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.monitoring_cfg = cfg.get('monitoring', {})

        self.gpu_threshold = self.monitoring_cfg.get('gpu_temp_threshold', 85)
        self.vram_threshold = self.monitoring_cfg.get('vram_threshold_gb', 7.5)
        self.disk_min_gb = self.monitoring_cfg.get('disk_min_gb', 5)
        self.stale_limit = self.monitoring_cfg.get('stale_generations', 10)
        self.formula_max_depth = self.monitoring_cfg.get('formula_max_depth', 20)

        self.start_time = datetime.now()
        self.generation_count = 0
        self.novelty_streak = 0  # Generations without novelty

    def check_all(self, population=None, sim_components=None) -> Dict[str, Any]:
        """
        Run all health checks.

        Args:
            population: Optional list of genomes for formula depth / NaN checks
            sim_components: Optional sim components for NaN ratio check

        Returns:
            Dict with all check results
        """
        results = {
            'gpu_temp': check_gpu_temp(self.gpu_threshold),
            'gpu_vram': check_gpu_vram(self.vram_threshold),
            'disk': check_disk_space(self.disk_min_gb),
            'staleness': check_staleness(self.novelty_streak, self.stale_limit),
        }

        # Optional deep checks (only when population provided)
        if population:
            results['formula_depth'] = check_formula_depth(population, self.formula_max_depth)
            if sim_components:
                results['nan_ratio'] = check_nan_ratio(population, sim_components, self.cfg)

        # Check for alerts
        alerts = []
        if not results['gpu_temp']['ok']:
            alerts.append(f"GPU temperature high: {results['gpu_temp']['temp']}°C")
        if not results['gpu_vram']['ok']:
            alerts.append(f"VRAM high: {results['gpu_vram']['used_gb']}/{results['gpu_vram']['total_gb']} GB")
        if not results['disk']['ok']:
            alerts.append(f"Low disk space: {results['disk']['free_gb']} GB")
        if results['staleness']['stale']:
            alerts.append(f"Evolution stale: {self.novelty_streak} generations without novelty")
        if results.get('formula_depth', {}).get('warning'):
            alerts.append(f"Formula bloat: max depth {results['formula_depth']['max_depth']}")
        if results.get('nan_ratio', {}).get('warning'):
            alerts.append(f"High NaN ratio: {results['nan_ratio']['nan_ratio']:.1%}")

        results['alerts'] = alerts
        results['ok'] = len(alerts) == 0

        return results

    def update_novelty(self, found_novel: bool):
        """Update novelty streak counter."""
        if found_novel:
            self.novelty_streak = 0
        else:
            self.novelty_streak += 1

    def get_elapsed_seconds(self) -> float:
        """Get elapsed time since monitor started."""
        return (datetime.now() - self.start_time).total_seconds()
