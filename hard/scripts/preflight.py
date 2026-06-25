"""
Pre-flight check: validate config and code before launching evolution.

Usage: python scripts/preflight.py [--config config/fast_debug.yaml]
"""
import yaml
import sys
import os
import importlib

def check_config(cfg_path: str) -> bool:
    """Validate YAML config has all required sections and keys."""
    print(f"Checking config: {cfg_path}")

    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    ok = True

    # Required top-level sections
    required = ['experiment', 'simulation', 'world', 'environment', 'gep',
                'evolution', 'map_elites', 'novelty', 'safety', 'vlm',
                'checkpoint', 'monitoring']
    missing = [s for s in required if s not in cfg]
    if missing:
        print(f"  FAIL: Missing config sections: {missing}")
        ok = False
    else:
        print(f"  OK: {len(required)} config sections present")

    # Environment keys
    env_required = ['nutrient_diffuse_rate', 'nutrient_decay_rate', 'base_metabolism',
                    'absorb_rate', 'waste_production_rate', 'waste_diffuse_rate']
    env_missing = [k for k in env_required if k not in cfg.get('environment', {})]
    if env_missing:
        print(f"  FAIL: Missing environment keys: {env_missing}")
        ok = False
    else:
        print("  OK: Environment config complete")

    # Terminal set
    term_set = cfg.get('gep', {}).get('terminal_set', [])
    for t in ['avg_nutrient', 'avg_waste']:
        if t not in term_set:
            print(f"  FAIL: Missing terminal: {t}")
            ok = False
    chemo = cfg.get('gep', {}).get('terminal_set_chemotaxis', [])
    if not chemo:
        print("  FAIL: terminal_set_chemotaxis is empty")
        ok = False
    else:
        print(f"  OK: Terminal sets present (potential={len(term_set)}, chemo={len(chemo)})")

    # Novelty dim
    dim = cfg.get('novelty', {}).get('behavior_vector_dim', 0)
    if dim != 15:
        print(f"  FAIL: behavior_vector_dim={dim}, expected 15")
        ok = False
    else:
        print("  OK: Novelty dim = 15D")

    return ok


def check_imports() -> bool:
    """Verify all key modules import without errors."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("\nChecking imports...")
    modules = [
        'src.simulation.particles',
        'src.simulation.spatial_hash',
        'src.simulation.integrator',
        'src.simulation.step',
        'src.simulation.environment',
        'src.simulation.vm',
        'src.evolution.genome',
        'src.evolution.mutation',
        'src.evolution.gep',
        'src.evolution.features',
        'src.evolution.map_elites',
        'src.storage.db',
        'src.storage.checkpoint',
        'src.novelty.filter',
    ]
    ok = True
    for mod in modules:
        try:
            importlib.import_module(mod)
            print(f"  OK: {mod}")
        except Exception as e:
            print(f"  FAIL: {mod} -> {e}")
            ok = False
    return ok


def check_numpy_safety() -> bool:
    """Scan for numpy truthiness bugs."""
    print("\nChecking for numpy truthiness bugs...")
    issues = []
    for root, dirs, files in os.walk('src'):
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(root, f)
            with open(path, encoding='utf-8') as fh:
                lines = fh.readlines()
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                # Pattern: if <var>: where var is likely numpy array
                if any(stripped.startswith(f'if {v}') for v in
                       ['features_12d', 'features_15d', 'features_3d']):
                    if 'is not None' not in stripped and 'is None' not in stripped:
                        issues.append(f"  {path}:{i}: {stripped}")

    if issues:
        print("  WARN: Potential numpy truthiness issues:")
        for iss in issues:
            print(f"    {iss}")
        return False
    else:
        print("  OK: No numpy truthiness issues found")
        return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pre-flight checks')
    parser.add_argument('--config', default='config/fast_debug.yaml')
    args = parser.parse_args()

    results = []
    results.append(("Config", check_config(args.config)))
    results.append(("Imports", check_imports()))
    results.append(("Numpy safety", check_numpy_safety()))

    print(f"\n{'='*40}")
    all_ok = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll pre-flight checks passed. Ready to run!")
    else:
        print("\nSome checks failed. Fix issues before running.")
        sys.exit(1)
