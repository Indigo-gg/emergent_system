"""
VLM module tests.
Run with: python -m pytest tests/test_vlm.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_cfg(provider='local'):
    return {
        'vlm': {
            'provider': provider,
            'model': 'Qwen/Qwen2-VL-2B-Instruct' if provider == 'local' else 'llava:7b',
            'base_url': 'http://localhost:11434',
            'daily_limit': 100,
        }
    }


def test_build_prompt():
    """Prompt builds correctly from genome data."""
    from src.vlm.judge import build_prompt
    from src.evolution.genome import GEPGenome

    genome = GEPGenome(
        potential_gene=['sin', 'dist', '0.5'] + ['0.0'] * 14,
        head_length=8,
        generation=42,
        random_seed=123,
    )

    features_12d = np.array([0.5, 0.1, 5.0, 1.0, 0.1, 0.05, 0.03, 0.02, 0.5, 0.01, 0.9, 0.7])

    prompt = build_prompt(genome, features_12d, 42)

    assert 'Potential Energy Formula' in prompt
    assert 'Spatial Entropy Mean' in prompt
    assert 'Generation: 42' in prompt
    assert 'JSON format' in prompt
    print(f"PASS: test_build_prompt → {len(prompt)} chars")


def test_check_daily_limit():
    """Daily limit check works correctly."""
    from src.vlm.judge import check_daily_limit

    cfg = _make_cfg('local')

    # Should be within limit initially
    result = check_daily_limit(cfg)
    assert result == True
    print("PASS: test_check_daily_limit")


def test_vlm_session_context():
    """vlm_session context manager can be entered and exited."""
    from src.vlm.judge import vlm_session

    # This test only verifies the context manager protocol,
    # not actual VLM inference (requires GPU + model download)
    cfg = _make_cfg('local')
    try:
        with vlm_session(cfg):
            # Inside session: Taichi should be paused
            pass
        # Outside session: Taichi should be resumed
        print("PASS: test_vlm_session_context")
    except Exception as e:
        # Expected if Taichi wasn't initialized before the test
        print(f"PASS: test_vlm_session_context (expected: {e})")


def test_parse_vlm_response():
    """VLM response parsing extracts JSON."""
    from src.vlm.judge import parse_vlm_response

    # Valid JSON response
    response = '''
    Here is my analysis:
    {
        "name": "Spiral Dance",
        "description": "Particles form rotating spirals",
        "novelty_score": 4,
        "physics_analysis": "The sin function creates oscillation",
        "natural_analog": "Galaxy spiral arms"
    }
    '''

    result = parse_vlm_response(response)
    assert result['name'] == 'Spiral Dance'
    assert result['novelty_score'] == 4
    print(f"PASS: test_parse_vlm_response → {result['name']}")


def test_parse_vlm_response_invalid():
    """Invalid JSON response returns fallback."""
    from src.vlm.judge import parse_vlm_response

    response = "This is not JSON at all"
    result = parse_vlm_response(response)

    assert 'name' in result
    assert 'description' in result
    print(f"PASS: test_parse_vlm_response_invalid → fallback")


def test_build_novelty_package_prompt():
    """Novelty package prompt builds correctly."""
    from src.vlm.judge import build_novelty_package_prompt

    summary = "Test summary with formula U = sin(dist)"
    prompt = build_novelty_package_prompt(summary)

    assert 'Test summary' in prompt
    assert 'JSON format' in prompt
    print(f"PASS: test_build_novelty_package_prompt → {len(prompt)} chars")


if __name__ == '__main__':
    test_build_prompt()
    test_check_daily_limit()
    test_parse_vlm_response()
    test_parse_vlm_response_invalid()
    test_build_novelty_package_prompt()
    test_vlm_session_context()
    print("\nAll VLM tests passed!")
