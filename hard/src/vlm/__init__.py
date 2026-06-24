"""
VLM (Vision Language Model) module for judging novel emergent patterns.
"""

from src.vlm.judge import (
    vlm_judge, vlm_session, build_prompt, check_daily_limit,
    should_call_vlm, compress_image_for_vlm, get_vlm_cache_stats
)

__all__ = [
    'vlm_judge', 'vlm_session', 'build_prompt', 'check_daily_limit',
    'should_call_vlm', 'compress_image_for_vlm', 'get_vlm_cache_stats'
]
