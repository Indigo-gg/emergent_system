"""
VLM Judge: Call local Vision Language Model to evaluate novel emergent patterns.

Supports two providers:
- "local": Qwen2-VL-2B loaded via transformers with time-sharing VRAM
           (pause Taichi → release VRAM → load model → inference → unload → resume)
- "ollama": Ollama HTTP API fallback

Design: docs/hard-mode.md §VLM
"""

import os
import json
import base64
import logging
import gc
import time
import hashlib
from typing import Optional
from datetime import datetime, date

import requests

logger = logging.getLogger('hard-mode')

# Daily call counter (in-memory, resets on restart)
_daily_calls = {}
_daily_limit_key = date.today().isoformat()

# VLM result cache (formula_hash → response)
_vlm_cache = {}

# ── Local model singleton ──────────────────────────────────────────────────
_local_model = None
_local_processor = None
_local_model_name = None


def _load_local_model(model_name: str):
    """Load Qwen2-VL model into VRAM. Idempotent — skips if already loaded."""
    global _local_model, _local_processor, _local_model_name

    if _local_model is not None and _local_model_name == model_name:
        return

    import torch
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    logger.info(f"Loading VLM into VRAM: {model_name}")
    t0 = time.time()

    _local_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _local_processor = AutoProcessor.from_pretrained(model_name)
    _local_model_name = model_name

    elapsed = time.time() - t0
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info(f"VLM loaded in {elapsed:.1f}s, VRAM used: {vram_mb:.0f}MB")


def _unload_local_model():
    """Release VRAM back to Taichi simulation."""
    global _local_model, _local_processor, _local_model_name

    if _local_model is None:
        return

    logger.info("Unloading VLM from VRAM...")
    del _local_model
    del _local_processor
    _local_model = None
    _local_processor = None
    _local_model_name = None

    gc.collect()
    import torch
    torch.cuda.empty_cache()

    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info(f"VLM unloaded, VRAM remaining: {vram_mb:.0f}MB")


def _pause_taichi():
    """Reset Taichi runtime to free GPU memory for VLM."""
    try:
        import taichi as ti
        ti.reset()
        logger.info("Taichi runtime reset (VRAM released)")
    except Exception as e:
        logger.warning(f"Taichi reset failed: {e}")


def _resume_taichi(cfg: dict):
    """Re-initialize Taichi CUDA runtime after VLM inference."""
    try:
        import taichi as ti
        ti.init(arch=ti.cuda, debug=False)
        logger.info("Taichi runtime resumed (arch=cuda)")
    except Exception as e:
        logger.error(f"Taichi resume failed: {e}")
        raise


def _run_local_inference(image_path: str, prompt: str, model_name: str) -> str:
    """Run inference with locally-loaded Qwen2-VL model."""
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    _load_local_model(model_name)

    # Build message in Qwen2-VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process inputs
    text = _local_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _local_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_local_model.device)

    # Generate
    with torch.no_grad():
        generated_ids = _local_model.generate(**inputs, max_new_tokens=512)

    # Decode (trim input tokens)
    input_len = inputs.input_ids.shape[1]
    output_ids = generated_ids[:, input_len:]
    response = _local_processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


# ── Public API ─────────────────────────────────────────────────────────────

def vlm_judge(image_path: str, prompt: str,
              cfg: dict, model: str = None) -> str:
    """
    Call VLM to judge an image. Results are cached by prompt hash.

    Supports two providers (cfg.vlm.provider):
    - "local": Qwen2-VL-2B with time-sharing VRAM
    - "ollama": Ollama HTTP API

    Args:
        image_path: Path to image file
        prompt: Text prompt for the VLM
        cfg: Configuration dict with vlm settings
        model: Model name (overrides cfg)

    Returns:
        VLM response text
    """
    vlm_cfg = cfg.get('vlm', {})
    provider = vlm_cfg.get('provider', 'local')
    model = model or vlm_cfg.get('model', 'Qwen2-VL-2B')

    # Check daily limit
    if not check_daily_limit(cfg):
        return "VLM daily limit reached. Skipping judgment."

    # Cache lookup (by prompt content hash — avoids re-evaluating same formulas)
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    if cache_key in _vlm_cache:
        logger.debug(f"VLM cache hit for {cache_key[:8]}")
        return _vlm_cache[cache_key]

    if provider == 'local':
        response = _run_local_inference(image_path, prompt, model)
    elif provider == 'ollama':
        response = _call_ollama(image_path, prompt, cfg, model)
    else:
        return f"Error: Unknown VLM provider '{provider}'"

    # Cache result
    if not response.startswith("Error:"):
        _vlm_cache[cache_key] = response

    _increment_daily_count()
    logger.info(f"VLM judgment received ({len(response)} chars)")
    return response


def vlm_session(cfg):
    """
    Context manager for time-sharing VRAM between Taichi and VLM.

    Usage:
        # Save simulation components that need restoration
        with vlm_session(cfg):
            response = vlm_judge(image, prompt, cfg)
        # Taichi is back, simulation can resume

    The context manager:
    1. Pauses Taichi (releases GPU memory)
    2. Yields — caller runs VLM inference
    3. Unloads VLM model from VRAM
    4. Resumes Taichi
    """
    class _VLMSession:
        def __enter__(self):
            _pause_taichi()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            _unload_local_model()
            _resume_taichi(cfg)
            return False

    return _VLMSession()


def _call_ollama(image_path: str, prompt: str, cfg: dict, model: str) -> str:
    """Call Ollama HTTP API (legacy provider)."""
    vlm_cfg = cfg.get('vlm', {})
    base_url = vlm_cfg.get('base_url', 'http://localhost:11434')

    # Read and encode image
    try:
        with open(image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        return f"Error: Could not read image: {e}"

    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    except requests.exceptions.ConnectionError:
        logger.warning("Ollama connection failed. Is Ollama running?")
        return "Error: VLM service not available"
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out")
        return "Error: VLM request timed out"
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return f"Error: {e}"


# ── VLM Optimization ──────────────────────────────────────────────────────

def should_call_vlm(features_12d, judged_features: list,
                    novelty_score: float = 1.0,
                    min_novelty_gap: float = 0.3) -> bool:
    """
    Two-stage filter: decide whether a pattern warrants a VLM call.

    Uses cosine distance on 12D feature vectors to skip patterns
    too similar to already-evaluated ones.

    Args:
        features_12d: 12D feature vector of the candidate pattern
        judged_features: list of 12D vectors already sent to VLM
        novelty_score: novelty score from archive (higher = more novel)
        min_novelty_gap: minimum cosine distance threshold

    Returns:
        True if the pattern should be sent to VLM
    """
    if features_12d is None:
        return False

    import numpy as np

    # Always call VLM if no previous evaluations (first novel pattern)
    if not judged_features:
        return True

    # Always call VLM if novelty score is high (>3)
    if novelty_score >= 3.0:
        return True

    # Compute minimum cosine distance to already-judged patterns
    candidate = np.array(features_12d, dtype=np.float64)
    norm_candidate = np.linalg.norm(candidate)
    if norm_candidate < 1e-8:
        return False

    min_dist = float('inf')
    for prev in judged_features:
        prev_arr = np.array(prev, dtype=np.float64)
        norm_prev = np.linalg.norm(prev_arr)
        if norm_prev < 1e-8:
            continue
        cos_sim = np.dot(candidate, prev_arr) / (norm_candidate * norm_prev)
        cos_dist = 1.0 - cos_sim
        min_dist = min(min_dist, cos_dist)

    # Call VLM only if sufficiently different from all judged patterns
    return min_dist >= min_novelty_gap


def compress_image_for_vlm(image_path: str, max_size: int = 128) -> str:
    """
    Compress image for VLM input to reduce token consumption.

    Args:
        image_path: Original image path
        max_size: Maximum dimension (default 128px)

    Returns:
        Path to compressed image (same dir, _small suffix)
    """
    try:
        from PIL import Image

        img = Image.open(image_path)
        w, h = img.size

        # Resize if larger than max_size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # Save compressed version
        base, ext = os.path.splitext(image_path)
        compressed_path = f"{base}_small{ext}"
        img.save(compressed_path, quality=85)

        return compressed_path
    except Exception as e:
        logger.warning(f"Image compression failed: {e}")
        return image_path  # fallback to original


def get_vlm_cache_stats() -> dict:
    """Get VLM cache statistics."""
    return {
        'cached_results': len(_vlm_cache),
        'daily_calls': get_daily_call_count(),
    }


# ── Prompt builders ────────────────────────────────────────────────────────

def build_prompt(genome, features_12d, generation: int) -> str:
    """
    Build VLM prompt from genome data.

    Args:
        genome: GEP genome with formula
        features_12d: 12D feature vector
        generation: Current generation number

    Returns:
        Formatted prompt string
    """
    formula = genome.to_formula() if hasattr(genome, 'to_formula') else 'N/A'

    prompt = f"""You are a complex systems researcher analyzing particle simulations.

[Feature Time Series]

Discovered a novel emergent pattern in a particle simulation.

Potential Energy Formula:
  U = {formula}
  Force computed via F = -dU/dr (symbolic differentiation, conserves momentum)

Time-Invariant Features:
  Spatial Entropy Mean: {features_12d[0]:.4f} (0=ordered, 1=chaotic)
  Islands Mean: {features_12d[2]:.1f} (connected components)
  Speed Variance Mean: {features_12d[4]:.4f} (activity level)
  FFT Amp 1: {features_12d[5]:.4f} (oscillation strength)
  Angular Momentum Skew: {features_12d[8]:.4f} (rotation vs translation)
  Survival Rate: {features_12d[10]:.1%} (self-sustaining)
  Autocorrelation Lag-10: {features_12d[11]:.4f} (periodicity)

Generation: {generation}
Random Seed: {genome.random_seed if hasattr(genome, 'random_seed') else 'N/A'}

Please:
1. Give this pattern a short name (2-4 English words)
2. Describe its dynamic behavior (what is it doing, not static appearance)
3. Rate its novelty (1-5, 5 = never seen before)
4. Analyze the physics of the potential formula (why does U produce this behavior?)
5. Suggest a natural analog (similar to what physical/biological system?)

Respond in JSON format:
{{
  "name": "pattern name",
  "description": "dynamic behavior description",
  "novelty_score": 1-5,
  "physics_analysis": "why this formula produces this behavior",
  "natural_analog": "similar natural system"
}}"""

    return prompt


def build_novelty_package_prompt(summary: str) -> str:
    """
    Build prompt from pre-generated text summary.

    Args:
        summary: Text summary from renderer

    Returns:
        Formatted prompt string
    """
    return f"""You are a complex systems researcher analyzing particle simulations.

[Trajectory Image]
[Feature Time Series]

{summary}

Please:
1. Give this pattern a short name (2-4 English words)
2. Describe its dynamic behavior (what is it doing, not static appearance)
3. Rate its novelty (1-5, 5 = never seen before)
4. Analyze the physics of the potential formula
5. Suggest a natural analog

Respond in JSON format:
{{
  "name": "pattern name",
  "description": "dynamic behavior description",
  "novelty_score": 1-5,
  "physics_analysis": "analysis",
  "natural_analog": "similar system"
}}"""


# ── Daily limit ────────────────────────────────────────────────────────────

def check_daily_limit(cfg: dict) -> bool:
    """
    Check if VLM daily call limit has been reached.

    Args:
        cfg: Configuration dict with vlm.daily_limit

    Returns:
        True if calls are still allowed
    """
    global _daily_limit_key

    today = date.today().isoformat()
    if today != _daily_limit_key:
        _daily_limit_key = today
        _daily_calls.clear()

    daily_limit = cfg.get('vlm', {}).get('daily_limit', 100)
    current_count = _daily_calls.get(_daily_limit_key, 0)

    return current_count < daily_limit


def get_daily_call_count() -> int:
    """Get number of VLM calls made today."""
    today = date.today().isoformat()
    return _daily_calls.get(today, 0)


def _increment_daily_count():
    """Increment daily call counter."""
    today = date.today().isoformat()
    _daily_calls[today] = _daily_calls.get(today, 0) + 1


# ── Response parsing ───────────────────────────────────────────────────────

def parse_vlm_response(response: str) -> dict:
    """
    Parse VLM JSON response.

    Args:
        response: Raw VLM response text

    Returns:
        Parsed dict with name, description, novelty_score, etc.
    """
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return {
        "name": "Unknown",
        "description": response[:200],
        "novelty_score": 0,
        "physics_analysis": "",
        "natural_analog": "",
    }
