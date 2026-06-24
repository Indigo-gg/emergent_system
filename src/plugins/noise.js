import { Plugin } from '../plugin.js';

/**
 * Noise / thermal perturbation plugin.
 *
 * Adds random forces to particles, simulating thermal fluctuations.
 * Also adds random perturbation to phase, preventing lockstep.
 *
 * Temperature controls the magnitude:
 *   - temperature = 0: no noise, fully deterministic
 *   - temperature = 1: moderate noise
 *   - temperature = 2: high noise, system approaches chaos
 *
 * Uses a simple seeded PRNG (mulberry32 inline) for reproducibility.
 */
export class NoisePlugin extends Plugin {
  /**
   * @param {number} temperature - noise magnitude (0 = silent, 2 = chaotic)
   * @param {number} [seed] - PRNG seed for reproducibility
   */
  constructor(temperature = 0.5, seed = 12345) {
    super('noise', 0.3); // low weight: noise supplements, not dominates
    this.temperature = temperature;
    // Inline mulberry32 PRNG state
    this._seed = seed | 0;
  }

  /** Fast inline PRNG. Returns float in [0, 1). */
  _rng() {
    let s = (this._seed + 0x6d2b79f5) | 0;
    this._seed = s;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  propose(i, state, _spatialHash, _field, _config) {
    const T = this.temperature;
    if (T < 0.001) return; // skip if essentially zero

    // Random force perturbation
    const angle = this._rng() * Math.PI * 2;
    const magnitude = T * 0.02;
    state.fx[i] += Math.cos(angle) * magnitude;
    state.fy[i] += Math.sin(angle) * magnitude;

    // Random phase perturbation
    state.dPhase[i] += (this._rng() - 0.5) * T * 0.1;
  }
}
