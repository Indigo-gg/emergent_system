/**
 * Random sampling strategy — Latin Hypercube Sampling (LHS).
 *
 * LHS divides each parameter's range into N equal intervals and samples
 * exactly once from each interval. This guarantees much better coverage
 * of the parameter space than pure random sampling, especially in
 * high-dimensional spaces with few samples.
 *
 * With 20 samples in 9 dimensions, uniform random would cluster badly.
 * LHS ensures every parameter value range is represented.
 */
import { PARAM_SPACE, DEFAULTS } from '../param-space.js';

export class RandomStrategy {
  constructor(paramSpace = PARAM_SPACE, rng = Math.random) {
    this.paramSpace = paramSpace;
    this.rng = rng;
    this._lhsSamples = null;
    this._sampleIndex = 0;
  }

  /** Generate next parameter set using LHS. */
  next() {
    if (!this._lhsSamples) {
      this._lhsSamples = this._generateLHS(20); // 20 samples for Phase A
    }

    if (this._sampleIndex >= this._lhsSamples.length) {
      // Fallback to random if we exceed pre-generated LHS samples
      return this._randomParams();
    }

    return this._lhsSamples[this._sampleIndex++];
  }

  /**
   * Generate Latin Hypercube Samples.
   * For each parameter, divide range into n intervals, shuffle, then sample.
   */
  _generateLHS(n) {
    const keys = Object.keys(this.paramSpace).filter(k => this.paramSpace[k].category !== 'render');
    const samples = [];

    // For each parameter, create a shuffled permutation of interval indices
    const perms = {};
    for (const key of keys) {
      perms[key] = this._shuffle(Array.from({ length: n }, (_, i) => i));
    }

    for (let i = 0; i < n; i++) {
      const params = {};
      for (const key of keys) {
        const spec = this.paramSpace[key];
        const intervalIdx = perms[key][i];
        // Sample uniformly within the assigned interval
        const intervalWidth = (spec.max - spec.min) / n;
        const raw = spec.min + (intervalIdx + this.rng()) * intervalWidth;
        // Align to step
        params[key] = Math.round(raw / spec.step) * spec.step;
        // Clamp
        params[key] = Math.max(spec.min, Math.min(spec.max, params[key]));
      }

      // Fill render params with defaults
      for (const [key, spec] of Object.entries(this.paramSpace)) {
        if (spec.category === 'render') {
          params[key] = DEFAULTS[key] ?? (spec.min + spec.max) / 2;
        }
      }

      samples.push(params);
    }

    return samples;
  }

  /** Fisher-Yates shuffle (in-place, returns same array). */
  _shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /** Pure random fallback. */
  _randomParams() {
    const params = {};
    for (const [key, spec] of Object.entries(this.paramSpace)) {
      if (spec.category === 'render') {
        params[key] = DEFAULTS[key] ?? (spec.min + spec.max) / 2;
        continue;
      }
      const range = spec.max - spec.min;
      const steps = Math.floor(range / spec.step);
      const randomStep = Math.floor(this.rng() * (steps + 1));
      params[key] = spec.min + randomStep * spec.step;
      params[key] = Math.round(params[key] / spec.step) * spec.step;
    }
    return params;
  }
}
