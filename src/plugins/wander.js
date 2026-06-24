import { Plugin } from '../plugin.js';

/**
 * Wander plugin — adds a small random drift each tick.
 * Prevents particles from freezing into static equilibrium.
 * The randomness comes from the seeded PRNG, so it's reproducible.
 */
export class WanderPlugin extends Plugin {
  constructor(rng) {
    super('wander', 0.5);
    this.rng = rng;
    this._angle = new Float32Array(0);
  }

  reset() {
    // No per-tick state needed — we use per-particle random angles
  }

  propose(i, state, _spatialHash, _field, config) {
    // Lazy-init angle buffer
    if (this._angle.length <= i) {
      const buf = new Float32Array(state.maxCount);
      buf.set(this._angle);
      this._angle = buf;
    }

    // Each particle gets a slowly drifting random angle
    this._angle[i] += (this.rng() - 0.5) * 0.3;

    const wanderStrength = 0.005;
    state.fx[i] += Math.cos(this._angle[i]) * wanderStrength;
    state.fy[i] += Math.sin(this._angle[i]) * wanderStrength;
  }
}
