import { Plugin } from '../plugin.js';

/**
 * Kuramoto synchronization plugin.
 *
 * Each particle has an internal phase that tends to synchronize with
 * its neighbors' phases. The coupling strength determines how strongly
 * particles influence each other's phase.
 *
 * This is the core mechanism for emergence of collective oscillation:
 * at low coupling, phases are random (chaos); above a critical coupling
 * threshold, spontaneous synchronization emerges (phase transition).
 *
 * Physics: dθ_i/dt = ω_i + (K / N_neighbors) * Σ sin(θ_j - θ_i)
 * We write the phase delta to state.dPhase[i], which the engine integrates.
 */
export class KuramotoPlugin extends Plugin {
  /**
   * @param {number} couplingStrength - K, how strongly particles sync
   * @param {number} perceptionRadius - radius to look for neighbors
   */
  constructor(couplingStrength = 1.0, perceptionRadius = 60) {
    super('kuramoto', 1.0);
    this.couplingStrength = couplingStrength;
    this.perceptionRadius = perceptionRadius;
  }

  propose(i, state, spatialHash, _field, config) {
    const K = this.couplingStrength;
    const radius = this.perceptionRadius;
    const phaseI = state.phase[i];

    let sumSin = 0;
    let neighborCount = 0;

    spatialHash.queryState(state.x[i], state.y[i], radius, state, (j, _d2) => {
      sumSin += Math.sin(state.phase[j] - phaseI);
      neighborCount++;
    });

    if (neighborCount === 0) return;

    // Kuramoto update: dθ = (K / N) * Σ sin(θ_j - θ_i)
    state.dPhase[i] += (K / neighborCount) * sumSin;
  }
}
