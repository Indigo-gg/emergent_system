import { Plugin } from '../plugin.js';

/**
 * Repulsor plugin — pushes particles away from nearby neighbors.
 * This is the minimal "emergence catalyst": attraction + repulsion
 * creates self-organizing clusters that don't collapse.
 */
export class RepulsorPlugin extends Plugin {
  constructor() {
    super('repulsor', 2.0); // higher weight = resolves before attractor
  }

  propose(i, state, spatialHash, _field, config) {
    const r = config.repulsorRadius;
    const r2 = r * r;
    const strength = config.repulsorStrength;

    let fx = 0;
    let fy = 0;

    spatialHash.queryState(state.x[i], state.y[i], r, state, (j, d2) => {
      const dx = state.x[i] - state.x[j];
      const dy = state.y[i] - state.y[j];
      const dist = Math.sqrt(d2);

      // Force is stronger the closer the neighbor
      const force = (1 - dist / r) * strength;
      fx += (dx / dist) * force;
      fy += (dy / dist) * force;
    });

    state.fx[i] += fx;
    state.fy[i] += fy;
  }
}
