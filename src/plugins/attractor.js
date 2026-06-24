import { Plugin } from '../plugin.js';

/**
 * Attractor plugin — pulls particles toward a target point.
 * Demonstrates the simplest possible plugin: one desire, no neighbors needed.
 */
export class AttractorPlugin extends Plugin {
  constructor() {
    super('attractor', 1.0);
  }

  propose(i, state, _spatialHash, _field, config) {
    const cx = config.canvasW * config.attractorCenterX;
    const cy = config.canvasH * config.attractorCenterY;

    const dx = cx - state.x[i];
    const dy = cy - state.y[i];
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < 1) return;

    // Normalize and scale by config strength
    const s = config.attractorStrength;
    state.fx[i] += (dx / dist) * s;
    state.fy[i] += (dy / dist) * s;
  }
}
