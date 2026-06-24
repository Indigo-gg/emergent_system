/**
 * Plugin base class.
 *
 * A plugin expresses "desires" — it proposes forces on particles.
 * The engine is responsible for integrating those forces and enforcing
 * constraints (speed limits, boundaries, etc.).
 *
 * Lifecycle per tick:
 *   1. reset()                     — clear any per-tick state
 *   2. propose(i, state, ...)      — for each particle, write force into state.fx/fy
 *
 * Plugins never directly modify position or velocity.
 */
export class Plugin {
  constructor(name, weight = 1.0) {
    this.name = name;
    this.weight = weight; // higher = more influence when blending
    this.enabled = true;
  }

  /** Called once at the start of each tick. Override if needed. */
  reset() {}

  /**
   * Propose a force for particle `i`.
   * Write directly to state.fx[i] and state.fy[i] (additive).
   *
   * @param {number} i - particle index
   * @param {ParticleState} state - current particle state (read-only positions/velocities)
   * @param {SpatialHash} spatialHash - for neighbor queries
   * @param {object|null} field - reserved for future field system
   * @param {object} config - runtime config
   */
  propose(i, state, spatialHash, field, config) {
    // Subclasses override this
  }
}

/**
 * Plugin registry — manages an ordered list of plugins.
 */
export class PluginRegistry {
  constructor() {
    this.plugins = [];
  }

  /** Register a plugin. Plugins are sorted by weight (descending) for priority. */
  register(plugin) {
    this.plugins.push(plugin);
    this.plugins.sort((a, b) => b.weight - a.weight);
  }

  /** Remove a plugin by name. */
  unregister(name) {
    this.plugins = this.plugins.filter(p => p.name !== name);
  }

  /** Get a plugin by name. */
  get(name) {
    return this.plugins.find(p => p.name === name);
  }

  /** Reset all plugins (called at start of tick). */
  resetAll() {
    for (const p of this.plugins) {
      if (p.enabled) p.reset();
    }
  }

  /**
   * Run all enabled plugins for particle `i`.
   * Each plugin adds its force to state.fx[i], state.fy[i].
   */
  proposeAll(i, state, spatialHash, field, config) {
    for (const p of this.plugins) {
      if (p.enabled) {
        p.propose(i, state, spatialHash, field, config);
      }
    }
  }
}
