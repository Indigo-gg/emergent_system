/**
 * Core emergence engine.
 *
 * Fixed-timestep tick loop with the intent-resolution pattern:
 *   1. Rebuild spatial index
 *   2. Plugins propose forces (intent)
 *   3. Engine integrates forces and enforces constraints (resolution)
 *
 * The engine never knows what "attraction" or "repulsion" means.
 * It only knows: forces come in, positions go out.
 */
export class Engine {
  constructor(config, state, spatialHash, pluginRegistry) {
    this.config = config;
    this.state = state;
    this.spatialHash = spatialHash;
    this.plugins = pluginRegistry;

    // Time accumulator for fixed timestep
    this._accumulator = 0;
    this._tickCount = 0;
  }

  /**
   * Advance the simulation by `frameTime` seconds.
   * May run 0 or multiple fixed-step ticks depending on accumulated time.
   */
  update(frameTime) {
    const dt = this.config.dt;
    this._accumulator += frameTime;

    while (this._accumulator >= dt) {
      this._tick();
      this._accumulator -= dt;
      this._tickCount++;
    }
  }

  /** Single fixed-step tick. */
  _tick() {
    const state = this.state;
    const n = state.count;
    const cfg = this.config;

    // 1. Rebuild spatial index
    this.spatialHash.build(state);

    // 2. Reset forces
    state.resetForces();

    // 3. Plugins propose forces
    this.plugins.resetAll();
    for (let i = 0; i < n; i++) {
      this.plugins.proposeAll(i, state, this.spatialHash, null, cfg);
    }

    // 4. Integrate forces → velocity → position
    const maxSpeed = cfg.maxSpeed;
    const maxForce = cfg.maxForce;
    const friction = cfg.friction;
    const w = cfg.canvasW;
    const h = cfg.canvasH;

    for (let i = 0; i < n; i++) {
      // Clamp force magnitude
      let fx = state.fx[i];
      let fy = state.fy[i];
      const fMag = Math.sqrt(fx * fx + fy * fy);
      if (fMag > maxForce) {
        const s = maxForce / fMag;
        fx *= s;
        fy *= s;
      }

      // Integrate: velocity += force
      state.vx[i] += fx;
      state.vy[i] += fy;

      // Apply friction
      state.vx[i] *= friction;
      state.vy[i] *= friction;

      // Clamp speed
      const speed = Math.sqrt(state.vx[i] * state.vx[i] + state.vy[i] * state.vy[i]);
      if (speed > maxSpeed) {
        const s = maxSpeed / speed;
        state.vx[i] *= s;
        state.vy[i] *= s;
      }

      // Integrate: position += velocity
      state.x[i] += state.vx[i];
      state.y[i] += state.vy[i];

      // Integrate: phase += dPhase (wrap to [0, 2π))
      state.phase[i] = (state.phase[i] + state.dPhase[i]) % (Math.PI * 2);
      if (state.phase[i] < 0) state.phase[i] += Math.PI * 2;

      // Boundary: wrap around (toroidal topology)
      if (state.x[i] < 0) state.x[i] += w;
      else if (state.x[i] >= w) state.x[i] -= w;
      if (state.y[i] < 0) state.y[i] += h;
      else if (state.y[i] >= h) state.y[i] -= h;
    }
  }

  /** Number of ticks completed. */
  get tickCount() {
    return this._tickCount;
  }
}
