/**
 * Slime Mold Engine
 *
 * Update loop:
 *   1. Diffuse + decay the chemical field
 *   2. Agents sense the field, turn, move, deposit
 *
 * This is NOT force-based — agents are kinematic.
 * No spatial hash needed (agents interact via field, not neighbors).
 */
export class SlimeMoldEngine {
  constructor(config, state, field, plugin) {
    this.config = config;
    this.state = state;
    this.field = field;
    this.plugin = plugin;

    this._accumulator = 0;
    this._tickCount = 0;
  }

  /**
   * Advance simulation by `frameTime` seconds.
   * Uses fixed timestep internally.
   */
  update(frameTime) {
    const dt = this.config.dt;
    this._accumulator += frameTime;

    // Cap accumulator to prevent spiral of death
    if (this._accumulator > dt * 10) {
      this._accumulator = dt * 10;
    }

    while (this._accumulator >= dt) {
      this._tick();
      this._accumulator -= dt;
      this._tickCount++;
    }
  }

  _tick() {
    const cfg = this.config;
    const sm = cfg.slimeMold;

    // 1. Diffuse + decay chemical field
    this.field.diffuseAndDecay(sm.diffuseRate, sm.decayRate);

    // 2. Agents: sense → turn → move → deposit
    this.plugin.update(this.state, this.field, cfg);
  }

  get tickCount() {
    return this._tickCount;
  }
}
