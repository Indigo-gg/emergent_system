/**
 * SoA (Structure of Arrays) particle state buffer.
 * All data lives in contiguous TypedArrays for cache-friendly access.
 *
 * Double-buffered: engine reads from `current`, writes to `next`,
 * then swaps references at the end of each tick.
 */
export class ParticleState {
  constructor(maxCount) {
    this.maxCount = maxCount;
    this.count = 0;

    // Core kinematic data
    this.x  = new Float32Array(maxCount);
    this.y  = new Float32Array(maxCount);
    this.vx = new Float32Array(maxCount);
    this.vy = new Float32Array(maxCount);

    // Metadata
    this.type = new Uint8Array(maxCount); // particle type (for multi-type plugins)

    // Oscillator / energy state (for Kuramoto + energy dynamics)
    this.phase  = new Float32Array(maxCount);  // internal oscillation phase [0, 2π)
    this.energy = new Float32Array(maxCount);  // abstract energy budget
    this.dPhase = new Float32Array(maxCount);  // phase accumulator (like fx/fy for phase)

    // Temporary force accumulator (reset each tick)
    this.fx = new Float32Array(maxCount);
    this.fy = new Float32Array(maxCount);
  }

  /** Add a particle. Returns its index, or -1 if full. */
  spawn(x, y, vx = 0, vy = 0, type = 0, phase = 0, energy = 1.0) {
    if (this.count >= this.maxCount) return -1;
    const i = this.count++;
    this.x[i] = x;
    this.y[i] = y;
    this.vx[i] = vx;
    this.vy[i] = vy;
    this.type[i] = type;
    this.phase[i] = phase;
    this.energy[i] = energy;
    this.dPhase[i] = 0;
    this.fx[i] = 0;
    this.fy[i] = 0;
    return i;
  }

  /** Remove a particle by swapping with the last one (O(1)). */
  kill(i) {
    if (i < 0 || i >= this.count) return;
    this.count--;
    if (i < this.count) {
      const j = this.count;
      this.x[i]  = this.x[j];
      this.y[i]  = this.y[j];
      this.vx[i] = this.vx[j];
      this.vy[i] = this.vy[j];
      this.type[i] = this.type[j];
      this.phase[i] = this.phase[j];
      this.energy[i] = this.energy[j];
    }
  }

  /** Zero out all force and phase accumulators. Called at the start of each tick. */
  resetForces() {
    this.fx.fill(0, 0, this.count);
    this.fy.fill(0, 0, this.count);
    this.dPhase.fill(0, 0, this.count);
  }
}
