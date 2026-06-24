import { Plugin } from '../plugin.js';

/**
 * Slime Mold (Physarum) Plugin
 *
 * Based on "Characteristics of pattern formation and evolution
 * in approximations of Physarum transport networks" (Jones 2010).
 *
 * Each agent has a heading angle and three chemical sensors
 * (left, center, right). The agent:
 *   1. Senses chemical concentration at three points ahead
 *   2. Turns toward the highest reading
 *   3. Moves forward
 *   4. Deposits chemical at its position
 *
 * This is NOT a force-based plugin. It directly updates position
 * and heading because Physarum agents are kinematic, not dynamic.
 * The engine must call `moveAgents()` after the propose phase.
 */
export class SlimeMoldPlugin extends Plugin {
  constructor() {
    super('slime-mold', 1.0);

    // Per-agent heading angles (initialized lazily)
    this.headings = null;
  }

  /** Initialize heading angles for all agents. */
  initHeadings(state, rng) {
    this.headings = new Float32Array(state.maxCount);
    for (let i = 0; i < state.count; i++) {
      this.headings[i] = rng() * Math.PI * 2;
    }
  }

  /**
   * Sense the chemical field at a point ahead of the agent.
   * Returns the average concentration in a small area.
   */
  _sense(field, nx, ny, sensorSize) {
    // Convert normalized coords to grid coords
    const gcx = nx * field.cols;
    const gcy = ny * field.rows;

    // Average over a small area (sensorSize x sensorSize)
    let sum = 0;
    let count = 0;
    const half = Math.floor(sensorSize / 2);
    for (let dy = -half; dy <= half; dy++) {
      for (let dx = -half; dx <= half; dx++) {
        const c = Math.floor(gcx) + dx;
        const r = Math.floor(gcy) + dy;
        if (c >= 0 && c < field.cols && r >= 0 && r < field.rows) {
          sum += field.current[r * field.cols + c];
          count++;
        }
      }
    }
    return count > 0 ? sum / count : 0;
  }

  /**
   * Main update: sense → turn → move → deposit.
   * This bypasses the force system entirely — Physarum agents
   * are kinematic (they set velocity directly, not via forces).
   */
  update(state, field, config) {
    if (!this.headings || this.headings.length < state.maxCount) return;

    const n = state.count;
    const cfg = config.slimeMold;
    const {
      sensorAngle,
      sensorDist,
      turnAngle,
      speed,
      depositAmount,
      sensorSize,
    } = cfg;

    const w = config.canvasW;
    const h = config.canvasH;

    for (let i = 0; i < n; i++) {
      const heading = this.headings[i];
      const nx = state.x[i] / w; // normalized [0,1]
      const ny = state.y[i] / h;

      // --- Sense ---
      // Three sensors: left, center, right
      const cosH = Math.cos(heading);
      const sinH = Math.sin(heading);

      // Center sensor
      const cx = nx + sinH * sensorDist;
      const cy = ny - cosH * sensorDist; // y-axis is inverted in screen coords
      const cVal = this._sense(field, cx, cy, sensorSize);

      // Left sensor
      const la = heading - sensorAngle;
      const lx = nx + Math.sin(la) * sensorDist;
      const ly = ny - Math.cos(la) * sensorDist;
      const lVal = this._sense(field, lx, ly, sensorSize);

      // Right sensor
      const ra = heading + sensorAngle;
      const rx = nx + Math.sin(ra) * sensorDist;
      const ry = ny - Math.cos(ra) * sensorDist;
      const rVal = this._sense(field, rx, ry, sensorSize);

      // --- Turn ---
      if (cVal > lVal && cVal > rVal) {
        // Center is highest: go straight
      } else if (cVal < lVal && cVal < rVal) {
        // Both sides higher: random turn
        this.headings[i] += (Math.random() < 0.5 ? turnAngle : -turnAngle);
      } else if (lVal > rVal) {
        this.headings[i] -= turnAngle;
      } else if (rVal > lVal) {
        this.headings[i] += turnAngle;
      }
      // If all equal, no turn

      // --- Move ---
      const newHeading = this.headings[i];
      state.vx[i] = Math.sin(newHeading) * speed;
      state.vy[i] = -Math.cos(newHeading) * speed;

      // Update position directly
      state.x[i] += state.vx[i];
      state.y[i] += state.vy[i];

      // --- Boundary: wrap (toroidal) ---
      if (state.x[i] < 0) state.x[i] += w;
      else if (state.x[i] >= w) state.x[i] -= w;
      if (state.y[i] < 0) state.y[i] += h;
      else if (state.y[i] >= h) state.y[i] -= h;

      // --- Deposit ---
      const gc = Math.floor((state.x[i] / w) * field.cols);
      const gr = Math.floor((state.y[i] / h) * field.rows);
      field.deposit(gc, gr, depositAmount);
    }
  }

  /** Standard propose is unused — we override with update(). */
  propose() {}
}
