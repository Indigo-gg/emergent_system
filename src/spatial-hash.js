/**
 * Spatial Hash Grid for fast neighbor queries.
 *
 * Uses the CellStart + ParticleNext linked-list scheme:
 *   - cellStart[cellIndex] = head of the particle chain in that cell
 *   - particleNext[particleIndex] = next particle in the same cell
 *
 * queryState(x, y, radius, state, cb) checks the surrounding cells.
 */
export class SpatialHash {
  constructor(cellSize, worldW, worldH) {
    this.cellSize = cellSize;
    this.cols = Math.ceil(worldW / cellSize);
    this.rows = Math.ceil(worldH / cellSize);
    this.numCells = this.cols * this.rows;

    this.cellStart = new Int32Array(this.numCells);
    this.particleNext = null;
  }

  /**
   * Rebuild the spatial index from current particle positions.
   * Must be called once per tick, before any queries.
   */
  build(state) {
    const n = state.count;

    if (!this.particleNext || this.particleNext.length < n) {
      this.particleNext = new Int32Array(Math.max(n, 1024));
    }

    this.cellStart.fill(-1);

    const cs = this.cellSize;
    const cols = this.cols;

    for (let i = 0; i < n; i++) {
      const col = (state.x[i] / cs) | 0;
      const row = (state.y[i] / cs) | 0;
      const ci = row * cols + col;

      if (ci < 0 || ci >= this.numCells) {
        this.particleNext[i] = -1;
        continue;
      }

      this.particleNext[i] = this.cellStart[ci];
      this.cellStart[ci] = i;
    }
  }

  /**
   * Find all particles within `radius` of (qx, qy).
   * Returns an array of neighbor indices.
   * More convenient than queryState when you need a list, not a callback.
   */
  query(qx, qy, radius, state) {
    const results = [];
    this.queryState(qx, qy, radius, state, (j, _d2) => {
      results.push(j);
    });
    return results;
  }

  /**
   * Find all particles within `radius` of (qx, qy).
   * Calls callback(neighborIndex, distSquared) for each neighbor.
   * Distances are SQUARED — caller compares with radius*radius.
   */
  queryState(qx, qy, radius, state, callback) {
    const cs = this.cellSize;
    const cols = this.cols;
    const rows = this.rows;
    const r2 = radius * radius;

    const minCol = Math.max(0, ((qx - radius) / cs) | 0);
    const maxCol = Math.min(cols - 1, ((qx + radius) / cs) | 0);
    const minRow = Math.max(0, ((qy - radius) / cs) | 0);
    const maxRow = Math.min(rows - 1, ((qy + radius) / cs) | 0);

    for (let row = minRow; row <= maxRow; row++) {
      for (let col = minCol; col <= maxCol; col++) {
        let idx = this.cellStart[row * cols + col];
        while (idx !== -1) {
          const dx = qx - state.x[idx];
          const dy = qy - state.y[idx];
          const d2 = dx * dx + dy * dy;
          if (d2 < r2 && d2 > 0) {
            callback(idx, d2);
          }
          idx = this.particleNext[idx];
        }
      }
    }
  }
}
