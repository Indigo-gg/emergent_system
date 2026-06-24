/**
 * 2D Chemical Field (Grid-based)
 *
 * Stores concentration values in a flat Float32Array.
 * Supports diffusion (3x3 box blur) and exponential decay.
 * Double-buffered to avoid read/write order dependency.
 */
export class Field {
  constructor(cols, rows) {
    this.cols = cols;
    this.rows = rows;
    this.size = cols * rows;

    // Double buffer
    this.current = new Float32Array(this.size);
    this.next = new Float32Array(this.size);
  }

  /** Read value at grid coords (clamped to bounds). */
  get(col, row) {
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return 0;
    return this.current[row * this.cols + col];
  }

  /** Read value using bilinear interpolation for continuous coords. */
  sample(x, y) {
    // Map world coords to grid coords
    const gx = x * this.cols;
    const gy = y * this.rows;

    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const x1 = x0 + 1;
    const y1 = y0 + 1;

    const fx = gx - x0;
    const fy = gy - y0;

    const v00 = this.get(x0, y0);
    const v10 = this.get(x1, y0);
    const v01 = this.get(x0, y1);
    const v11 = this.get(x1, y1);

    return v00 * (1 - fx) * (1 - fy)
         + v10 * fx * (1 - fy)
         + v01 * (1 - fx) * fy
         + v11 * fx * fy;
  }

  /** Add value at grid coords (clamped). */
  deposit(col, row, amount) {
    if (col < 0 || col >= this.cols || row < 0 || row >= this.rows) return;
    const idx = row * this.cols + col;
    this.current[idx] = Math.min(this.current[idx] + amount, 5.0); // cap to prevent infinity
  }

  /**
   * Diffuse + Decay.
   * 3x3 box blur for diffusion, then exponential decay.
   * Writes to `next`, then swaps.
   */
  diffuseAndDecay(diffuseRate, decayRate) {
    const { cols, rows, current, next } = this;

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        const idx = row * cols + col;

        // 3x3 box blur average of neighbors
        let sum = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nc = col + dx;
            const nr = row + dy;
            if (nc >= 0 && nc < cols && nr >= 0 && nr < rows) {
              sum += current[nr * cols + nc];
              count++;
            }
          }
        }
        const avg = sum / count;

        // Blend: diffuseRate controls how much blending with neighbors
        const blended = current[idx] * (1 - diffuseRate) + avg * diffuseRate;

        // Decay
        next[idx] = blended * decayRate;
      }
    }

    // Swap
    const tmp = this.current;
    this.current = this.next;
    this.next = tmp;
  }

  /** Clear all values. */
  clear() {
    this.current.fill(0);
    this.next.fill(0);
  }
}
