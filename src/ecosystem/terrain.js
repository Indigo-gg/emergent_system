/**
 * Simple 2D terrain generator using value noise.
 *
 * Generates a grid of terrain values [0, 1]:
 *   - < wallThreshold: passable (safe zone)
 *   - wallThreshold ~ harshThreshold: harsh zone (2x metabolism)
 *   - > wallThreshold: wall (impassable)
 */

// Simple hash-based noise (no external dependency)
function hash(x, y, seed) {
  let h = seed + x * 374761393 + y * 668265263;
  h = (h ^ (h >> 13)) * 1274126177;
  h = h ^ (h >> 16);
  return (h & 0x7fffffff) / 0x7fffffff;
}

function smoothNoise(x, y, seed) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;

  // Smoothstep
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);

  const v00 = hash(ix, iy, seed);
  const v10 = hash(ix + 1, iy, seed);
  const v01 = hash(ix, iy + 1, seed);
  const v11 = hash(ix + 1, iy + 1, seed);

  const top = v00 + sx * (v10 - v00);
  const bottom = v01 + sx * (v11 - v01);
  return top + sy * (bottom - top);
}

/**
 * Generate multi-octave noise terrain.
 * @param {number} cols - grid width
 * @param {number} rows - grid height
 * @param {number} scale - noise scale (0.01 = large features, 0.1 = small)
 * @param {number} seed - random seed
 * @returns {Float32Array} terrain values [0, 1]
 */
export function generateTerrain(cols, rows, scale, seed) {
  const data = new Float32Array(cols * rows);

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      let value = 0;
      let amplitude = 1;
      let frequency = scale;
      let maxValue = 0;

      // 4 octaves of noise
      for (let oct = 0; oct < 4; oct++) {
        value += smoothNoise(x * frequency, y * frequency, seed + oct * 1000) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2;
      }

      data[y * cols + x] = value / maxValue;
    }
  }

  return data;
}

/**
 * Classify terrain value.
 * @param {number} value - terrain noise value [0, 1]
 * @param {number} wallThreshold - above this = wall
 * @param {number} harshThreshold - above this = harsh zone
 * @returns {number} 0=safe, 1=harsh, 2=wall
 */
export function classifyTerrain(value, wallThreshold, harshThreshold) {
  if (value > wallThreshold) return 2;  // wall
  if (value > harshThreshold) return 1; // harsh
  return 0; // safe
}
