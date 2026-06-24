import { SPECIES } from './species.js';
import { generateTerrain, classifyTerrain } from './terrain.js';

/**
 * Ecosystem Engine v3
 *
 * Features:
 *   - Chemical trails (per species)
 *   - Nutrient field (with multiple drifting hotspots)
 *   - Waste/toxin field
 *   - Terrain (walls + harsh zones from noise)
 *
 * Tick order:
 *   1. Update hotspot positions (noise-based drift)
 *   2. Inject nutrients at hotspot locations
 *   3. Diffuse + decay trails
 *   4. Diffuse + decay nutrients
 *   5. Diffuse + decay waste
 *   6. Agents: sense → turn → move → deposit → eat → waste → terrain check → spore/reproduce
 */
export class EcosystemEngine {
  constructor(config, state, trails, nutrientField, wasteField, plugin) {
    this.config = config;
    this.state = state;
    this.trails = trails;
    this.nutrientField = nutrientField;
    this.wasteField = wasteField;
    this.plugin = plugin;

    this._accumulator = 0;
    this._tickCount = 0;

    // Generate terrain
    const eco = config.ecosystem;
    if (eco.terrainEnabled) {
      this.terrain = generateTerrain(
        config.fieldCols, config.fieldRows,
        eco.terrainScale || 0.04,
        eco.terrainSeed || 123
      );
      this.terrainWallThreshold = eco.terrainWallThreshold || 0.7;
      this.terrainHarshThreshold = eco.terrainHarshThreshold || 0.55;
    } else {
      this.terrain = null;
    }

    // Multiple drifting hotspots with variable speed
    const baseSpeed = eco.nutrientDriftSpeed || 0.0008;
    this.hotspots = [
      { x: 0.3, y: 0.3, angle: 0, baseSpeed: baseSpeed, radius: 0.25, phase: 0 },
      { x: 0.7, y: 0.7, angle: Math.PI, baseSpeed: baseSpeed * 0.6, radius: 0.2, phase: 2 },
      { x: 0.5, y: 0.2, angle: Math.PI / 2, baseSpeed: baseSpeed * 1.2, radius: 0.18, phase: 4 },
    ];
  }

  update(frameTime) {
    const dt = this.config.dt;
    this._accumulator += frameTime;
    if (this._accumulator > dt * 10) this._accumulator = dt * 10;

    while (this._accumulator >= dt) {
      this._tick();
      this._accumulator -= dt;
      this._tickCount++;
    }
  }

  /** Headless compute: run N ticks without rendering. */
  computeTicks(ticks) {
    for (let t = 0; t < ticks; t++) {
      this._tick();
      this._tickCount++;
    }
  }

  _tick() {
    const cfg = this.config;
    const eco = cfg.ecosystem;
    const nf = this.nutrientField;
    const wf = this.wasteField;

    // 1. Update hotspot positions
    this._updateHotspots();

    // 2. Inject nutrients at all hotspots
    this._injectNutrients();

    // 3. Field diffusion — every 5 ticks (fields change slowly, huge perf win)
    if (this._tickCount % 5 === 0) {
      for (const trail of this.trails) {
        trail.diffuseAndDecay(eco.trailDiffuseRate, eco.trailDecayRate);
      }
      nf.diffuseAndDecay(eco.nutrientDiffuseRate, eco.nutrientDecayRate);
      wf.diffuseAndDecay(eco.wasteDiffuseRate || 0.04, eco.wasteDecayRate || 0.998);
    }

    // 4. Agents update every tick (with terrain info)
    this.plugin.update(this.state, this.trails, nf, wf, cfg, this);
  }

  /**
   * Update hotspot positions using noise-like drift.
   * Each hotspot follows its own orbital pattern.
   */
  _updateHotspots() {
    const t = this._tickCount;
    for (const hs of this.hotspots) {
      // Variable speed: oscillates between 0.3x and 1.7x base speed
      // Creates "seasons" — fast drift then pause, organisms can catch up
      const speedMod = 1.0 + 0.7 * Math.sin(t * 0.0005 + hs.phase);
      const effectiveSpeed = hs.baseSpeed * Math.max(0.3, speedMod);

      hs.angle += effectiveSpeed;

      // Lissajous pattern with slow secondary oscillation
      hs.x = 0.5 + Math.cos(hs.angle) * hs.radius + Math.sin(hs.angle * 1.7) * hs.radius * 0.25;
      hs.y = 0.5 + Math.sin(hs.angle * 0.8) * hs.radius + Math.cos(hs.angle * 1.3) * hs.radius * 0.25;

      // Clamp to [0.1, 0.9]
      hs.x = Math.max(0.1, Math.min(0.9, hs.x));
      hs.y = Math.max(0.1, Math.min(0.9, hs.y));
    }
  }

  /**
   * Inject nutrients at all hotspot locations.
   */
  _injectNutrients() {
    const cfg = this.config;
    const eco = cfg.ecosystem;
    const nf = this.nutrientField;

    if (this._tickCount % eco.nutrientInjectInterval !== 0) return;

    const patchRadius = eco.nutrientPatchRadius;
    const amount = eco.nutrientPatchAmount;
    const rng = this.plugin.rng;

    // Inject at each hotspot
    const patchesPerHotspot = Math.max(1, Math.floor(eco.nutrientPatchCount / this.hotspots.length));

    for (const hs of this.hotspots) {
      const hcx = Math.floor(hs.x * nf.cols);
      const hcy = Math.floor(hs.y * nf.rows);

      for (let p = 0; p < patchesPerHotspot; p++) {
        const angle = rng() * Math.PI * 2;
        const dist = rng() * patchRadius * 1.5;
        const cx = Math.floor(hcx + Math.cos(angle) * dist);
        const cy = Math.floor(hcy + Math.sin(angle) * dist);

        for (let dy = -patchRadius; dy <= patchRadius; dy++) {
          for (let dx = -patchRadius; dx <= patchRadius; dx++) {
            const d = Math.sqrt(dx * dx + dy * dy);
            if (d > patchRadius) continue;

            const nc = cx + dx;
            const nr = cy + dy;
            if (nc < 0 || nc >= nf.cols || nr < 0 || nr >= nf.rows) continue;

            // Check terrain: don't inject on walls
            if (this.terrain) {
              const tIdx = nr * nf.cols + nc;
              if (this.terrain[tIdx] > this.terrainWallThreshold) continue;
            }

            const idx = nr * nf.cols + nc;
            const falloff = 1 - (d / patchRadius);
            nf.current[idx] = Math.min(nf.current[idx] + amount * falloff, 5.0);
          }
        }
      }
    }
  }

  /**
   * Check if a position is a wall.
   * @returns {boolean} true if wall (impassable)
   */
  isWall(worldX, worldY) {
    if (!this.terrain) return false;
    const { canvasW, canvasH, fieldCols, fieldRows } = this.config;
    const gx = Math.floor((worldX / canvasW) * fieldCols);
    const gy = Math.floor((worldY / canvasH) * fieldRows);
    if (gx < 0 || gx >= fieldCols || gy < 0 || gy >= fieldRows) return true;
    const idx = gy * fieldCols + gx;
    return this.terrain[idx] > this.terrainWallThreshold;
  }

  /**
   * Check if a position is a harsh zone.
   * @returns {boolean} true if harsh (2x metabolism)
   */
  isHarshZone(worldX, worldY) {
    if (!this.terrain) return false;
    const { canvasW, canvasH, fieldCols, fieldRows } = this.config;
    const gx = Math.floor((worldX / canvasW) * fieldCols);
    const gy = Math.floor((worldY / canvasH) * fieldRows);
    if (gx < 0 || gx >= fieldCols || gy < 0 || gy >= fieldRows) return false;
    const idx = gy * fieldCols + gx;
    const val = this.terrain[idx];
    return val > this.terrainHarshThreshold && val <= this.terrainWallThreshold;
  }

  getPopulations() {
    const n = this.state.count;
    const counts = new Array(SPECIES.length).fill(0);
    for (let i = 0; i < n; i++) {
      counts[this.plugin.speciesId[i]]++;
    }
    return counts;
  }

  getHotspotPositions() {
    return this.hotspots.map(hs => ({ x: hs.x, y: hs.y }));
  }

  get tickCount() {
    return this._tickCount;
  }
}
