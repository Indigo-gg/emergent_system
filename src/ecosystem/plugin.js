import { Plugin } from '../plugin.js';
import { SPECIES } from './species.js';

/**
 * Multi-Species Ecosystem Plugin v3
 *
 * Features:
 *   1. Spore mechanism: dormant state instead of death
 *   2. Cross-species pheromone interaction
 *   3. Waste/toxin field: agents produce waste when eating, repelled by high waste
 *
 * Per-tick lifecycle:
 *   1. Sense: own trail + other species trails + nutrient + waste
 *   2. Turn: toward nutrients/trails, away from waste
 *   3. Move: advance position, burn energy
 *   4. Deposit trail
 *   5. Eat nutrients → produce waste
 *   6. Metabolism (modified by waste concentration)
 *   7. Spore/revive/reproduce
 */

// Cross-species interaction matrix
const INTERACTION_MATRIX = [
  [1.0,  0.3, -0.2],   // Explorer
  [-0.1, 1.0,  0.4],   // Grazer
  [0.2, -0.1,  1.0],   // Harvester
];

// Spore configuration
const SPORE_METABOLISM = 0.001;
const SPORE_REVIVE_NUTRIENT = 0.3;
const SPORE_MAX_TICKS = 600;

export class EcosystemPlugin extends Plugin {
  constructor(rng) {
    super('ecosystem', 1.0);
    this.rng = rng;

    this.headings = null;
    this.energy = null;
    this.speciesId = null;
    this.isSpore = null;
    this.sporeTicks = null;

    this._toKill = [];
    this._toSpawn = [];
    this._toRevive = [];
  }

  init(state, rng) {
    const max = state.maxCount;
    this.headings = new Float32Array(max);
    this.energy = new Float32Array(max);
    this.speciesId = new Uint8Array(max);
    this.isSpore = new Uint8Array(max);
    this.sporeTicks = new Uint16Array(max);

    for (let i = 0; i < state.count; i++) {
      this.speciesId[i] = i % SPECIES.length;
      this.headings[i] = rng() * Math.PI * 2;
      this.energy[i] = SPECIES[this.speciesId[i]].maxEnergy * 0.6;
      this.isSpore[i] = 0;
      this.sporeTicks[i] = 0;
    }
  }

  update(state, trails, nutrientField, wasteField, config, engine) {
    const n = state.count;
    const w = config.canvasW;
    const h = config.canvasH;
    const eco = config.ecosystem;
    const rng = this.rng;
    const terrain = engine ? engine.terrain : null;

    this._toKill.length = 0;
    this._toSpawn.length = 0;
    this._toRevive.length = 0;

    const wasteProductionRate = eco.wasteProductionRate || 0.15;
    const wasteMetabolismFactor = eco.wasteMetabolismFactor || 2.0;
    const wasteRepelStrength = eco.wasteRepelStrength || 0.8;

    for (let i = 0; i < n; i++) {
      const sid = this.speciesId[i];
      const sp = SPECIES[sid];
      const heading = this.headings[i];

      const nx = state.x[i] / w;
      const ny = state.y[i] / h;

      // ===== SPORE STATE =====
      if (this.isSpore[i]) {
        this.sporeTicks[i]++;
        this.energy[i] -= SPORE_METABOLISM;

        const ngc = Math.floor(nx * nutrientField.cols);
        const ngr = Math.floor(ny * nutrientField.rows);
        const nIdx = ngr * nutrientField.cols + ngc;
        const nearbyNutrient = (ngc >= 0 && ngc < nutrientField.cols && ngr >= 0 && ngr < nutrientField.rows)
          ? nutrientField.current[nIdx] : 0;

        if (nearbyNutrient >= SPORE_REVIVE_NUTRIENT) {
          this._toRevive.push(i);
          continue;
        }

        if (this.energy[i] <= 0 || this.sporeTicks[i] >= SPORE_MAX_TICKS) {
          this._toKill.push(i);
        }
        continue;
      }

      // ===== ACTIVE STATE =====
      const trail = trails[sid];
      const cosH = Math.cos(heading);
      const sinH = Math.sin(heading);
      const sd = sp.sensorDist;
      const sa = sp.sensorAngle;

      // Helper: sample at sensor position
      const sense = (field, dx, dy) => field.sample(nx + dx, ny + dy);
      const sC = [sinH * sd, -cosH * sd];
      const sL = [Math.sin(heading - sa) * sd, -Math.cos(heading - sa) * sd];
      const sR = [Math.sin(heading + sa) * sd, -Math.cos(heading + sa) * sd];

      // --- Sense own trail ---
      const trailC = sense(trail, sC[0], sC[1]);
      const trailL = sense(trail, sL[0], sL[1]);
      const trailR = sense(trail, sR[0], sR[1]);

      // --- Sense other species trails ---
      let otherC = 0, otherL = 0, otherR = 0;
      for (let s = 0; s < SPECIES.length; s++) {
        if (s === sid) continue;
        const weight = INTERACTION_MATRIX[sid][s];
        if (Math.abs(weight) < 0.01) continue;
        const ot = trails[s];
        otherC += sense(ot, sC[0], sC[1]) * weight;
        otherL += sense(ot, sL[0], sL[1]) * weight;
        otherR += sense(ot, sR[0], sR[1]) * weight;
      }

      // --- Sense nutrient field ---
      const nutC = sense(nutrientField, sC[0], sC[1]);
      const nutL = sense(nutrientField, sL[0], sL[1]);
      const nutR = sense(nutrientField, sR[0], sR[1]);

      // --- Sense waste field (repulsion) ---
      const wasteC = sense(wasteField, sC[0], sC[1]);
      const wasteL = sense(wasteField, sL[0], sL[1]);
      const wasteR = sense(wasteField, sR[0], sR[1]);

      // Combine signals
      const hunger = 1 - this.energy[i] / sp.maxEnergy;
      const nutWeight = 1.0 + hunger * 2.0;

      const sigC = trailC + otherC + nutC * nutWeight - wasteC * wasteRepelStrength;
      const sigL = trailL + otherL + nutL * nutWeight - wasteL * wasteRepelStrength;
      const sigR = trailR + otherR + nutR * nutWeight - wasteR * wasteRepelStrength;

      // --- Turn ---
      if (sigC > sigL && sigC > sigR) {
        // go straight
      } else if (sigC < sigL && sigC < sigR) {
        this.headings[i] += (rng() < 0.5 ? sp.turnAngle : -sp.turnAngle);
      } else if (sigL > sigR) {
        this.headings[i] -= sp.turnAngle;
      } else if (sigR > sigL) {
        this.headings[i] += sp.turnAngle;
      }

      // --- Move ---
      const newHeading = this.headings[i];
      state.vx[i] = Math.sin(newHeading) * sp.speed;
      state.vy[i] = -Math.cos(newHeading) * sp.speed;
      state.x[i] += state.vx[i];
      state.y[i] += state.vy[i];

      // Boundary: wrap
      if (state.x[i] < 0) state.x[i] += w;
      else if (state.x[i] >= w) state.x[i] -= w;
      if (state.y[i] < 0) state.y[i] += h;
      else if (state.y[i] >= h) state.y[i] -= h;

      // Terrain collision: walls block movement
      if (engine && engine.isWall(state.x[i], state.y[i])) {
        // Bounce back
        state.x[i] -= state.vx[i] * 1.5;
        state.y[i] -= state.vy[i] * 1.5;
        // Re-wrap
        if (state.x[i] < 0) state.x[i] += w;
        else if (state.x[i] >= w) state.x[i] -= w;
        if (state.y[i] < 0) state.y[i] += h;
        else if (state.y[i] >= h) state.y[i] -= h;
        // Random turn
        this.headings[i] += (rng() - 0.5) * Math.PI;
      }

      // --- Deposit trail ---
      const gc = Math.floor(nx * trail.cols);
      const gr = Math.floor(ny * trail.rows);
      trail.deposit(gc, gr, sp.depositAmount);

      // --- Eat nutrients → produce waste ---
      const ngc2 = Math.floor(nx * nutrientField.cols);
      const ngr2 = Math.floor(ny * nutrientField.rows);
      const nIdx2 = ngr2 * nutrientField.cols + ngc2;
      if (ngc2 >= 0 && ngc2 < nutrientField.cols && ngr2 >= 0 && ngr2 < nutrientField.rows) {
        const available = nutrientField.current[nIdx2];
        const consumed = Math.min(available, 0.5);
        nutrientField.current[nIdx2] -= consumed;
        this.energy[i] += consumed * sp.nutrientGain;

        // Produce waste proportional to consumption
        if (consumed > 0) {
          const wasteAmount = consumed * wasteProductionRate;
          const wgc = Math.floor(nx * wasteField.cols);
          const wgr = Math.floor(ny * wasteField.rows);
          wasteField.deposit(wgc, wgr, wasteAmount);
        }
      }

      // --- Metabolism cost (increased by waste + harsh terrain) ---
      const localWaste = wasteField.sample(nx, ny);
      const wasteFactor = 1 + localWaste * wasteMetabolismFactor;
      const harshFactor = (engine && engine.isHarshZone(state.x[i], state.y[i])) ? 2.0 : 1.0;
      this.energy[i] -= sp.metabolismCost * wasteFactor * harshFactor;

      // Clamp energy
      if (this.energy[i] > sp.maxEnergy) this.energy[i] = sp.maxEnergy;

      // --- Spore check ---
      if (this.energy[i] <= 0) {
        this.energy[i] = 0;
        this.isSpore[i] = 1;
        this.sporeTicks[i] = 0;
        state.vx[i] = 0;
        state.vy[i] = 0;
        continue;
      }

      // --- Reproduction ---
      const speciesCount = this._countSpecies(sid, n);
      if (
        this.energy[i] >= sp.reproduceThreshold &&
        speciesCount < sp.carryingCapacity &&
        n < state.maxCount - 1
      ) {
        this._toSpawn.push({
          x: state.x[i] + (rng() - 0.5) * 10,
          y: state.y[i] + (rng() - 0.5) * 10,
          sid,
          energy: sp.reproduceCost,
        });
        this.energy[i] -= sp.reproduceCost;
      }
    }

    // --- Execute revivals ---
    for (const idx of this._toRevive) {
      this.isSpore[idx] = 0;
      this.sporeTicks[idx] = 0;
      this.energy[idx] = SPECIES[this.speciesId[idx]].maxEnergy * 0.3;
      this.headings[idx] = this.rng() * Math.PI * 2;
    }

    // --- Execute deaths ---
    this._toKill.sort((a, b) => b - a);
    for (const idx of this._toKill) {
      this._removeAgent(idx, state);
    }

    // --- Execute spawns ---
    for (const s of this._toSpawn) {
      this._spawnAgent(s, state, w, h);
    }
  }

  _countSpecies(sid, n) {
    let count = 0;
    for (let i = 0; i < n; i++) {
      if (this.speciesId[i] === sid) count++;
    }
    return count;
  }

  _removeAgent(i, state) {
    const last = state.count - 1;
    if (i < last) {
      state.x[i] = state.x[last];
      state.y[i] = state.y[last];
      state.vx[i] = state.vx[last];
      state.vy[i] = state.vy[last];
      state.type[i] = state.type[last];
      this.headings[i] = this.headings[last];
      this.energy[i] = this.energy[last];
      this.speciesId[i] = this.speciesId[last];
      this.isSpore[i] = this.isSpore[last];
      this.sporeTicks[i] = this.sporeTicks[last];
    }
    state.count--;
  }

  _spawnAgent(s, state, w, h) {
    const i = state.count;
    state.x[i] = ((s.x % w) + w) % w;
    state.y[i] = ((s.y % h) + h) % h;
    state.vx[i] = 0;
    state.vy[i] = 0;
    state.type[i] = s.sid;
    this.headings[i] = this.rng() * Math.PI * 2;
    this.energy[i] = s.energy;
    this.speciesId[i] = s.sid;
    this.isSpore[i] = 0;
    this.sporeTicks[i] = 0;
    state.count++;
  }

  getSporeCounts() {
    const counts = new Array(SPECIES.length).fill(0);
    for (let i = 0; i < this.isSpore.length; i++) {
      if (this.isSpore[i]) counts[this.speciesId[i]]++;
    }
    return counts;
  }

  propose() {}
}
