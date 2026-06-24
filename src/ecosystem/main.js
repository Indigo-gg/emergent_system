import p5 from 'p5';
import { createRNG } from '../prng.js';
import { ParticleState } from '../state.js';
import { Field } from '../field.js';
import { SPECIES } from './species.js';
import { EcosystemPlugin } from './plugin.js';
import { EcosystemEngine } from './engine.js';
import { createEcosystemRenderer } from './renderer.js';

// --- Configuration ---
const CONFIG = {
  canvasW: 900,
  canvasH: 900,
  dt: 1 / 60,

  // Field resolution
  fieldCols: 250,
  fieldRows: 250,

  // Initial agents per species
  initialPerSpecies: 80,

  // Ecosystem parameters
  ecosystem: {
    // Trail behavior
    trailDiffuseRate: 0.2,
    trailDecayRate: 0.97,

    // Nutrient behavior
    nutrientDiffuseRate: 0.08,
    nutrientDecayRate: 0.999,

    // Nutrient injection
    nutrientInjectInterval: 60,
    nutrientPatchCount: 5,
    nutrientPatchRadius: 10,
    nutrientPatchAmount: 1.5,

    // Moving hotspot (seasonal drift)
    nutrientDriftSpeed: 0.0008,     // much slower drift

    // Waste/toxin system
    wasteDiffuseRate: 0.04,
    wasteDecayRate: 0.998,          // much slower decay — waste persists
    wasteProductionRate: 0.5,       // much more waste per nutrient consumed
    wasteMetabolismFactor: 2.5,     // metabolism multiplier at max waste
    wasteRepelStrength: 1.2,        // stronger repulsion

    // Terrain
    terrainEnabled: true,
    terrainSeed: 123,
    terrainScale: 0.04,             // noise scale (larger = bigger features)
    terrainWallThreshold: 0.7,      // noise > this = wall (impassable)
    terrainHarshThreshold: 0.55,    // noise > this = harsh zone (2x metabolism)
  },

  // Rendering
  debugMode: false,

  // PRNG
  seed: 42,
};

// --- Bootstrap ---
const rng = createRNG(CONFIG.seed);

// Particle state
const totalAgents = CONFIG.initialPerSpecies * SPECIES.length;
const state = new ParticleState(totalAgents + 500); // extra room for reproduction

// Spawn initial agents, distributed across species
for (let s = 0; s < SPECIES.length; s++) {
  for (let i = 0; i < CONFIG.initialPerSpecies; i++) {
    const x = rng() * CONFIG.canvasW;
    const y = rng() * CONFIG.canvasH;
    state.spawn(x, y, 0, 0, s);
  }
}

// Chemical trails — one per species
const trails = SPECIES.map(() => new Field(CONFIG.fieldCols, CONFIG.fieldRows));

// Nutrient field
const nutrientField = new Field(CONFIG.fieldCols, CONFIG.fieldRows);

// Waste/toxin field
const wasteField = new Field(CONFIG.fieldCols, CONFIG.fieldRows);

// Plugin
const plugin = new EcosystemPlugin(rng);
plugin.init(state, rng);

// Engine
const engine = new EcosystemEngine(CONFIG, state, trails, nutrientField, wasteField, plugin);

// Renderer
const sketch = createEcosystemRenderer(engine, CONFIG);
new p5(sketch);

console.log(`[Ecosystem] ${SPECIES.length} species, ${state.count} initial agents`);
console.log('Species:', SPECIES.map(s => s.name).join(', '));
console.log('Watch the population bars — no species should go extinct!');
