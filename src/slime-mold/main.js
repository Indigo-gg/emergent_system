import p5 from 'p5';
import { createRNG } from '../prng.js';
import { ParticleState } from '../state.js';
import { Field } from '../field.js';
import { SlimeMoldPlugin } from '../plugins/slime-mold.js';
import { SlimeMoldEngine } from './engine.js';
import { createSlimeMoldRenderer } from './renderer.js';

// --- Configuration ---
const CONFIG = {
  // Canvas
  canvasW: 800,
  canvasH: 800,

  // Particles
  particleCount: 3000,

  // Physics (used by engine for timing)
  dt: 1 / 60,

  // Chemical field resolution (lower than canvas for performance)
  fieldCols: 200,
  fieldRows: 200,

  // Slime mold parameters — TWEAK THESE to get different patterns!
  slimeMold: {
    sensorAngle: 0.4,       // radians (~23°) — angle of side sensors
    sensorDist: 0.03,       // normalized distance to sensors (0-1)
    turnAngle: 0.3,         // radians per turn decision
    speed: 1.5,             // pixels per tick
    depositAmount: 0.6,     // chemical deposited per agent per tick
    diffuseRate: 0.2,       // blending with neighbors (0=none, 1=full)
    decayRate: 0.98,        // exponential decay per tick (<1 = fade)
    sensorSize: 1,          // sensor area in grid cells (1=point, 3=3x3 avg)
  },

  // Rendering
  debugMode: false,

  // PRNG
  seed: 42,
};

// --- Bootstrap ---
const rng = createRNG(CONFIG.seed);

// Create particle state — agents start at random positions
const state = new ParticleState(CONFIG.particleCount);
for (let i = 0; i < CONFIG.particleCount; i++) {
  const x = rng() * CONFIG.canvasW;
  const y = rng() * CONFIG.canvasH;
  state.spawn(x, y, 0, 0);
}

// Chemical field
const field = new Field(CONFIG.fieldCols, CONFIG.fieldRows);

// Slime mold plugin
const plugin = new SlimeMoldPlugin();
plugin.initHeadings(state, rng);

// Engine
const engine = new SlimeMoldEngine(CONFIG, state, field, plugin);

// Renderer
const sketch = createSlimeMoldRenderer(engine, CONFIG);
new p5(sketch);

console.log(`[Slime Mold] ${CONFIG.particleCount} agents, field ${CONFIG.fieldCols}x${CONFIG.fieldRows}`);
console.log('Parameters to tweak:');
console.log('  sensorAngle  — wider = more exploration, narrower = sharper veins');
console.log('  sensorDist   — how far ahead agents look');
console.log('  decayRate    — lower = faster trail fading, higher = persistent trails');
console.log('  depositAmount — heavier deposit = thicker veins');
