/**
 * Headless Ecosystem Runner v2
 *
 * Key optimization: compute/render decoupling.
 * - Physics runs headlessly (no canvas) for 15,000 ticks
 * - Rendering only happens at 5 key frames for screenshots
 * - Pure compute: ~5 seconds for 15k ticks
 *
 * New features:
 *   - Waste/toxin field (B-Z oscillation dynamics)
 *   - Moving nutrient hotspots (seasonal migration)
 *   - Spore mechanism (dormancy instead of death)
 *   - Cross-species pheromone interaction
 */
import { createRNG } from '../prng.js';
import { ParticleState } from '../state.js';
import { Field } from '../field.js';
import { SPECIES } from '../ecosystem/species.js';
import { EcosystemPlugin } from '../ecosystem/plugin.js';
import { EcosystemEngine } from '../ecosystem/engine.js';

window.runEmergenceExperiment = async function (params) {
  const {
    canvasW = 900, canvasH = 900, seed = 42,
    simFrames = 15000,
    initialPerSpecies = 80,
    // Ecosystem params
    trailDiffuseRate = 0.2,
    trailDecayRate = 0.97,
    nutrientDiffuseRate = 0.08,
    nutrientDecayRate = 0.999,
    nutrientInjectInterval = 60,
    nutrientPatchCount = 5,
    nutrientPatchRadius = 10,
    nutrientPatchAmount = 1.5,
    nutrientDriftSpeed = 0.0008,
    // Waste params
    wasteDiffuseRate = 0.06,
    wasteDecayRate = 0.995,
    wasteProductionRate = 0.15,
    wasteMetabolismFactor = 2.0,
    wasteRepelStrength = 0.8,
    // Terrain
    terrainEnabled = true,
    terrainSeed = 123,
    terrainScale = 0.04,
    terrainWallThreshold = 0.7,
    terrainHarshThreshold = 0.55,
  } = params;

  const fieldCols = 250, fieldRows = 250;
  const rng = createRNG(seed);

  // Particle state
  const totalAgents = initialPerSpecies * SPECIES.length;
  const state = new ParticleState(totalAgents + 2000);
  for (let s = 0; s < SPECIES.length; s++) {
    for (let i = 0; i < initialPerSpecies; i++) {
      state.spawn(rng() * canvasW, rng() * canvasH, 0, 0, s);
    }
  }

  // Fields
  const trails = SPECIES.map(() => new Field(fieldCols, fieldRows));
  const nutrientField = new Field(fieldCols, fieldRows);
  const wasteField = new Field(fieldCols, fieldRows);

  // Plugin + Engine
  const plugin = new EcosystemPlugin(rng);
  plugin.init(state, rng);

  const config = {
    canvasW, canvasH, dt: 1 / 60,
    fieldCols, fieldRows,
    ecosystem: {
      trailDiffuseRate, trailDecayRate,
      nutrientDiffuseRate, nutrientDecayRate,
      nutrientInjectInterval, nutrientPatchCount,
      nutrientPatchRadius, nutrientPatchAmount,
      nutrientDriftSpeed,
      wasteDiffuseRate, wasteDecayRate,
      wasteProductionRate, wasteMetabolismFactor,
      wasteRepelStrength,
      terrainEnabled, terrainSeed, terrainScale,
      terrainWallThreshold, terrainHarshThreshold,
    },
  };
  const engine = new EcosystemEngine(config, state, trails, nutrientField, wasteField, plugin);

  // Canvas for rendering (only used at key frames)
  const canvas = document.createElement('canvas');
  canvas.width = canvasW;
  canvas.height = canvasH;
  const ctx = canvas.getContext('2d');

  // Key frames — [RISK-1] only post-burn-in slices (70/80/90/100%)
  const renderPoints = [
    Math.floor(simFrames * 0.70),   // 10500 tick — post-burn-in stabilization
    Math.floor(simFrames * 0.80),   // 12000 tick — mature structure
    Math.floor(simFrames * 0.90),   // 13500 tick — pre-final convergence
    simFrames,                       // 15000 tick — final state
  ];
  const sliceScreenshots = [];
  let nextRenderIdx = 0;

  // Stats collection (last 3000 frames, every 30 frames)
  const statsWindowStart = simFrames - 3000;
  const statsSampleInterval = 30;
  const timeSeries = {
    populations: SPECIES.map(() => []),
    totalPop: [],
    avgEnergy: SPECIES.map(() => []),
    wasteLevel: [],
  };

  // ===== HEADLESS COMPUTE PHASE =====
  // Run physics without rendering for maximum speed
  const computeStart = Date.now();

  for (let frame = 0; frame < simFrames; frame++) {
    // Pure tick (no frame time, no rendering)
    engine.computeTicks(1);

    // Render only at key frames
    if (nextRenderIdx < renderPoints.length && frame === renderPoints[nextRenderIdx]) {
      sliceScreenshots.push(renderSnapshot(ctx, engine, config, SPECIES));
      nextRenderIdx++;
    }

    // Collect stats
    if (frame >= statsWindowStart && frame % statsSampleInterval === 0) {
      const pops = engine.getPopulations();
      for (let s = 0; s < SPECIES.length; s++) {
        timeSeries.populations[s].push(pops[s]);
        timeSeries.avgEnergy[s].push(computeAvgEnergyForSpecies(plugin, s));
      }
      timeSeries.totalPop.push(state.count);
      timeSeries.wasteLevel.push(computeTotalWaste(wasteField));
    }
  }

  const computeMs = Date.now() - computeStart;
  console.log(`[Runner] Compute: ${simFrames} ticks in ${computeMs}ms`);

  // Trail rendering (80 frames with fade)
  ctx.fillStyle = 'rgb(5, 5, 10)';
  ctx.fillRect(0, 0, canvasW, canvasH);
  for (let f = 0; f < 80; f++) {
    engine.computeTicks(1);
    renderFrame(ctx, engine, config, SPECIES, 0.15);
  }
  const trailScreenshot = canvas.toDataURL('image/png');

  // Generate difference map (70% vs 100%) — compare early stable vs final
  const diffMap = await generateDiffMap(sliceScreenshots[0], sliceScreenshots[3], canvasW, canvasH);

  // Build composite
  const compositeScreenshot = await buildComposite(
    sliceScreenshots, trailScreenshot, diffMap, canvasW, canvasH, SPECIES
  );

  // Final stats
  const finalPops = engine.getPopulations();
  const sporeCounts = plugin.getSporeCounts();
  const stats = {
    tickCount: engine.tickCount,
    populations: finalPops,
    totalPopulation: state.count,
    sporeCounts,
    activeCounts: finalPops.map((p, i) => p - sporeCounts[i]),
    avgEnergy: SPECIES.map((_, s) => computeAvgEnergyForSpecies(plugin, s)),
    speciesRatios: finalPops.map(p => p / state.count),
    hotspots: engine.getHotspotPositions(),
    temporal_variance: {
      population_variance: timeSeries.totalPop.length > 1 ? variance(timeSeries.totalPop) : 0,
      population_trend: trend(timeSeries.totalPop),
      species_balance: computeBalance(finalPops),
      waste_trend: trend(timeSeries.wasteLevel),
    },
  };

  return {
    screenshots: [...sliceScreenshots, trailScreenshot],
    composite_screenshot: compositeScreenshot,
    stats,
  };
};

// ═══════════════════════════════════════
// Rendering helpers
// ═══════════════════════════════════════

function renderSnapshot(ctx, engine, config, SPECIES) {
  const { canvasW, canvasH } = config;
  ctx.fillStyle = 'rgb(5, 5, 10)';
  ctx.fillRect(0, 0, canvasW, canvasH);
  renderFrame(ctx, engine, config, SPECIES, 1.0);
  return ctx.canvas.toDataURL('image/png');
}

function renderFrame(ctx, engine, config, SPECIES, alpha) {
  const { canvasW, canvasH, fieldCols, fieldRows } = config;
  const scaleX = canvasW / fieldCols;
  const scaleY = canvasH / fieldRows;

  // Terrain (walls + harsh zones)
  if (engine.terrain) {
    const eco = config.ecosystem;
    for (let row = 0; row < fieldRows; row++) {
      for (let col = 0; col < fieldCols; col++) {
        const val = engine.terrain[row * fieldCols + col];
        if (val > (eco.terrainWallThreshold || 0.7)) {
          ctx.fillStyle = `rgba(45,40,35,${alpha})`;
          ctx.fillRect(col * scaleX, row * scaleY, scaleX + 1, scaleY + 1);
        } else if (val > (eco.terrainHarshThreshold || 0.55)) {
          ctx.fillStyle = `rgba(60,35,15,${0.5 * alpha})`;
          ctx.fillRect(col * scaleX, row * scaleY, scaleX + 1, scaleY + 1);
        }
      }
    }
  }

  // Nutrient field (amber)
  const nf = engine.nutrientField;
  for (let row = 0; row < fieldRows; row++) {
    for (let col = 0; col < fieldCols; col++) {
      const val = nf.current[row * fieldCols + col];
      const t = Math.min(val / 2.0, 1.0);
      if (t > 0.01) {
        const r = Math.floor(40 + t * 10);
        const g = Math.floor(t * 200);
        const b = Math.floor(t * 60);
        ctx.fillStyle = `rgba(${r},${g},${b},${t * 160 * alpha / 255})`;
        ctx.fillRect(col * scaleX, row * scaleY, scaleX + 1, scaleY + 1);
      }
    }
  }

  // Waste field (toxic red-purple)
  const wf = engine.wasteField;
  for (let row = 0; row < fieldRows; row++) {
    for (let col = 0; col < fieldCols; col++) {
      const val = wf.current[row * fieldCols + col];
      const t = Math.min(val / 1.5, 1.0);
      if (t > 0.01) {
        ctx.fillStyle = `rgba(${Math.floor(t * 200)},${Math.floor(t * 30)},${Math.floor(t * 80)},${t * 140 * alpha / 255})`;
        ctx.fillRect(col * scaleX, row * scaleY, scaleX + 1, scaleY + 1);
      }
    }
  }

  // Species trails
  const trailColors = [[25, 100, 100], [130, 85, 95], [195, 90, 100]];
  for (let s = 0; s < SPECIES.length; s++) {
    const trail = engine.trails[s];
    const col = trailColors[s];
    const [h, sat, bri] = col;
    for (let row = 0; row < fieldRows; row++) {
      for (let cc = 0; cc < fieldCols; cc++) {
        const val = trail.current[row * fieldCols + cc];
        const t = Math.min(val / 1.2, 1.0);
        if (t > 0.01) {
          const rgb = hsbToRgb(h, sat, bri * t);
          ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${t * 180 * alpha / 255})`;
          ctx.fillRect(cc * scaleX, row * scaleY, scaleX + 1, scaleY + 1);
        }
      }
    }
  }

  // Agents
  const state = engine.state;
  const n = state.count;
  for (let i = 0; i < n; i++) {
    const sid = engine.plugin.speciesId[i];
    const sp = SPECIES[sid];
    const col = trailColors[sid];
    const energy = engine.plugin.energy[i];
    const energyRatio = energy / sp.maxEnergy;
    const dormant = engine.plugin.isSpore[i];

    if (dormant) {
      ctx.fillStyle = `rgba(80,80,80,${0.4 * alpha})`;
    } else {
      const [h, sat, _] = col;
      const b = 50 + energyRatio * 50;
      const rgb = hsbToRgb(h, sat, b);
      ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${0.85 * alpha})`;
    }
    ctx.beginPath();
    ctx.arc(state.x[i], state.y[i], dormant ? 1.5 : 2.5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function hsbToRgb(h, s, b) {
  s /= 100; b /= 100;
  const k = (n) => (n + h / 60) % 6;
  const f = (n) => b * (1 - s * Math.max(0, Math.min(k(n), 4 - k(n), 1)));
  return [Math.round(255 * f(5)), Math.round(255 * f(3)), Math.round(255 * f(1))];
}

async function generateDiffMap(earlyDataUrl, lateDataUrl, canvasW, canvasH) {
  const canvas = document.createElement('canvas');
  canvas.width = canvasW;
  canvas.height = canvasH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });

  // Decode via fetch → blob → createImageBitmap (reliable in headless)
  const decode = async (dataUrl) => {
    const resp = await fetch(dataUrl);
    const blob = await resp.blob();
    return createImageBitmap(blob);
  };

  const [img1, img2] = await Promise.all([decode(earlyDataUrl), decode(lateDataUrl)]);

  ctx.drawImage(img1, 0, 0, canvasW, canvasH);
  const data1 = ctx.getImageData(0, 0, canvasW, canvasH);
  ctx.drawImage(img2, 0, 0, canvasW, canvasH);
  const data2 = ctx.getImageData(0, 0, canvasW, canvasH);

  const diffData = ctx.createImageData(canvasW, canvasH);
  for (let i = 0; i < data1.data.length; i += 4) {
    const dr = Math.abs(data1.data[i] - data2.data[i]);
    const dg = Math.abs(data1.data[i + 1] - data2.data[i + 1]);
    const db = Math.abs(data1.data[i + 2] - data2.data[i + 2]);
    const diff = (dr + dg + db) / 3;
    const t = Math.min(diff / 80, 1.0);
    diffData.data[i]     = Math.floor(t * 255);
    diffData.data[i + 1] = Math.floor((1 - t) * 40);
    diffData.data[i + 2] = Math.floor((1 - t) * 200);
    diffData.data[i + 3] = 200;
  }

  ctx.putImageData(diffData, 0, 0);
  img1.close();
  img2.close();
  return canvas.toDataURL('image/png');
}

async function buildComposite(slices, trailDataUrl, diffMapDataUrl, canvasW, canvasH, SPECIES) {
  const thumbW = Math.floor(canvasW / 2);
  const thumbH = Math.floor(canvasH / 2);
  const compositeW = canvasW * 2;
  const compositeCanvas = document.createElement('canvas');
  compositeCanvas.width = compositeW;
  compositeCanvas.height = canvasH;
  const compCtx = compositeCanvas.getContext('2d');

  compCtx.fillStyle = 'rgb(5, 5, 10)';
  compCtx.fillRect(0, 0, compositeW, canvasH);

  // Left: 4 time slices in 2x2 grid
  const labels = ['70% (后预热)', '80% (成熟期)', '90% (收敛期)', '100% (最终态)'];
  for (let i = 0; i < slices.length; i++) {
    const img = await loadImage(slices[i]);
    const dx = (i % 2) * thumbW;
    const dy = Math.floor(i / 2) * thumbH;
    compCtx.drawImage(img, dx, dy, thumbW, thumbH);

    compCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
    compCtx.fillRect(dx + 5, dy + 5, 130, 28);
    compCtx.fillStyle = '#fff';
    compCtx.font = 'bold 14px monospace';
    compCtx.fillText(`Time: ${labels[i]}`, dx + 12, dy + 22);

    if (i === 0) {
      let ly = dy + 40;
      for (const sp of SPECIES) {
        const rgb = hsbToRgb(sp.color[0], sp.color[1], sp.color[2]);
        compCtx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        compCtx.fillText(`● ${sp.name}`, dx + 12, ly);
        ly += 16;
      }
    }
  }

  // Right top: trail
  const halfH = Math.floor(canvasH / 2);
  const trailImg = await loadImage(trailDataUrl);
  compCtx.drawImage(trailImg, canvasW, 0, canvasW, halfH);
  compCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  compCtx.fillRect(canvasW + 5, 5, 200, 28);
  compCtx.fillStyle = '#fff';
  compCtx.font = 'bold 14px monospace';
  compCtx.fillText('Trail (80 frames)', canvasW + 12, 22);

  // Right bottom: diff map
  const diffImg = await loadImage(diffMapDataUrl);
  compCtx.drawImage(diffImg, canvasW, halfH, canvasW, halfH);
  compCtx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  compCtx.fillRect(canvasW + 5, halfH + 5, 280, 28);
  compCtx.fillStyle = '#fff';
  compCtx.font = 'bold 14px monospace';
  compCtx.fillText('Δ Map (70%→100%): blue=stable red=moved', canvasW + 12, halfH + 22);

  return compositeCanvas.toDataURL('image/png');
}

async function loadImage(dataUrl) {
  const resp = await fetch(dataUrl);
  const blob = await resp.blob();
  return createImageBitmap(blob);
}

// ═══════════════════════════════════════
// Stats helpers
// ═══════════════════════════════════════

function computeAvgEnergyForSpecies(plugin, speciesId) {
  let total = 0, count = 0;
  for (let i = 0; i < plugin.energy.length; i++) {
    if (plugin.speciesId[i] === speciesId) {
      total += plugin.energy[i];
      count++;
    }
  }
  return count > 0 ? total / count : 0;
}

function computeTotalWaste(wasteField) {
  let total = 0;
  for (let i = 0; i < wasteField.size; i++) {
    total += wasteField.current[i];
  }
  return total;
}

function computeBalance(pops) {
  const total = pops.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  const ideal = total / pops.length;
  const deviation = pops.reduce((sum, p) => sum + Math.abs(p - ideal), 0) / total;
  return 1 - deviation;
}

function variance(arr) {
  if (arr.length < 2) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.reduce((sum, v) => sum + (v - mean) ** 2, 0) / arr.length;
}

function trend(arr) {
  if (arr.length < 4) return 'stable';
  const half = Math.floor(arr.length / 2);
  const firstHalf = arr.slice(0, half).reduce((a, b) => a + b, 0) / half;
  const secondHalf = arr.slice(half).reduce((a, b) => a + b, 0) / (arr.length - half);
  const diff = (secondHalf - firstHalf) / (firstHalf || 1);
  if (diff > 0.1) return 'rising';
  if (diff < -0.1) return 'falling';
  return 'stable';
}
