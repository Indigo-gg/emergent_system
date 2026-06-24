import p5 from 'p5';
import { SPECIES } from './species.js';

/**
 * Ecosystem Renderer — enhanced visuals.
 *
 * Features:
 *   - Warm nutrient field with smooth gradient
 *   - Vibrant species trails with glow
 *   - Agent dots with energy-based brightness + halos
 *   - Intra-species connection lines
 *   - Clean HUD with population bars
 */
export function createEcosystemRenderer(engine, config) {
  let nutrientImg;
  let wasteImg;
  let terrainImg;
  let trailImgs = [];
  let glowLayer;

  return (p) => {
    p.setup = function () {
      p.createCanvas(config.canvasW, config.canvasH);
      p.pixelDensity(1);
      p.colorMode(p.HSB, 360, 100, 100, 100);

      nutrientImg = p.createImage(config.fieldCols, config.fieldRows);
      wasteImg = p.createImage(config.fieldCols, config.fieldRows);
      terrainImg = p.createImage(config.fieldCols, config.fieldRows);
      for (let i = 0; i < SPECIES.length; i++) {
        trailImgs.push(p.createImage(config.fieldCols, config.fieldRows));
      }

      // Glow buffer
      glowLayer = p.createGraphics(config.canvasW, config.canvasH);
      glowLayer.colorMode(p.HSB, 360, 100, 100, 100);
      glowLayer.noStroke();
    };

    p.draw = function () {
      const frameTime = p.deltaTime / 1000;
      engine.update(frameTime);

      const fieldCols = config.fieldCols;
      const fieldRows = config.fieldRows;

      // Slightly brighter dark background
      p.background(240, 30, 8);

      // --- Render terrain (walls + harsh zones) ---
      if (engine.terrain) {
        const eco = config.ecosystem;
        terrainImg.loadPixels();
        for (let row = 0; row < fieldRows; row++) {
          for (let col = 0; col < fieldCols; col++) {
            const val = engine.terrain[row * fieldCols + col];
            const idx = (row * fieldCols + col) * 4;
            if (val > (eco.terrainWallThreshold || 0.7)) {
              // Wall: dark rocky gray
              terrainImg.pixels[idx]     = 45;
              terrainImg.pixels[idx + 1] = 40;
              terrainImg.pixels[idx + 2] = 35;
              terrainImg.pixels[idx + 3] = 255;
            } else if (val > (eco.terrainHarshThreshold || 0.55)) {
              // Harsh zone: dim orange-brown
              terrainImg.pixels[idx]     = 60;
              terrainImg.pixels[idx + 1] = 35;
              terrainImg.pixels[idx + 2] = 15;
              terrainImg.pixels[idx + 3] = 200;
            } else {
              // Safe: transparent
              terrainImg.pixels[idx]     = 0;
              terrainImg.pixels[idx + 1] = 0;
              terrainImg.pixels[idx + 2] = 0;
              terrainImg.pixels[idx + 3] = 0;
            }
          }
        }
        terrainImg.updatePixels();
        p.push();
        p.blendMode(p.BLEND);
        p.noSmooth();
        p.image(terrainImg, 0, 0, config.canvasW, config.canvasH);
        p.pop();
      }

      // --- Render nutrient field (warm amber glow) ---
      const nf = engine.nutrientField;
      nutrientImg.loadPixels();
      for (let row = 0; row < fieldRows; row++) {
        for (let col = 0; col < fieldCols; col++) {
          const val = nf.current[row * fieldCols + col];
          const t = Math.min(val / 2.0, 1.0);
          const idx = (row * fieldCols + col) * 4;
          // Warm amber: HSB(40, 80-100%, brightness)
          nutrientImg.pixels[idx]     = Math.floor(40 + t * 10);  // R (warm)
          nutrientImg.pixels[idx + 1] = Math.floor(t * 200);       // G
          nutrientImg.pixels[idx + 2] = Math.floor(t * 60);        // B
          nutrientImg.pixels[idx + 3] = Math.floor(t * 160);       // A
        }
      }
      nutrientImg.updatePixels();

      p.push();
      p.blendMode(p.ADD);
      p.noSmooth();
      p.image(nutrientImg, 0, 0, config.canvasW, config.canvasH);

      // --- Render waste field (toxic red-purple glow) ---
      const wf = engine.wasteField;
      wasteImg.loadPixels();
      for (let row = 0; row < fieldRows; row++) {
        for (let col = 0; col < fieldCols; col++) {
          const val = wf.current[row * fieldCols + col];
          const t = Math.min(val / 1.0, 1.0);
          const idx = (row * fieldCols + col) * 4;
          // Toxic red-purple
          wasteImg.pixels[idx]     = Math.floor(t * 220);   // R
          wasteImg.pixels[idx + 1] = Math.floor(t * 40);    // G
          wasteImg.pixels[idx + 2] = Math.floor(t * 100);   // B
          wasteImg.pixels[idx + 3] = Math.floor(t * 200);   // A
        }
      }
      wasteImg.updatePixels();
      // Draw waste with normal blend (not ADD) for visibility
      p.blendMode(p.BLEND);
      p.image(wasteImg, 0, 0, config.canvasW, config.canvasH);
      p.blendMode(p.ADD);

      // --- Render species trails (vibrant colors) ---
      const trailColors = [
        [25, 100, 100],   // Explorer: orange-gold
        [130, 85, 95],    // Grazer: emerald green
        [195, 90, 100],   // Harvester: cyan-blue
      ];

      for (let s = 0; s < SPECIES.length; s++) {
        const trail = engine.trails[s];
        const col = trailColors[s];
        const img = trailImgs[s];

        img.loadPixels();
        for (let row = 0; row < fieldRows; row++) {
          for (let cc = 0; cc < fieldCols; cc++) {
            const val = trail.current[row * fieldCols + cc];
            const t = Math.min(val / 1.2, 1.0);
            const idx = (row * fieldCols + cc) * 4;

            // HSB to RGB conversion inline
            const h = col[0] / 60;
            const s2 = col[1] / 100;
            const v = col[2] / 100 * t;
            const c = v * s2;
            const x = c * (1 - Math.abs(h % 2 - 1));
            const m = v - c;
            let r, g, b;
            if (h < 1)      { r = c; g = x; b = 0; }
            else if (h < 2) { r = x; g = c; b = 0; }
            else if (h < 3) { r = 0; g = c; b = x; }
            else if (h < 4) { r = 0; g = x; b = c; }
            else if (h < 5) { r = x; g = 0; b = c; }
            else            { r = c; g = 0; b = x; }

            img.pixels[idx]     = Math.floor((r + m) * 255);
            img.pixels[idx + 1] = Math.floor((g + m) * 255);
            img.pixels[idx + 2] = Math.floor((b + m) * 255);
            img.pixels[idx + 3] = Math.floor(t * 200);
          }
        }
        img.updatePixels();
        p.image(img, 0, 0, config.canvasW, config.canvasH);
      }

      p.blendMode(p.BLEND);
      p.pop();

      // --- Glow layer for agents ---
      glowLayer.clear();
      const state = engine.state;
      const n = state.count;

      p.push();
      p.blendMode(p.ADD);
      for (let i = 0; i < n; i++) {
        const sid = engine.plugin.speciesId[i];
        const sp = SPECIES[sid];
        const col = trailColors[sid];
        const energy = engine.plugin.energy[i];
        const energyRatio = energy / sp.maxEnergy;
        const dormant = engine.plugin.isSpore[i];

        // Glow halo (dimmer for spores)
        const glowR = dormant ? 4 : (8 + energyRatio * 6);
        for (let l = 3; l >= 0; l--) {
          const r = glowR * (l / 3);
          const a = dormant ? p.map(l, 0, 3, 6, 1) : p.map(l, 0, 3, 20, 3);
          if (dormant) {
            glowLayer.fill(0, 0, 30, a); // gray glow for spores
          } else {
            glowLayer.fill(col[0], col[1] * 0.7, col[2] * energyRatio, a);
          }
          glowLayer.circle(state.x[i], state.y[i], r * 2);
        }
      }
      p.image(glowLayer, 0, 0);
      p.pop();

      // --- Draw agent cores ---
      p.noStroke();
      for (let i = 0; i < n; i++) {
        const sid = engine.plugin.speciesId[i];
        const sp = SPECIES[sid];
        const col = trailColors[sid];
        const energy = engine.plugin.energy[i];
        const energyRatio = energy / sp.maxEnergy;
        const dormant = engine.plugin.isSpore[i];

        if (dormant) {
          // Spore: small dim gray dot
          p.fill(0, 0, 40, 50);
          p.circle(state.x[i], state.y[i], 2);
          p.fill(0, 0, 60, 30);
          p.circle(state.x[i], state.y[i], 1);
        } else {
          const bri = 50 + energyRatio * 50;
          p.fill(col[0], col[1] * 0.8, bri, 85);
          p.circle(state.x[i], state.y[i], 3);

          // Bright center
          p.fill(col[0], col[1] * 0.4, 100, 50);
          p.circle(state.x[i], state.y[i], 1.5);
        }
      }

      // --- Hotspot indicators on canvas ---
      const hotspots = engine.getHotspotPositions();
      const pulse = 0.5 + 0.5 * Math.sin(engine.tickCount * 0.05);
      p.noStroke();
      const hsColors = [[40, 100, 100], [30, 90, 100], [50, 80, 100]];
      for (let hi = 0; hi < hotspots.length; hi++) {
        const hs = hotspots[hi];
        const hx = hs.x * config.canvasW;
        const hy = hs.y * config.canvasH;
        const col = hsColors[hi];
        p.fill(col[0], col[1], col[2], 6 + pulse * 8);
        p.circle(hx, hy, 60 + pulse * 15);
        p.fill(col[0], col[1] * 0.8, col[2], 12 + pulse * 10);
        p.circle(hx, hy, 30);
        p.fill(col[0], col[1] * 0.5, col[2], 25);
        p.circle(hx, hy, 6);
      }

      // --- HUD ---
      drawHUD(p, engine, config, trailColors);
    };
  };
}

function drawHUD(p, engine, config, trailColors) {
  const pops = engine.getPopulations();
  const state = engine.state;

  const barW = 90;
  const barH = 10;
  const startX = 16;
  let startY = 16;

  p.noStroke();
  p.textSize(11);
  p.textAlign(p.LEFT);
  p.fill(0, 0, 100, 90);

  const sporeCounts = engine.plugin.getSporeCounts ? engine.plugin.getSporeCounts() : [];

  for (let i = 0; i < SPECIES.length; i++) {
    const sp = SPECIES[i];
    const ratio = pops[i] / sp.carryingCapacity;
    const col = trailColors[i];
    const dormant = sporeCounts[i] || 0;

    // Species dot + name + count
    p.fill(col[0], col[1], col[2], 90);
    p.circle(startX + 5, startY + 5, 8);
    p.fill(0, 0, 90, 85);
    const dormantStr = dormant > 0 ? ` (${dormant} dormant)` : '';
    p.text(`${sp.name}: ${pops[i]}${dormantStr}`, startX + 14, startY + 10);

    // Bar background
    p.fill(240, 20, 15, 50);
    p.rect(startX + 85, startY, barW, barH, 3);

    // Bar fill
    p.fill(col[0], col[1], 80, 75);
    p.rect(startX + 85, startY, barW * Math.min(ratio, 1), barH, 3);

    startY += 20;
  }

  // Separator
  p.stroke(0, 0, 30, 20);
  p.strokeWeight(0.5);
  p.line(startX, startY, startX + 180, startY);
  p.noStroke();

  // Stats
  p.fill(0, 0, 60, 70);
  p.textSize(10);
  p.text(`Total: ${state.count}   Tick: ${engine.tickCount}`, startX, startY + 14);
  p.text(`[SPACE] pause   [R] reset`, startX, startY + 28);

  // Hotspot indicator (bottom-left)
  const hotspots = engine.getHotspotPositions();
  const hsX = startX;
  let hsY = config.canvasH - 55;
  p.fill(40, 80, 70, 60);
  p.textSize(9);
  p.text(`☀ ${hotspots.length} hotspots (drifting)`, hsX, hsY);

  // Mini-map showing hotspot positions + terrain
  const mmSize = 50;
  const mmX = startX;
  const mmY = hsY - mmSize - 8;
  p.fill(240, 20, 10, 30);
  p.rect(mmX, mmY, mmSize, mmSize, 3);

  // Terrain dots on minimap
  if (engine.terrain) {
    const eco = config.ecosystem;
    for (let my = 0; my < 5; my++) {
      for (let mx = 0; mx < 5; mx++) {
        const tIdx = Math.floor(my * config.fieldRows / 5) * config.fieldCols + Math.floor(mx * config.fieldCols / 5);
        const val = engine.terrain[tIdx];
        if (val > (eco.terrainWallThreshold || 0.7)) {
          p.fill(80, 70, 60, 40);
          p.rect(mmX + mx * 10, mmY + my * 10, 10, 10);
        }
      }
    }
  }

  const hsDotColors = [[40, 100, 100], [30, 90, 100], [50, 80, 100]];
  for (let hi = 0; hi < hotspots.length; hi++) {
    const col = hsDotColors[hi];
    p.fill(col[0], col[1], col[2], 80);
    p.circle(mmX + hotspots[hi].x * mmSize, mmY + hotspots[hi].y * mmSize, 5);
  }

  // Legend (bottom-right)
  p.textSize(9);
  p.fill(300, 60, 60, 50);
  p.text(`🟣 waste/toxin`, config.canvasW - 110, config.canvasH - 24);
  p.fill(60, 30, 10, 60);
  p.text(`🟫 harsh zone (2x metabolism)`, config.canvasW - 185, config.canvasH - 12);
}
