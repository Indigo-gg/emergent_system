import p5 from 'p5';

/**
 * Slime Mold Renderer
 *
 * Draws the chemical field as a colored background,
 * with agents as bright dots on top.
 *
 * Press [D] to toggle debug overlay.
 * Press [C] to clear the field (reset trails).
 */
export function createSlimeMoldRenderer(engine, config) {
  return (p) => {
    let img; // offscreen buffer for the field

    p.setup = function () {
      p.createCanvas(config.canvasW, config.canvasH);
      p.pixelDensity(1);

      // Create offscreen image at field resolution
      img = p.createImage(engine.field.cols, engine.field.rows);
    };

    p.draw = function () {
      const frameTime = p.deltaTime / 1000;
      engine.update(frameTime);

      // --- Render chemical field to image ---
      const field = engine.field;
      img.loadPixels();

      for (let row = 0; row < field.rows; row++) {
        for (let col = 0; col < field.cols; col++) {
          const val = field.current[row * field.cols + col];
          const idx = (row * field.cols + col) * 4;

          // Color mapping: dark → teal → white
          const t = Math.min(val / 1.5, 1.0);
          const r = Math.floor(t * t * 255);
          const g = Math.floor(t * 200 + (1 - t) * 20);
          const b = Math.floor((1 - t * 0.5) * 80 + t * 200);

          img.pixels[idx]     = r;
          img.pixels[idx + 1] = g;
          img.pixels[idx + 2] = b;
          img.pixels[idx + 3] = 255;
        }
      }

      img.updatePixels();

      // Draw field scaled to canvas size
      p.noSmooth();
      p.image(img, 0, 0, config.canvasW, config.canvasH);

      // --- Draw agents on top ---
      const state = engine.state;
      const n = state.count;

      p.noStroke();
      for (let i = 0; i < n; i++) {
        // Agent color based on heading
        const heading = engine.plugin.headings ? engine.plugin.headings[i] : 0;
        const hue = p.map(heading, -Math.PI, Math.PI, 0, 360);
        p.fill(hue, 80, 100, 60);
        p.circle(state.x[i], state.y[i], 2);
      }

      // --- HUD ---
      p.noStroke();
      p.fill(0, 0, 70, 60);
      p.textSize(11);
      p.textAlign(p.LEFT);
      p.text(`agents: ${n}  ticks: ${engine.tickCount}`, 12, config.canvasH - 40);
      p.text('[D] debug  [C] clear field  [SPACE] pause', 12, config.canvasH - 22);

      // --- Debug: show sensor rays for first agent ---
      if (config.debugMode && engine.plugin.headings) {
        drawDebugSensors(p, engine, config);
      }
    };

    p.keyPressed = function () {
      if (p.key === 'd' || p.key === 'D') {
        config.debugMode = !config.debugMode;
      }
      if (p.key === 'c' || p.key === 'C') {
        engine.field.clear();
      }
    };
  };
}

/** Draw sensor rays for the first few agents. */
function drawDebugSensors(p, engine, config) {
  const state = engine.state;
  const plugin = engine.plugin;
  const sm = config.slimeMold;
  const n = Math.min(state.count, 5);

  const w = config.canvasW;
  const h = config.canvasH;

  for (let i = 0; i < n; i++) {
    const x = state.x[i];
    const y = state.y[i];
    const heading = plugin.headings[i];

    const cosH = Math.cos(heading);
    const sinH = Math.sin(heading);
    const d = sm.sensorDist * w; // scale to canvas

    // Center
    p.stroke(0, 100, 100, 50);
    p.strokeWeight(1);
    p.line(x, y, x + sinH * d, y - cosH * d);

    // Left
    const la = heading - sm.sensorAngle;
    p.stroke(120, 100, 100, 50);
    p.line(x, y, x + Math.sin(la) * d, y - Math.cos(la) * d);

    // Right
    const ra = heading + sm.sensorAngle;
    p.stroke(0, 100, 100, 50);
    p.line(x, y, x + Math.sin(ra) * d, y - Math.cos(ra) * d);
  }
  p.noStroke();
}
