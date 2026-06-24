/**
 * Puppeteer-based experiment runner.
 *
 * Key optimization: browser is launched ONCE and reused for all runs.
 * This eliminates 10-15 seconds of startup overhead per run.
 */
import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';

let _browser = null;
let _page = null;

/**
 * Launch browser once. Called by orchestrator before first run.
 */
export async function launchBrowser(vitePort) {
  if (_browser) return;

  console.log('[Browser] Launching Chrome...');
  _browser = await puppeteer.launch({
    headless: true,
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });

  _page = await _browser.newPage();

  _page.on('console', msg => {
    if (msg.type() === 'error') console.error('[browser]', msg.text());
  });
  _page.on('pageerror', err => console.error('[pageerror]', err.message));

  console.log(`[Browser] Loading explorer.html from port ${vitePort}...`);
  await _page.goto(`http://localhost:${vitePort}/explorer.html`, {
    waitUntil: 'networkidle0',
    timeout: 30000,
  });

  await _page.waitForFunction('typeof window.runEmergenceExperiment === "function"', { timeout: 10000 });
  _page.setDefaultTimeout(300000);

  console.log('[Browser] Ready.');
}

/**
 * Close browser. Called by orchestrator after all runs.
 */
export async function closeBrowser() {
  if (_browser) {
    await _browser.close();
    _browser = null;
    _page = null;
  }
}

/**
 * Run a single experiment. Reuses the existing browser page.
 */
export async function runExperiment(params, roundId, options = {}) {
  const runsDir = options.runsDir || path.resolve('runs');
  const runId = String(roundId).padStart(4, '0');
  const runDir = path.join(runsDir, runId);
  fs.mkdirSync(runDir, { recursive: true });

  if (!_page) {
    throw new Error('Browser not launched. Call launchBrowser() first.');
  }

  // 1. Save params with full derived runtime config
  const derived = {
    canvasW: 900,
    canvasH: 900,
    seed: 42,
    simFrames: params.simFrames || 15000,
    initialPerSpecies: params.initialPerSpecies || 80,
    fieldCols: 250,
    fieldRows: 250,
    dt: 1 / 60,
  };
  const paramsFile = {
    runId,
    timestamp: new Date().toISOString(),
    round: roundId,
    params,
    derived,
  };
  fs.writeFileSync(
    path.join(runDir, 'params.json'),
    JSON.stringify(paramsFile, null, 2)
  );

  // 1b. Save config-snapshot.json for 100% reproducibility
  const configSnapshot = {
    ...derived,
    ecosystem: {
      trailDiffuseRate: params.trailDiffuseRate ?? 0.2,
      trailDecayRate: params.trailDecayRate ?? 0.97,
      nutrientDiffuseRate: params.nutrientDiffuseRate ?? 0.08,
      nutrientDecayRate: params.nutrientDecayRate ?? 0.999,
      nutrientInjectInterval: params.nutrientInjectInterval ?? 60,
      nutrientPatchCount: params.nutrientPatchCount ?? 5,
      nutrientPatchRadius: params.nutrientPatchRadius ?? 10,
      nutrientPatchAmount: params.nutrientPatchAmount ?? 1.5,
      nutrientDriftSpeed: params.nutrientDriftSpeed ?? 0.0008,
      wasteDiffuseRate: params.wasteDiffuseRate ?? 0.06,
      wasteDecayRate: params.wasteDecayRate ?? 0.995,
      wasteProductionRate: params.wasteProductionRate ?? 0.15,
      wasteMetabolismFactor: params.wasteMetabolismFactor ?? 2.0,
      wasteRepelStrength: params.wasteRepelStrength ?? 0.8,
      terrainEnabled: params.terrainEnabled ?? true,
      terrainSeed: params.terrainSeed ?? 123,
      terrainScale: params.terrainScale ?? 0.04,
      terrainWallThreshold: params.terrainWallThreshold ?? 0.7,
      terrainHarshThreshold: params.terrainHarshThreshold ?? 0.55,
    },
  };
  fs.writeFileSync(
    path.join(runDir, 'config-snapshot.json'),
    JSON.stringify(configSnapshot, null, 2)
  );

  // 2. Run simulation (reuse existing page)
  const fullParams = { ...params, ...derived };
  let result;
  try {
    result = await _page.evaluate(async (p) => {
      return window.runEmergenceExperiment(p);
    }, fullParams);
  } catch (err) {
    console.error('[Puppeteer] evaluate failed:', err?.message || err || 'unknown');
    // Try to get more info from the page
    try {
      const pageError = await _page.evaluate(() => {
        try { return window.runEmergenceExperiment.toString().slice(0, 200); }
        catch(e) { return 'function not found'; }
      });
      console.error('[Puppeteer] function check:', pageError);
    } catch(_) {}
    throw err;
  }

  // 3. Save screenshots
  const sliceLabels = ['t70', 't80', 't90', 't100', 'trail'];
  for (let i = 0; i < result.screenshots.length; i++) {
    const label = sliceLabels[i] || `slice${i}`;
    const base64 = result.screenshots[i].replace(/^data:image\/png;base64,/, '');
    fs.writeFileSync(path.join(runDir, `screenshot_${label}.png`), base64, 'base64');
  }

  // 4. Save composite
  const compositeData = result.composite_screenshot.replace(/^data:image\/png;base64,/, '');
  fs.writeFileSync(path.join(runDir, 'screenshot_composite.png'), compositeData, 'base64');

  // 5. Save stats
  fs.writeFileSync(
    path.join(runDir, 'stats.json'),
    JSON.stringify(result.stats, null, 2)
  );

  return {
    runId,
    runDir,
    composite_screenshot: result.composite_screenshot,
    screenshots: result.screenshots,
    stats: result.stats,
  };
}
