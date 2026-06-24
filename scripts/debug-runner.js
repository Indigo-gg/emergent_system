import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({
  headless: true,
  executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

const page = await browser.newPage();

page.on('console', msg => console.log(`[${msg.type()}] ${msg.text()}`));
page.on('pageerror', err => console.log(`[pageerror] ${err.message}`));

console.log('Loading explorer.html...');
await page.goto('http://localhost:5174/explorer.html', { waitUntil: 'networkidle0', timeout: 30000 });

console.log('Waiting for function...');
await page.waitForFunction('typeof window.runEmergenceExperiment === "function"', { timeout: 10000 });
console.log('Function found. Running with small params...');

try {
  const result = await page.evaluate(async (p) => {
    return window.runEmergenceExperiment(p);
  }, {
    canvasW: 200, canvasH: 200, seed: 42,
    simFrames: 100,
    initialPerSpecies: 20,
    trailDiffuseRate: 0.2, trailDecayRate: 0.97,
    nutrientDiffuseRate: 0.08, nutrientDecayRate: 0.999,
    nutrientInjectInterval: 60, nutrientPatchCount: 3,
    nutrientPatchRadius: 5, nutrientPatchAmount: 1.5,
    nutrientDriftSpeed: 0.001,
    wasteDiffuseRate: 0.04, wasteDecayRate: 0.998,
    wasteProductionRate: 0.5, wasteMetabolismFactor: 2.5,
    wasteRepelStrength: 1.2,
    terrainEnabled: true, terrainSeed: 123,
    terrainScale: 0.04, terrainWallThreshold: 0.7, terrainHarshThreshold: 0.55,
    fieldCols: 50, fieldRows: 50,
  });
  console.log('SUCCESS! Stats:', JSON.stringify(result.stats, null, 2));
} catch (err) {
  console.error('FAILED:', err.message);
}

await browser.close();
