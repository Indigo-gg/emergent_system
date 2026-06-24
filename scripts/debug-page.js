import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({
  headless: true,
  executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  args: ['--no-sandbox', '--disable-setuid-sandbox'],
});

const page = await browser.newPage();

page.on('console', msg => console.log(`[console.${msg.type()}] ${msg.text()}`));
page.on('pageerror', err => console.log(`[pageerror] ${err.message}`));
page.on('requestfailed', req => console.log(`[requestfailed] ${req.url()} ${req.failure()?.errorText}`));
page.on('response', res => {
  if (res.status() >= 400) console.log(`[${res.status()}] ${res.url()}`);
});

console.log('Navigating to http://localhost:5174/explorer.html ...');
try {
  await page.goto('http://localhost:5174/explorer.html', { waitUntil: 'networkidle0', timeout: 15000 });
  console.log('Page loaded. URL:', page.url());
  console.log('Title:', await page.title());

  const hasFn = await page.evaluate(() => typeof window.runEmergenceExperiment);
  console.log('typeof runEmergenceExperiment:', hasFn);
} catch (e) {
  console.error('Navigation error:', e.message);
}

await browser.close();
