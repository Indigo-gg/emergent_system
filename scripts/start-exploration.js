#!/usr/bin/env node

/**
 * Emergence Explorer — Entry point
 *
 * Usage:
 *   MIMO_API_KEY=your-key node scripts/start-exploration.js
 *
 * Prerequisites:
 *   1. npm install puppeteer
 *   2. npm run dev (in another terminal — Vite must be running)
 *
 * The script runs a three-phase exploration:
 *   Phase A: 20 random samples (baseline)
 *   Phase B: 30 LLM-guided samples (find interesting regions)
 *   Phase C: 50 genetic algorithm samples (fine-search)
 *
 * All results are saved to runs/ and checkpoints/ with full traceability.
 */
import { Orchestrator } from '../src/explorer/orchestrator.js';

const apiKey = process.env.MIMO_API_KEY;
if (!apiKey) {
  console.error('Error: MIMO_API_KEY environment variable is required.');
  console.error('Usage: MIMO_API_KEY=your-key node scripts/start-exploration.js');
  process.exit(1);
}

const orchestrator = new Orchestrator({
  seed: 42,
  vitePort: 5174,
  llm: {
    apiKey,
    baseUrl: 'https://token-plan-sgp.xiaomimimo.com/v1',
    model: 'mimo-v2.5',
  },
});

console.log('╔══════════════════════════════════════╗');
console.log('║   Emergence Explorer v3              ║');
console.log('║   MiMo v2.5 Multimodal Evaluator     ║');
console.log('╚══════════════════════════════════════╝');
console.log(`Session: ${orchestrator.sessionName}`);
console.log(`Budget: ${orchestrator.state.budget.maxRuns} runs`);
console.log(`LLM: mimo-v2.5 @ api.xiaomimimo.com`);
console.log(`Output: ${orchestrator.RUNS_DIR}`);
console.log('');

try {
  orchestrator.state.budget.maxRuns = 250;  // ~3-4 hours at ~50s/run
  await orchestrator.runAll();
} catch (err) {
  console.error('\n[ERROR]', err.message);
  console.error('State saved. Re-run to resume from last checkpoint.');
  process.exit(1);
}
