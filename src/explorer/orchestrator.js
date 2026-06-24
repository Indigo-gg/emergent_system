/**
 * Emergence Explorer — Orchestrator
 *
 * Runs in Node.js. Coordinates the three-phase exploration:
 *   Phase A: Random sampling (20 runs)
 *   Phase B: LLM-guided search (30 runs)
 *   Phase C: Genetic algorithm fine-search (50 runs)
 *
 * Manages experiment database, traceability, and state persistence.
 */
import fs from 'fs';
import path from 'path';
import { createRNG } from '../prng.js';
import { PARAM_SPACE } from './param-space.js';
import { RandomStrategy } from './strategies/random.js';
import { LLMGuidedStrategy } from './strategies/llm-guided.js';
import { GeneticStrategy } from './strategies/genetic.js';
import { runExperiment, launchBrowser, closeBrowser } from './puppeteer-runner.js';
import { evaluateScreenshot } from './llm-evaluator.js';

const BASE_RUNS_DIR = path.resolve('runs');
const BASE_CHECKPOINT_DIR = path.resolve('checkpoints');

/**
 * [RISK-2] Post-evaluation hard constraint enforcement.
 * Clamps LLM scores when objective metrics violate hard rules.
 * Prevents "confabulatory pareidolia" — LLM seeing beauty in broken systems.
 */
function enforceHardConstraints(evalResult, stats) {
  const scores = evalResult.scores;
  if (!scores) return;

  const pops = stats.populations || [];
  const total = stats.totalPopulation || 1;
  const tv = stats.temporal_variance || {};

  // --- Species Coexistence constraints ---
  // If any species is functionally extinct (< 5% of total), cap at 4
  // If any species dominates (> 80%), cap at 3
  for (let i = 0; i < pops.length; i++) {
    const ratio = pops[i] / total;
    const name = ['Explorer', 'Grazer', 'Harvester'][i];
    if (ratio < 0.05 && scores.species_coexistence > 4) {
      console.log(`[HardConstraint] ${name} at ${(ratio * 100).toFixed(1)}% → species_coexistence clamped ${scores.species_coexistence} → 4`);
      scores.species_coexistence = 4;
    }
    if (ratio > 0.8 && scores.species_coexistence > 3) {
      console.log(`[HardConstraint] ${name} at ${(ratio * 100).toFixed(1)}% → species_coexistence clamped ${scores.species_coexistence} → 3`);
      scores.species_coexistence = 3;
    }
  }

  // species_balance < 0.3 means severe imbalance → cap at 4
  if (tv.species_balance !== undefined && tv.species_balance < 0.3 && scores.species_coexistence > 4) {
    console.log(`[HardConstraint] species_balance=${tv.species_balance.toFixed(3)} → species_coexistence clamped ${scores.species_coexistence} → 4`);
    scores.species_coexistence = 4;
  }

  // --- Structural Stability constraints ---
  // High population variance means system is unstable → cap at 4
  if (tv.population_variance !== undefined && tv.population_variance > 100 && scores.structural_stability > 4) {
    console.log(`[HardConstraint] population_variance=${tv.population_variance.toFixed(1)} → structural_stability clamped ${scores.structural_stability} → 4`);
    scores.structural_stability = 4;
  }

  // Population falling → ecosystem collapsing → stability cap at 3
  if (tv.population_trend === 'falling' && scores.structural_stability > 3) {
    console.log(`[HardConstraint] population_trend=falling → structural_stability clamped ${scores.structural_stability} → 3`);
    scores.structural_stability = 3;
  }

  // --- Recompute composite_score after clamping ---
  const weights = { species_coexistence: 0.2, spatial_organization: 0.2, nutrient_network: 0.15, structural_stability: 0.15, dynamic_complexity: 0.15, aesthetic_emergence: 0.15 };
  let weightedSum = 0, weightTotal = 0;
  for (const [key, w] of Object.entries(weights)) {
    if (scores[key] !== undefined) {
      weightedSum += scores[key] * w;
      weightTotal += w;
    }
  }
  if (weightTotal > 0) {
    const newComposite = Math.round((weightedSum / weightTotal) * 100) / 100;
    if (newComposite !== evalResult.composite_score) {
      console.log(`[HardConstraint] composite_score recomputed: ${evalResult.composite_score} → ${newComposite}`);
      evalResult.composite_score = newComposite;
    }
  }

  // --- Recompute emergence_level from composite ---
  const cs = evalResult.composite_score;
  if (cs >= 8.5) evalResult.emergence_level = 'extraordinary';
  else if (cs >= 7.0) evalResult.emergence_level = 'strong';
  else if (cs >= 5.0) evalResult.emergence_level = 'moderate';
  else if (cs >= 3.0) evalResult.emergence_level = 'weak';
  else evalResult.emergence_level = 'none';
}

export class Orchestrator {
  constructor(config) {
    this.config = config;
    this.rng = createRNG(config.seed || 42);

    // Each run gets its own session directory
    this.sessionName = `exp-${Date.now()}`;
    this.RUNS_DIR = path.join(BASE_RUNS_DIR, this.sessionName);
    this.CHECKPOINT_DIR = path.join(BASE_CHECKPOINT_DIR, this.sessionName);
    fs.mkdirSync(this.RUNS_DIR, { recursive: true });
    fs.mkdirSync(this.CHECKPOINT_DIR, { recursive: true });

    this.state = this.loadOrCreateState();
    this.previousRuns = this.loadAllRuns();
    this.geneticStrategy = null;
  }

  /**
   * Three-phase exploration: random → llm-guided → genetic.
   */
  async runAll() {
    const totalBudget = this.state.budget.maxRuns;
    const phaseA = Math.min(50, Math.floor(totalBudget * 0.15));
    const phaseB = Math.floor(totalBudget * 0.35);
    const phaseC = totalBudget - phaseA - phaseB;

    // Launch browser once for all runs
    await launchBrowser(this.config.vitePort || 5173);

    try {
      console.log(`=== Phase A: Latin Hypercube Sampling (${phaseA} runs) ===`);
      this.state.strategy = 'random';
      await this.run(phaseA);

      console.log(`\n=== Phase B: LLM-Guided Search (${phaseB} runs) ===`);
      this.state.strategy = 'llm-guided';
      await this.run(phaseB);

      console.log(`\n=== Phase C: Genetic Fine-Search (${phaseC} runs) ===`);
      this.state.strategy = 'genetic';
      await this.run(phaseC);

      this.printSummary();
    } finally {
      await closeBrowser();
    }
  }

  async run(maxRounds = 50) {
    const remaining = this.state.budget.maxRuns - this.state.budget.runsUsed;
    console.log(`[Orchestrator] Strategy: ${this.state.strategy}, Budget: ${remaining} remaining`);

    const strategy = this.createStrategy();

    for (let round = 0; round < maxRounds; round++) {
      if (this.state.budget.runsUsed >= this.state.budget.maxRuns) {
        console.log('[Orchestrator] Budget exhausted.');
        break;
      }

      const roundId = this.state.lastRunId + 1;
      console.log(`\n--- Round ${roundId} (${this.state.strategy}) ---`);

      // 1. Get next parameters
      let params;
      if (this.state.strategy === 'llm-guided' && this.previousRuns.length >= 3) {
        let suggestion = null;
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            suggestion = await strategy.next(this.previousRuns);
            break;
          } catch (err) {
            console.error(`[LLM-Guidance] Attempt ${attempt} failed: ${err?.message || err}`);
            if (attempt < 3) await new Promise(r => setTimeout(r, attempt * 3000));
          }
        }
        if (suggestion) {
          params = suggestion.params;
          console.log(`[LLM] Reasoning: ${suggestion.reasoning}`);
        } else {
          console.log('[LLM-Guidance] All attempts failed, falling back to random.');
          params = new RandomStrategy(PARAM_SPACE, this.rng).next();
        }
      } else if (this.state.strategy === 'genetic') {
        params = strategy.next();
        if (!params) {
          console.log(`[GA] Generation ${this.geneticStrategy.generation} complete. Evolving...`);
          this.geneticStrategy.evolve(this.geneticStrategy.population);
          params = strategy.next();
        }
        console.log(`[GA] Gen ${this.geneticStrategy.generation}`);
      } else {
        params = strategy.next();
        console.log(`[Random] waste=${params.wasteProductionRate?.toFixed(2)}, drift=${params.nutrientDriftSpeed?.toFixed(3)}, decay=${params.trailDecayRate?.toFixed(3)}`);
      }

      if (!params) break;

      // 2. Run simulation
      console.log('[Runner] Running simulation...');
      let runResult;
      try {
        runResult = await runExperiment(
          params, roundId, { vitePort: this.config.vitePort || 5173, runsDir: this.RUNS_DIR }
        );
      } catch (err) {
        console.error(`[Runner] Simulation failed: ${err?.message || err}. Skipping round ${roundId}.`);
        continue;
      }
      const { runId, runDir, composite_screenshot, stats } = runResult;
      const pops = stats.populations || [];
      const spores = stats.sporeCounts || [];
      const names = ['Explorer', 'Grazer', 'Harvester'];
      const popStr = pops.map((p, i) => {
        const s = spores[i] > 0 ? `(${spores[i]}sp)` : '';
        return `${names[i]}=${p}${s}`;
      }).join(', ');
      const waste = stats.temporal_variance?.waste_trend || '?';
      const hpCount = stats.hotspots?.length || 0;
      console.log(`[Runner] ${popStr} | total=${stats.totalPopulation} | waste=${waste} | hotspots=${hpCount}`);

      // 3. LLM evaluation (with retry)
      let llmResult = null;
      let lastRawResponse = null;
      let lastLatencyMs = null;
      let lastTokenUsage = null;
      for (let attempt = 1; attempt <= 3; attempt++) {
        try {
          console.log(`[LLM] Evaluating (attempt ${attempt}/3)...`);
          llmResult = await evaluateScreenshot(
            composite_screenshot, params, stats, this.previousRuns, this.config.llm
          );
          break; // success
        } catch (err) {
          const msg = err?.message || String(err);
          console.error(`[LLM] Attempt ${attempt} failed: ${msg}`);
          // Capture raw response even on parse failure for debugging
          if (err.raw_response) lastRawResponse = err.raw_response;
          if (err.latency_ms) lastLatencyMs = err.latency_ms;
          if (err.token_usage) lastTokenUsage = err.token_usage;
          if (attempt < 3) {
            const delay = attempt * 5000; // 5s, 10s backoff
            console.log(`[LLM] Retrying in ${delay / 1000}s...`);
            await new Promise(r => setTimeout(r, delay));
          }
        }
      }

      if (!llmResult) {
        console.error(`[LLM] All 3 attempts failed. Skipping round ${roundId}.`);
        // Save error record with raw response for debugging
        fs.writeFileSync(
          path.join(runDir, 'llm-response.json'),
          JSON.stringify({
            runId, error: 'LLM evaluation failed after 3 attempts',
            timestamp: new Date().toISOString(),
            raw_response: lastRawResponse,
            latency_ms: lastLatencyMs,
            token_usage: lastTokenUsage,
          }, null, 2)
        );
        continue; // skip to next round
      }

      const evalResult = llmResult.parsed;

      // 3b. [RISK-2] Enforce hard math constraints post-evaluation
      enforceHardConstraints(evalResult, stats);

      console.log(`[LLM] Score: ${evalResult.composite_score}, Level: ${evalResult.emergence_level}`);
      console.log(`[LLM] Description: ${evalResult.description}`);
      if (evalResult.is_novel) {
        console.log(`[LLM] ★ NOVEL: ${evalResult.novelty_note}`);
      }

      // 4. Save LLM response (with full traceability)
      fs.writeFileSync(
        path.join(runDir, 'llm-response.json'),
        JSON.stringify({
          runId,
          timestamp: new Date().toISOString(),
          model: this.config.llm.model,
          prompt_version: 'v3',
          raw_response: llmResult.raw_response,
          latency_ms: llmResult.latency_ms,
          token_usage: llmResult.token_usage,
          parsed: evalResult,
        }, null, 2)
      );

      // 5. Append to manifest
      fs.appendFileSync(
        path.join(this.RUNS_DIR, 'manifest.jsonl'),
        JSON.stringify({
          runId,
          timestamp: new Date().toISOString(),
          params,
          composite_score: evalResult.composite_score,
          emergence_level: evalResult.emergence_level,
          description: evalResult.description,
          is_novel: evalResult.is_novel || false,
          populations: stats.populations,
          totalPopulation: stats.totalPopulation,
          species_balance: stats.temporal_variance?.species_balance,
          prompt_version: 'v3',
          llm_model: this.config.llm.model,
          strategy: this.state.strategy,
        }) + '\n'
      );

      // 6. Update state
      const runRecord = {
        runId, params,
        score: evalResult.composite_score,
        level: evalResult.emergence_level,
        description: evalResult.description,
      };
      this.previousRuns.push(runRecord);

      // GA: write score back to population
      if (this.state.strategy === 'genetic' && this.geneticStrategy) {
        const individual = this.geneticStrategy.population.find(p => p.score === 0);
        if (individual) individual.score = evalResult.composite_score;
      }

      this.state.lastRunId = roundId;
      this.state.budget.runsUsed++;
      this.state.budget.llmCallsUsed++;
      if (evalResult.composite_score > this.state.bestScore) {
        this.state.bestScore = evalResult.composite_score;
        this.state.bestRunId = roundId;
      }
      this.saveState();

      if (evalResult.composite_score >= 7.0) {
        this.recordBestResult(runId, evalResult, params, stats);
        console.log(`[!] HIGH SCORE: Run ${runId} scored ${evalResult.composite_score}!`);
      }
    }
  }

  createStrategy() {
    switch (this.state.strategy) {
      case 'random':
        return new RandomStrategy(PARAM_SPACE, this.rng);
      case 'llm-guided':
        return new LLMGuidedStrategy(PARAM_SPACE, this.config.llm);
      case 'genetic':
        if (!this.geneticStrategy) {
          this.geneticStrategy = new GeneticStrategy(PARAM_SPACE, this.rng);
          const seeds = [...this.previousRuns]
            .sort((a, b) => b.score - a.score)
            .slice(0, 5);
          this.geneticStrategy.initialize(seeds);
        }
        return this.geneticStrategy;
      default:
        throw new Error(`Unknown strategy: ${this.state.strategy}`);
    }
  }

  recordBestResult(runId, evalResult, params, stats) {
    const bestPath = path.join(this.CHECKPOINT_DIR, 'best-results.json');
    let best = [];
    if (fs.existsSync(bestPath)) {
      best = JSON.parse(fs.readFileSync(bestPath, 'utf-8'));
    }
    best.push({
      runId,
      score: evalResult.composite_score,
      level: evalResult.emergence_level,
      description: evalResult.description,
      is_novel: evalResult.is_novel || false,
      params,
      stats,
      timestamp: new Date().toISOString(),
    });
    best.sort((a, b) => b.score - a.score);
    fs.writeFileSync(bestPath, JSON.stringify(best, null, 2));
  }

  loadOrCreateState() {
    const statePath = path.join(this.CHECKPOINT_DIR, 'exploration-state.json');
    if (fs.existsSync(statePath)) {
      return JSON.parse(fs.readFileSync(statePath, 'utf-8'));
    }
    const state = {
      experimentName: `exp-${Date.now()}`,
      startedAt: new Date().toISOString(),
      lastRunId: 0,
      strategy: 'random',
      bestScore: 0,
      bestRunId: 0,
      budget: { maxRuns: 100, maxLLMCalls: 100, runsUsed: 0, llmCallsUsed: 0 },
    };
    fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
    return state;
  }

  loadAllRuns() {
    const manifestPath = path.join(this.RUNS_DIR, 'manifest.jsonl');
    if (!fs.existsSync(manifestPath)) return [];
    return fs.readFileSync(manifestPath, 'utf-8')
      .trim().split('\n').filter(Boolean)
      .map(line => {
        const e = JSON.parse(line);
        return {
          runId: e.runId,
          params: e.params,
          score: e.composite_score,
          level: e.emergence_level,
          description: e.description,
        };
      });
  }

  saveState() {
    fs.writeFileSync(
      path.join(this.CHECKPOINT_DIR, 'exploration-state.json'),
      JSON.stringify(this.state, null, 2)
    );
  }

  printSummary() {
    console.log('\n========== EXPERIMENT SUMMARY ==========');
    console.log(`Session: ${this.sessionName}`);
    console.log(`Output: ${this.RUNS_DIR}`);
    console.log(`Total runs: ${this.state.budget.runsUsed}`);
    console.log(`Best score: ${this.state.bestScore} (Run ${this.state.bestRunId})`);

    const novelRuns = this.previousRuns.filter(r => r.is_novel);
    if (novelRuns.length > 0) {
      console.log(`\n★ Novel discoveries: ${novelRuns.length}`);
      for (const r of novelRuns) {
        console.log(`  Run ${r.runId}: ${r.score} - ${r.description}`);
      }
    }

    const bestRuns = [...this.previousRuns].sort((a, b) => b.score - a.score).slice(0, 5);
    console.log('\nTop 5 results:');
    for (const run of bestRuns) {
      console.log(`  Run ${run.runId}: ${run.score} - ${run.description}`);
    }
    console.log('========================================');
  }
}
