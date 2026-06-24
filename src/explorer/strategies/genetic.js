/**
 * Genetic Algorithm exploration strategy (Phase C).
 *
 * Uses mutation + tournament selection to explore parameter space.
 * Naturally adapts to non-linear phase boundaries.
 * Includes elite decay (0.95x) to prevent accidental high scores
 * from permanently dominating the gene pool.
 */
import { PARAM_SPACE, DEFAULTS } from '../param-space.js';

export class GeneticStrategy {
  constructor(paramSpace = PARAM_SPACE, rng = Math.random, config = {}) {
    this.paramSpace = paramSpace;
    this.rng = rng;
    this.populationSize = config.populationSize || 10;
    this.mutationRate = config.mutationRate || 0.15;
    this.mutationScale = config.mutationScale || 0.1;
    this.eliteCount = config.eliteCount || 3;
    this.population = [];
    this.generation = 0;
  }

  /**
   * Initialize population from high-score seed runs.
   * @param {Array} seedRuns - [{params, score}, ...]
   */
  initialize(seedRuns = []) {
    this.population = [];

    for (const run of seedRuns.slice(0, this.eliteCount)) {
      this.population.push({ params: { ...run.params }, score: 0, _prevScore: run.score });
    }

    while (this.population.length < this.populationSize) {
      this.population.push({ params: this.randomParams(), score: 0 });
    }

    this.generation = 0;
  }

  /**
   * Evolve to the next generation based on evaluated scores.
   */
  evolve(evaluatedPopulation) {
    evaluatedPopulation.sort((a, b) => b.score - a.score);

    const nextGen = [];

    // Elite carry-over with 0.95x score decay
    for (let i = 0; i < this.eliteCount && i < evaluatedPopulation.length; i++) {
      nextGen.push({
        params: { ...evaluatedPopulation[i].params },
        score: 0,
        _prevScore: evaluatedPopulation[i].score * 0.95,
      });
    }

    // Fill rest with tournament-selected + mutated offspring
    while (nextGen.length < this.populationSize) {
      const parent = this.tournamentSelect(evaluatedPopulation);
      const child = this.mutate({ ...parent.params });
      nextGen.push({ params: child, score: 0 });
    }

    this.population = nextGen;
    this.generation++;
    return this.population;
  }

  /** Get next unevaluated individual's params. Returns null if generation is done. */
  next() {
    const individual = this.population.find(p => p.score === 0);
    return individual ? individual.params : null;
  }

  /** Tournament selection: pick 3 random, return the best. */
  tournamentSelect(pop) {
    let best = null;
    for (let i = 0; i < 3; i++) {
      const idx = Math.floor(this.rng() * pop.length);
      if (!best || pop[idx].score > best.score) {
        best = pop[idx];
      }
    }
    return best;
  }

  /** Mutate each parameter with probability mutationRate. */
  mutate(params) {
    for (const [key, spec] of Object.entries(this.paramSpace)) {
      if (spec.category === 'render') continue;
      if (this.rng() < this.mutationRate) {
        const range = spec.max - spec.min;
        const delta = (this.rng() - 0.5) * 2 * range * this.mutationScale;
        params[key] = Math.max(spec.min, Math.min(spec.max, params[key] + delta));
        params[key] = Math.round(params[key] / spec.step) * spec.step;
      }
    }
    return params;
  }

  randomParams() {
    const params = {};
    for (const [key, spec] of Object.entries(this.paramSpace)) {
      if (spec.category === 'render') {
        params[key] = DEFAULTS[key] ?? (spec.min + spec.max) / 2;
        continue;
      }
      const steps = Math.floor((spec.max - spec.min) / spec.step);
      params[key] = spec.min + Math.floor(this.rng() * (steps + 1)) * spec.step;
    }
    return params;
  }
}
