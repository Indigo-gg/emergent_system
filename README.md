# Emergent System

A research platform for exploring emergent behavior in multi-agent particle systems. Two complementary implementations share the same design philosophy — **minimal rules, maximal emergence** — but target different scales:

| Mode | Stack | Scale | Purpose |
|------|-------|-------|---------|
| **Browser** | JavaScript + p5.js | 1K–5K agents | Real-time interactive visualization, LLM-guided parameter exploration |
| **Hard** | Python + Taichi (CUDA) | 500K+ particles | GPU-accelerated evolutionary search with GEP, MAP-Elites, and VLM evaluation |

## Design Philosophy

> **Weak control, strong rules.**

The system defines a minimal physics core (time stepping, spatial hashing, boundary conditions, field diffusion) and lets plugins or evolved formulas produce complex macroscopic behavior from simple microscopic interactions. Agents never directly modify positions — they propose "desire forces," and the engine resolves them under physical constraints. This **intent-resolution pattern** guarantees momentum conservation and makes emergent behavior attributable to the rules, not hand-tuned hacks.

## Browser Mode

A real-time p5.js simulation with three visualization modes:

- **Slime Mold** — 3,000 agents following chemotaxis rules on a 2D chemical field
- **Ecosystem** — 3 species (Explorer, Grazer, Harvester) with pheromone trails, nutrient hotspots, waste dynamics, and terrain
- **Explorer** — Headless Puppeteer automation that searches parameter space in 3 phases:
  - **Phase A** — Latin Hypercube Sampling (broad coverage)
  - **Phase B** — LLM-guided search (MiMo v2.5 evaluates composite screenshots across 6 dimensions)
  - **Phase C** — Genetic algorithm (refine the best candidates)

### Quick Start

```bash
npm install
npm run dev          # opens Vite dev server with all visualization pages
npm run explore      # headless LLM-guided exploration (requires .env with API key)
```

### Project Structure

```
src/
  engine.js            # Core intent-resolution engine (fixed timestep)
  state.js             # SoA particle buffer
  field.js             # 2D chemical field (diffusion + decay)
  spatial-hash.js      # O(1) neighbor queries
  plugin.js            # Plugin base class and registry
  prng.js              # Seedable PRNG
  slime-mold/          # Single-species chemotaxis simulation
  ecosystem/           # Multi-species ecosystem (3 species, terrain, nutrients)
  explorer/            # Automated parameter search orchestration
```

## Hard Mode

A GPU-accelerated evolutionary system for discovering emergent physics at scale. See [`hard/README.md`](hard/README.md) for full details.

**Key innovations:**
- **Symbolic differentiation** — Evolves potential energy fields U, derives forces as F = -dU/dr, guaranteeing Newton's 3rd law by construction
- **Stack-based VM** — GEP formulas compile to bytecode; formula changes swap data, no recompilation
- **MAP-Elites + Novelty Archive** — Quality-diversity search across 12-dimensional feature space
- **VLM evaluation** — Local Qwen2-VL-2B (时分复用显存) names patterns, rates novelty, suggests natural analogs

```bash
cd hard
pip install -r requirements.txt
python -m src.main --mode evolution   # full evolutionary loop
python -m src.main --mode phase1      # single simulation run
```

## Documentation

| File | Description |
|------|-------------|
| [`plan.md`](plan.md) | Design philosophy manifesto (Chinese) |
| [`docs/emergence-scheme.md`](docs/emergence-scheme.md) | Ecosystem explorer scheme — species, parameters, scoring |
| [`docs/hard-mode.md`](docs/hard-mode.md) | Hard Mode architecture document (v5) |
| [`hard/PHASE3_COMPLETION_REPORT.md`](hard/PHASE3_COMPLETION_REPORT.md) | Phase 3 completion report |

## Tech Stack

**Browser:** JavaScript, p5.js, Puppeteer, Vite, Node.js

**Hard Mode:** Python, Taichi (CUDA), NumPy, Matplotlib, SQLite, PyYAML, Ollama

## License

Private research project.
