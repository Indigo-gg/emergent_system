# Hard Mode — GPU-Accelerated Emergent Evolution

A Taichi/CUDA-based evolutionary system that discovers emergent physics in 500K+ particle simulations. GEP (Gene Expression Programming) evolves potential energy formulas, symbolic differentiation derives forces, and a quality-diversity archive (MAP-Elites + Novelty) maintains a diverse population of interesting behaviors.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Evolution Loop                        │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐  │
│  │   GEP    │───>│Simulation│───>│  Feature Extract  │  │
│  │ Genome   │    │ (50K     │    │  (12D behavior    │  │
│  │ Decode   │    │  steps)  │    │   vector)         │  │
│  └──────────┘    └──────────┘    └───────────────────┘  │
│       │                                │                │
│       │         ┌──────────┐           │                │
│       │         │  Novelty │<──────────┘                │
│       │         │  Filter  │                            │
│       │         └────┬─────┘                            │
│       │              │                                  │
│       │    ┌─────────▼──────────┐                       │
│       │    │  MAP-Elites Grid   │                       │
│       │    │  (3375 cells)      │                       │
│       │    └─────────┬──────────┘                       │
│       │              │                                  │
│       │    ┌─────────▼──────────┐                       │
│       └────│  Parent Selection  │                       │
│            │  (70% grid /       │                       │
│            │   30% novelty)     │                       │
│            └─────────┬──────────┘                       │
│                      │                                  │
│            ┌─────────▼──────────┐                       │
│            │  Mutation (6 ops)  │                       │
│            └────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

### Simulation Pipeline (per step)

```
Build Spatial Hash  ──>  Compute Forces (VM)  ──>  Integrate (Velocity Verlet)
     (GPU)                  (GPU bytecode)           (damping + clamping)
```

Forces are derived from potential energy via **symbolic differentiation**:

```
U(r)  ──>  dU/dr  ──>  bytecode  ──>  F = -dU/dr
  (evolved)   (symbolic)    (VM)       (momentum-conserving)
```

## Key Components

### Simulation (`src/simulation/`)

| Module | What it does |
|--------|-------------|
| `particles.py` | SoA particle system (pos, vel, state[4], force, alive) on Taichi fields |
| `spatial_hash.py` | GPU-native spatial hash with atomic ops, bucket cap of 128 |
| `vm.py` | Stack-based bytecode VM — 15 opcodes, 16-deep stack, 128-instruction programs |
| `potential.py` | Expression tree AST, symbolic differentiation (chain rule), simplification, bytecode compiler |
| `integrator.py` | Velocity Verlet with viscous damping, force/speed/displacement clamping, periodic boundaries |
| `step.py` | Orchestrator: spatial hash → VM forces → integration |

### Evolution (`src/evolution/`)

| Module | What it does |
|--------|-------------|
| `genome.py` | GEP genome: head (functions + terminals) + tail (terminals only), encode/decode to expression tree |
| `gep.py` | Fitness evaluation, dead universe filtering, parsimony pressure, hybrid parent selection |
| `mutation.py` | 6 genetic operators: point mutation, constant fine-tune, IS/RIS transposition, 1/2-point recombination |
| `features.py` | 12D feature extraction: spatial entropy, island count, FFT amplitudes, angular momentum, density Laplacian, survival rate, autocorrelation |
| `map_elites.py` | 3D MAP-Elites grid (15³ = 3375 cells) + unbounded Novelty Archive with k-NN scoring |

### Supporting Systems

| Module | What it does |
|--------|-------------|
| `novelty/filter.py` | k-NN novelty scoring, adaptive threshold, dead universe filter |
| `rendering/renderer.py` | Density heatmaps, trajectory overlays (time-encoded colors), GIF generation |
| `vlm/judge.py` | Qwen2-VL-2B (local, 时分复用显存) — pattern naming, behavior description, novelty rating, natural analog suggestions |
| `storage/db.py` | SQLite (WAL mode) — grid_cells, novelty_archive, evolution_log |
| `storage/checkpoint.py` | Full state checkpoint: grid, archive, population, RNG, adaptive threshold |
| `monitoring/monitor.py` | GPU temp (nvidia-smi), disk space, staleness detection |

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- Taichi 1.7+

### Install

```bash
cd hard
pip install -r requirements.txt
```

### Run

```bash
# Full evolutionary loop (GEP + MAP-Elites + Novelty Archive)
python -m src.main --mode evolution

# Single simulation run (phase1 — basic physics test)
python -m src.main --mode phase1
```

### Test

```bash
# Run all 147 tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_vm.py -v
python -m pytest tests/test_potential.py -v
python -m pytest tests/test_spatial_hash.py -v
```

## Configuration

All parameters live in `config/default.yaml`:

```yaml
simulation:
  num_particles: 500000
  dt: 0.01
  steps_per_eval: 50000
  boundary: "periodic"

gep:
  head_length: 8
  bytecode_length: 128
  vm_stack_depth: 16
  terminal_set: ["dist", "density", "speed", "angle", ...]
  function_set: ["+", "-", "*", "sin", "cos", "tanh", "sqrt", "abs", "max", "min"]

map_elites:
  grid_features: ["entropy", "islands", "fft_freq"]
  resolution_per_dim: 15    # 15³ = 3375 cells

novelty:
  k_neighbors: 15
  threshold_adaptive: true
  grid_selection_prob: 0.7  # 70% MAP-Elites / 30% Novelty Archive

safety:
  max_force: 10.0
  max_speed: 5.0
  nan_penalty: 0.0
```

## VM Opcodes

| Opcode | Stack Effect | Description |
|--------|-------------|-------------|
| `CONST` | → value | Push constant |
| `VAR` | → value | Push terminal variable (dist, density, etc.) |
| `ADD` | a, b → a+b | Addition |
| `SUB` | a, b → a-b | Subtraction |
| `MUL` | a, b → a*b | Multiplication |
| `DIV` | a, b → a/b | Safe division (returns 0 on /0) |
| `SIN` | a → sin(a) | Sine |
| `COS` | a → cos(a) | Cosine |
| `TANH` | a → tanh(a) | Hyperbolic tangent |
| `SQRT` | a → sqrt(|a|) | Safe square root |
| `ABS` | a → |a| | Absolute value |
| `NEG` | a → -a | Negation |
| `MAX` | a, b → max(a,b) | Maximum |
| `MIN` | a, b → min(a,b) | Minimum |
| `CLAMP` | a → clamp(a) | Clamp to safety range |
| `HALT` | — | Stop execution |

## GEP Genetic Operators

| Operator | Rate | Description |
|----------|------|-------------|
| Point mutation | 30% | Random gene symbol replacement |
| Constant fine-tune | 20% | Gaussian perturbation of numeric constants |
| IS transposition | 15% | Copy random segment to gene head |
| RIS transposition | 10% | Copy from function to gene head start |
| One-point recombination | 15% | Swap gene tails between two parents |
| Two-point recombination | 10% | Swap gene segments between two parents |

## Feature Vector (12D)

The behavior descriptor used for MAP-Elites and novelty scoring:

| # | Feature | What it captures |
|---|---------|-----------------|
| 1 | Spatial entropy | How uniformly distributed particles are |
| 2 | Island count | Number of disconnected clusters |
| 3 | Speed variance | Heterogeneity of particle velocities |
| 4–6 | FFT amplitudes (top 3) | Dominant spatial frequencies |
| 7 | Angular momentum skewness | Rotation asymmetry |
| 8 | Density Laplacian variance | Smoothness of density field |
| 9 | Survival rate | Fraction of particles still alive |
| 10–12 | Autocorrelation (lag 1/2/3) | Temporal persistence of patterns |

## Directory Structure

```
hard/
├── config/
│   └── default.yaml          # All experiment parameters
├── scripts/
│   ├── vm_pressure_test.py   # 1M particles × 100 VM steps stress test
│   └── phase3_demo.py        # Phase 3 demonstration
├── src/
│   ├── main.py               # CLI entry point (phase1 / evolution)
│   ├── simulation/           # Taichi GPU simulation
│   ├── evolution/            # GEP + MAP-Elites + mutation
│   ├── novelty/              # Novelty scoring and filtering
│   ├── rendering/            # Visualization and GIF generation
│   ├── vlm/                  # VLM judge (Ollama)
│   ├── storage/              # SQLite DB + checkpoints
│   ├── monitoring/           # GPU/disk/staleness monitoring
│   └── visualization/        # Charts and reports
├── tests/                    # 17 test files, 147 tests
├── PHASE3_COMPLETION_REPORT.md
└── requirements.txt
```

## Design Decisions

**Why symbolic differentiation?** Evolving force rules directly violates Newton's 3rd law (F_ij ≠ -F_ji). By evolving potential energy U and computing F = -dU/dr symbolically, momentum conservation is guaranteed by construction — zero precision loss, no numerical artifacts.

**Why a stack-based VM?** GEP formulas change every generation. Compiling to bytecode means the GPU kernel stays identical; only the bytecode data array is swapped. No recompilation, no kernel launches, no host-device sync per formula change.

**Why MAP-Elites + Novelty?** Pure fitness search collapses to a single niche. MAP-Elites maintains a diverse archive across feature dimensions. The Novelty Archive rescues interesting outliers that don't fit the grid. The 70/30 selection split balances exploitation and exploration.
