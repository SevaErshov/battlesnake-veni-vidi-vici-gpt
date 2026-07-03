# Battlesnake Hybrid Search

This branch keeps the original upstream ML baseline and adds a working local
game engine for testing a search-based candidate before training new models.

## Current Architecture

```text
Battlesnake JSON
  -> typed GameState parser
  -> hard safety filter
  -> heuristic minimax candidate for benchmark
  -> preserved original ML baseline fallback
```

The planned Policy-Value hybrid is still the target architecture, but no
CatBoost `.cbm` artifacts are included yet. Production `choose_move()` returns
the preserved baseline until both policy and value models are available.

## Important Files

- `backend.py` exposes `GET /`, `POST /start`, `POST /move`, and `POST /end`.
- `logic.py` is a thin adapter around `src.strategy` with baseline fallback.
- `src/baseline.py` preserves the original upstream ML baseline behavior.
- `src/state.py` parses Battlesnake JSON into typed immutable state objects.
- `src/safety.py` implements snake-aware `legal_moves(state, snake_id)` and
  `safe_moves(state, snake_id)`.
- `src/simulator.py` simulates one simultaneous turn, including food, health,
  hazards, tail release, body collisions, and head-to-head resolution.
- `src/evaluator.py` contains a configurable heuristic evaluator, not a trained
  model.
- `src/search/minimax.py` implements deadline-aware 1v1 simultaneous minimax.
- `src/game_runner.py` runs reproducible local 1v1 games and writes trajectories.
- `benchmark/run_duels.py` runs paired candidate-vs-baseline duels.

## Local Runner Rules

The local runner uses the shared `simulate_turn()` rules and deterministic
`random.Random(seed)` food spawning. Initial 11x11 1v1 positions are symmetric,
length 3, health 100, and can be swapped with `--swap-positions`.

Food spawning is simplified: after each turn the runner keeps up to three food
cells on random empty cells. This is intentionally not claimed to be an exact
copy of the official Battlesnake engine.

## Running

```bash
python -m compileall -q .
python -m pytest -q
python backend.py
```

The HTTP contract remains:

```text
GET  /
POST /start
POST /move -> {"move": "up|down|left|right"}
POST /end
```

## Benchmark

Smoke:

```bash
python -m benchmark.run_duels \
  --games 20 \
  --seed 10000 \
  --swap-positions \
  --max-turns 100 \
  --output benchmark/smoke_results.json
```

Full local run:

```bash
python -m benchmark.run_duels \
  --games 200 \
  --seed 20000 \
  --swap-positions \
  --max-turns 300 \
  --output benchmark/results.json
```

With `--swap-positions`, `--games` is the final number of games; seeds are paired
until that total is reached. Trajectories are written to the output stem as JSONL
unless `--trajectory-output` is provided.

Latest local results from this workspace:

| Run | Games | Wins | Losses | Draws | Win rate | Median latency | p95 latency | Max latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 20 | 9 | 0 | 11 | 0.45 | 25.91 ms | 50.88 ms | 65.81 ms |
| full | 200 | 132 | 54 | 14 | 0.66 | 25.98 ms | 51.64 ms | 252.09 ms |

Both runs recorded `fallback_count=0`, `invalid_moves=0`, and `timeouts=0`.

Generated benchmark outputs are ignored by Git:

- `benchmark/smoke_results.json`
- `benchmark/smoke_results.jsonl`
- `benchmark/results.json`
- `benchmark/results.jsonl`

## Trajectory Format

Each JSONL row contains the game id, seed, turn, Battlesnake-compatible state,
moves, per-bot latency, winner, and final candidate result. This is intended to
support future policy/value dataset generation and teacher-search replay.

## ML Status

No real `policy_model.cbm` or `value_model.cbm` is present. The model registry
loads CatBoost artifacts lazily if they appear, but local tests and production
fallback do not require CatBoost to be installed.

The next ML stage should build datasets from trajectories, train policy and
value models, validate them against the preserved baseline, then enable the
hybrid production path only when both artifacts are available.

## Limitations

- Multiplayer expectimax is intentionally simple; the main benchmark is 1v1.
- Heuristic evaluator weights are hand-written starting values, not optimized.
- Local food spawning and draw handling are simplified and deterministic.
- Training scripts remain scaffold-level and do not produce final ML artifacts.
