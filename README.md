# Battlesnake Hybrid ML Search

Hybrid Battlesnake solution combining safety filters, simulation, search,
and ML-guided policy/value ranking.

## Architecture

flowchart TD
    A[Game State] --> B[Hard Safety]
    B --> C[Policy Model]
    C --> D[Simulator and Search]
    D --> E[Value Model]
    E --> F[Best Move]
    F --> G[Response]
    D -. error or deadline .-> H[Baseline Fallback]

## What Changed

This branch preserves the original baseline as a fallback and introduces:

- Typed `GameState`, `SnakeState`, and `Point` models.
- Hard safety rules for legal and safe move filtering.
- Deterministic turn simulator for simultaneous snake actions.
- Policy and value feature builders with shared schema.
- Model registry loading CatBoost artifacts once at startup.
- Minimax/expectimax search guided by ML rankings and safety.
- Benchmark and training pipeline scaffolding.

## Project Structure

- `backend.py` — Flask server exposing Battlesnake endpoints.
- `logic.py` — thin adapter preserving public HTTP contract.
- `src/` — core game engine: state, safety, simulator, features, models, search, strategy.
- `training/` — pipeline scripts for game generation and model training.
- `benchmark/` — duel runner and smoke test utilities.
- `artifacts/` — feature schema and model metadata.
- `tests/` — pytest unit tests.

## Local Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backend.py
```

Open `http://localhost:8000/` to verify the snake metadata response.

## Tests

```bash
/usr/local/bin/python3 -m pytest -q
```

## Training and Data Generation

The scaffolded pipeline is present, but full dataset generation and model
training are not yet completed.

```bash
python -m training.generate_games --games 100 --seed 42 --output data/trajectories.jsonl
python -m training.build_datasets --output data/dataset.csv
python -m training.train_policy --output artifacts/policy_model.cbm
python -m training.train_value --output artifacts/value_model.cbm
```

## Benchmark

```bash
python -m benchmark.run_duels --games 200 --seed 10000 --swap-positions --output benchmark/results.json
```

## Environment Variables

- `SEARCH_BUDGET_MS` — internal search budget in milliseconds (default: `250`).
- `SEARCH_MAX_DEPTH` — search depth for minimax/expectimax (default: `3`).
- `SEARCH_BEAM_WIDTH` — beam width for multiplicative opponent pruning (default: `4`).

## Fallback

If ML inference, search, or simulation fails, the system logs the failure and
returns the preserved baseline move.

## Render Deployment

Render uses `render.yaml`:

- `buildCommand`: `pip install -r requirements.txt`
- `startCommand`: `gunicorn backend:app --bind 0.0.0.0:$PORT`

## Current Limitations

- Training pipeline is scaffolded, not fully implemented.
- Benchmark runner has placeholder hooks for state initialization.
- No model artifacts are included yet.
- Search is limited to shallow depths to avoid timeouts.
