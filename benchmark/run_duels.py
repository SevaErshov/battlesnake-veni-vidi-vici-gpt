from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from src.baseline import choose_move as baseline_choose_move
from src.state import parse_game_state
from src.strategy import choose_move as hybrid_choose_move


def run_duels(output: Path, games: int, seed: int, swap_positions: bool) -> None:
    random.seed(seed)
    results = []
    for game_id in range(games):
        state = _initial_state(game_id)
        steps = 0
        start = time.monotonic()
        while True:
            move_a = hybrid_choose_move(_to_raw_state(state), fallback_move="up")
            move_b = baseline_choose_move(_to_raw_state(state))
            state = _step(state, move_a, move_b)
            steps += 1
            if steps > 100:
                break
        duration = int((time.monotonic() - start) * 1000)
        results.append({"game_id": game_id, "duration_ms": duration, "steps": steps})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))


def _initial_state(game_id: int) -> Any:
    raise NotImplementedError("Initial state setup not implemented")


def _to_raw_state(state: Any) -> dict:
    raise NotImplementedError("Raw state conversion not implemented")


def _step(state: Any, move_a: str, move_b: str) -> Any:
    raise NotImplementedError("Step simulation not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--swap-positions", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results.json"))
    args = parser.parse_args()
    run_duels(args.output, args.games, args.seed, args.swap_positions)
