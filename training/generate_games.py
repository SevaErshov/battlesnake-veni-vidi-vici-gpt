from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from src.baseline import choose_move as baseline_choose_move
from src.state import parse_game_state
from src.simulator import simulate_turn


def run_games(output_path: Path, games: int, seed: int) -> None:
    random.seed(seed)
    with output_path.open("w", encoding="utf-8") as handle:
        for game_id in range(games):
            state = _initial_state(game_id)
            while True:
                move = baseline_choose_move(_to_raw_state(state))
                opponent_move = "up"
                state = simulate_turn(state, {state.our_snake_id: move, "enemy": opponent_move})
                handle.write(json.dumps({"game_id": game_id, "state": _to_raw_state(state)}) + "\n")
                if not any(snake.id == state.our_snake_id for snake in state.snakes):
                    break


def _initial_state(game_id: int) -> Any:
    raise NotImplementedError("Game runner not implemented")


def _to_raw_state(state: Any) -> dict:
    raise NotImplementedError("Raw state conversion not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/trajectories.jsonl"))
    args = parser.parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_games(output_path, args.games, args.seed)
