from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.moves import DIRECTIONS
from src.safety import legal_moves, safe_moves
from src.simulator import simulate_turn
from src.state import GameState, Point, SnakeState


@dataclass
class BotSpec:
    name: str
    choose_move: Callable[[dict], str]
    fallback_move: Callable[[dict], str] | None = None
    timeout_ms: float | None = None


@dataclass
class GameResult:
    game_id: str
    seed: int
    winner_id: str | None
    result: str
    turns: int
    death_reasons: dict[str, str]
    trajectory: list[dict]
    latencies_ms: dict[str, list[float]]
    fallback_counts: dict[str, int]
    invalid_moves: dict[str, int] = field(default_factory=dict)
    timeouts: dict[str, int] = field(default_factory=dict)


def create_initial_state(
    seed: int,
    snake_a_id: str = "candidate",
    snake_b_id: str = "baseline",
    width: int = 11,
    height: int = 11,
    swap_positions: bool = False,
    food_count: int = 3,
) -> GameState:
    rng = random.Random(seed)
    y = height // 2
    left_body = ((3, y), (2, y), (1, y))
    right_body = ((width - 4, y), (width - 3, y), (width - 2, y))
    if swap_positions:
        body_a, body_b = right_body, left_body
    else:
        body_a, body_b = left_body, right_body
    snakes = (
        SnakeState(id=snake_a_id, head=body_a[0], body=body_a, health=100, length=len(body_a)),
        SnakeState(id=snake_b_id, head=body_b[0], body=body_b, health=100, length=len(body_b)),
    )
    state = GameState(
        width=width,
        height=height,
        turn=0,
        our_snake_id=snake_a_id,
        snakes=snakes,
        food=(),
        hazards=(),
        hazard_damage=0,
        max_health=100,
    )
    return _spawn_food(state, rng, food_count)


def run_game(
    bot_a: BotSpec,
    bot_b: BotSpec,
    initial_state: GameState,
    seed: int,
    max_turns: int = 300,
) -> GameResult:
    rng = random.Random(seed)
    state = _with_perspective(initial_state, bot_a.name)
    bots = {bot_a.name: bot_a, bot_b.name: bot_b}
    game_id = f"game_{seed}_{bot_a.name}_vs_{bot_b.name}"
    latencies = {bot_a.name: [], bot_b.name: []}
    fallback_counts = {bot_a.name: 0, bot_b.name: 0}
    invalid_moves = {bot_a.name: 0, bot_b.name: 0}
    timeouts = {bot_a.name: 0, bot_b.name: 0}
    death_reasons: dict[str, str] = {}
    trajectory: list[dict] = []

    while len(state.snakes) > 1 and state.turn < max_turns:
        turn_moves: dict[str, str] = {}
        turn_latencies: dict[str, float] = {}
        raw_states: dict[str, dict] = {}
        for snake in state.snakes:
            bot = bots[snake.id]
            raw_state = to_raw_state(state, snake.id, game_id)
            raw_states[snake.id] = raw_state
            started = time.monotonic()
            try:
                move = bot.choose_move(raw_state)
            except Exception:
                move = None
            elapsed_ms = (time.monotonic() - started) * 1000.0
            latencies[snake.id].append(elapsed_ms)
            turn_latencies[snake.id] = elapsed_ms
            if bot.timeout_ms is not None and elapsed_ms > bot.timeout_ms:
                timeouts[snake.id] += 1
            if move not in DIRECTIONS:
                invalid_moves[snake.id] += 1
                fallback_counts[snake.id] += 1
                move = _fallback_move(bot, raw_state, state, snake.id)
            turn_moves[snake.id] = move

        trajectory.append(
            {
                "game_id": game_id,
                "seed": seed,
                "turn": state.turn,
                "state": to_raw_state(state, bot_a.name, game_id),
                "moves": dict(turn_moves),
                "latencies_ms": turn_latencies,
                "winner": None,
                "final_result_candidate": None,
            }
        )
        before_ids = {snake.id for snake in state.snakes}
        state = simulate_turn(state, turn_moves)
        state = _spawn_food(state, rng, target_food=3)
        after_ids = {snake.id for snake in state.snakes}
        for dead_id in before_ids - after_ids:
            death_reasons.setdefault(dead_id, "eliminated_by_local_simulator")

    alive_ids = [snake.id for snake in state.snakes]
    winner_id = alive_ids[0] if len(alive_ids) == 1 else None
    if winner_id == bot_a.name:
        result = "bot_a_win"
        final_result_candidate = 1
    elif winner_id == bot_b.name:
        result = "bot_b_win"
        final_result_candidate = -1
    else:
        result = "draw"
        final_result_candidate = 0
        for snake in state.snakes:
            death_reasons.setdefault(snake.id, "max_turns_draw")

    for item in trajectory:
        item["winner"] = winner_id
        item["final_result_candidate"] = final_result_candidate

    return GameResult(
        game_id=game_id,
        seed=seed,
        winner_id=winner_id,
        result=result,
        turns=state.turn,
        death_reasons=death_reasons,
        trajectory=trajectory,
        latencies_ms=latencies,
        fallback_counts=fallback_counts,
        invalid_moves=invalid_moves,
        timeouts=timeouts,
    )


def to_raw_state(state: GameState, perspective_snake_id: str, game_id: str = "local-game") -> dict:
    you = _snake_by_id(state, perspective_snake_id)
    snakes = [_snake_to_raw(snake) for snake in state.snakes]
    return {
        "game": {"id": game_id, "ruleset": {"name": "local-simplified"}, "timeout": 500},
        "turn": state.turn,
        "board": {
            "height": state.height,
            "width": state.width,
            "food": [_point_to_raw(point) for point in state.food],
            "hazards": [_point_to_raw(point) for point in state.hazards],
            "snakes": snakes,
            "hazardDamage": state.hazard_damage,
            "health": state.max_health,
        },
        "you": _snake_to_raw(you),
    }


def write_trajectory_jsonl(result: GameResult, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for item in result.trajectory:
            handle.write(json.dumps(item, sort_keys=True) + "\n")


def _fallback_move(bot: BotSpec, raw_state: dict, state: GameState, snake_id: str) -> str:
    if bot.fallback_move is not None:
        try:
            move = bot.fallback_move(raw_state)
            if move in DIRECTIONS:
                return move
        except Exception:
            pass
    moves = safe_moves(state, snake_id) or legal_moves(state, snake_id)
    return moves[0] if moves else "up"


def _spawn_food(state: GameState, rng: random.Random, target_food: int) -> GameState:
    food = set(state.food)
    occupied = {segment for snake in state.snakes for segment in snake.body}
    available = [
        (x, y)
        for x in range(state.width)
        for y in range(state.height)
        if (x, y) not in occupied and (x, y) not in food
    ]
    while len(food) < target_food and available:
        index = rng.randrange(len(available))
        food.add(available.pop(index))
    return GameState(
        width=state.width,
        height=state.height,
        turn=state.turn,
        our_snake_id=state.our_snake_id,
        snakes=state.snakes,
        food=tuple(sorted(food)),
        hazards=state.hazards,
        hazard_damage=state.hazard_damage,
        max_health=state.max_health,
    )


def _with_perspective(state: GameState, snake_id: str) -> GameState:
    return GameState(
        width=state.width,
        height=state.height,
        turn=state.turn,
        our_snake_id=snake_id,
        snakes=state.snakes,
        food=state.food,
        hazards=state.hazards,
        hazard_damage=state.hazard_damage,
        max_health=state.max_health,
    )


def _snake_by_id(state: GameState, snake_id: str) -> SnakeState:
    for snake in state.snakes:
        if snake.id == snake_id:
            return snake
    raise ValueError(f"Snake {snake_id} not found")


def _snake_to_raw(snake: SnakeState) -> dict:
    return {
        "id": snake.id,
        "name": snake.id,
        "health": snake.health,
        "body": [_point_to_raw(point) for point in snake.body],
        "head": _point_to_raw(snake.head),
        "length": snake.length,
    }


def _point_to_raw(point: Point) -> dict[str, int]:
    return {"x": point[0], "y": point[1]}
