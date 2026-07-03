"""Local Battlesnake matchup simulator for quick heuristic tuning.

This is not the official engine, but it models the core Standard rules closely
enough to compare move policies before deploying.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logic import DIRECTIONS, choose_move

Point = Tuple[int, int]
Policy = Callable[[Dict], str]

MOVE_ORDER = ("up", "right", "down", "left")


@dataclass
class Snake:
    snake_id: str
    body: List[Point]
    health: int = 100
    alive: bool = True

    @property
    def length(self) -> int:
        return len(self.body)

    @property
    def head(self) -> Point:
        return self.body[0]


@dataclass
class Game:
    width: int
    height: int
    snakes: Dict[str, Snake]
    food: Set[Point]
    turn: int = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--max-turns", type=int, default=160)
    parser.add_argument("--opponent", default="all")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    policies: Dict[str, Policy] = {
        "vlad_snake_like": aggressive_policy,
        "space": space_policy,
        "food": food_policy,
        "mirror": choose_move,
    }

    selected = policies.items()
    if args.opponent != "all":
        if args.opponent not in policies:
            raise SystemExit(f"unknown opponent {args.opponent!r}; choose one of: {', '.join(policies)}")
        selected = [(args.opponent, policies[args.opponent])]

    for name, opponent in selected:
        result = run_series(args.games, args.width, args.height, args.max_turns, opponent)
        print(
            f"{name:16} wins={result['wins']:4d} "
            f"losses={result['losses']:4d} draws={result['draws']:4d} "
            f"win_rate={result['wins'] / args.games:.3f} "
            f"avg_turns={result['turns'] / args.games:.1f}"
        )


def run_series(games: int, width: int, height: int, max_turns: int, opponent_policy: Policy) -> Dict[str, int]:
    totals = {"wins": 0, "losses": 0, "draws": 0, "turns": 0}
    for index in range(games):
        game = initial_game(width, height, index)
        winner = run_game(game, {"rtyuyrtutu": choose_move, "vlad_snake": opponent_policy}, max_turns)
        totals["turns"] += game.turn
        if winner == "rtyuyrtutu":
            totals["wins"] += 1
        elif winner == "vlad_snake":
            totals["losses"] += 1
        else:
            totals["draws"] += 1
    return totals


def initial_game(width: int, height: int, index: int) -> Game:
    starts = [
        ([(1, 1), (1, 0), (0, 0)], [(width - 2, height - 2), (width - 2, height - 1), (width - 1, height - 1)]),
        ([(1, height - 2), (1, height - 1), (0, height - 1)], [(width - 2, 1), (width - 2, 0), (width - 1, 0)]),
        ([(width // 2 - 2, height // 2), (width // 2 - 3, height // 2), (width // 2 - 4, height // 2)], [(width // 2 + 2, height // 2), (width // 2 + 3, height // 2), (width // 2 + 4, height // 2)]),
        ([(width // 2, 1), (width // 2, 0), (width // 2 - 1, 0)], [(width // 2, height - 2), (width // 2, height - 1), (width // 2 + 1, height - 1)]),
    ]
    mine, theirs = starts[index % len(starts)]
    food = {
        (width // 2, height // 2),
        (2, height - 3),
        (width - 3, 2),
    }
    return Game(
        width=width,
        height=height,
        snakes={
            "rtyuyrtutu": Snake("rtyuyrtutu", list(mine)),
            "vlad_snake": Snake("vlad_snake", list(theirs)),
        },
        food=food,
    )


def run_game(game: Game, policies: Dict[str, Policy], max_turns: int = 300) -> Optional[str]:
    while game.turn < max_turns:
        alive = [snake for snake in game.snakes.values() if snake.alive]
        if len(alive) <= 1:
            return alive[0].snake_id if alive else None

        states = {snake.snake_id: build_state(game, snake.snake_id) for snake in alive}
        moves = {}
        for snake in alive:
            move = policies[snake.snake_id](copy.deepcopy(states[snake.snake_id]))
            moves[snake.snake_id] = move if move in DIRECTIONS else "up"

        step(game, moves)
    return None


def step(game: Game, moves: Dict[str, str]) -> None:
    game.turn += 1
    old_food = set(game.food)
    new_heads: Dict[str, Point] = {}
    grew: Set[str] = set()

    for snake in game.snakes.values():
        if not snake.alive:
            continue
        dx, dy = DIRECTIONS[moves[snake.snake_id]]
        new_head = (snake.head[0] + dx, snake.head[1] + dy)
        new_heads[snake.snake_id] = new_head
        snake.health -= 1
        if new_head in old_food:
            grew.add(snake.snake_id)
            snake.health = 100
            game.food.discard(new_head)
        snake.body = [new_head] + snake.body
        if snake.snake_id not in grew:
            snake.body.pop()

    dead: Set[str] = set()
    for snake_id, head in new_heads.items():
        snake = game.snakes[snake_id]
        if not in_bounds(head, game.width, game.height) or snake.health <= 0:
            dead.add(snake_id)

    heads_by_cell: Dict[Point, List[Snake]] = {}
    for snake_id, head in new_heads.items():
        if snake_id not in dead:
            heads_by_cell.setdefault(head, []).append(game.snakes[snake_id])
    for snakes in heads_by_cell.values():
        if len(snakes) <= 1:
            continue
        longest = max(snake.length for snake in snakes)
        if sum(1 for snake in snakes if snake.length == longest) > 1:
            dead.update(snake.snake_id for snake in snakes)
        else:
            dead.update(snake.snake_id for snake in snakes if snake.length < longest)

    bodies = []
    for snake in game.snakes.values():
        if snake.alive:
            bodies.extend((snake.snake_id, part) for part in snake.body[1:])
    for snake_id, head in new_heads.items():
        if snake_id in dead:
            continue
        if any(head == part for _, part in bodies):
            dead.add(snake_id)

    for snake_id in dead:
        game.snakes[snake_id].alive = False

    ensure_food(game)


def build_state(game: Game, you_id: str) -> Dict:
    snakes = [snake_json(snake) for snake in game.snakes.values() if snake.alive]
    you = next(snake for snake in snakes if snake["id"] == you_id)
    return {
        "game": {"id": "local", "ruleset": {"settings": {"hazardDamagePerTurn": 15}}},
        "turn": game.turn,
        "board": {
            "height": game.height,
            "width": game.width,
            "food": [{"x": x, "y": y} for x, y in sorted(game.food)],
            "hazards": [],
            "snakes": snakes,
        },
        "you": you,
    }


def snake_json(snake: Snake) -> Dict:
    return {
        "id": snake.snake_id,
        "name": snake.snake_id,
        "health": snake.health,
        "body": [{"x": x, "y": y} for x, y in snake.body],
        "head": {"x": snake.head[0], "y": snake.head[1]},
        "length": snake.length,
    }


def aggressive_policy(state: Dict) -> str:
    """Approximate a direct 1v1 opponent: safe moves, food, then pressure."""
    board = state["board"]
    you = state["you"]
    head = point(you["head"])
    enemy = next(s for s in board["snakes"] if s["id"] != you["id"])
    enemy_head = point(enemy["head"])
    safe = legal_moves(state)
    if not safe:
        return "up"

    def score(move: str) -> float:
        dx, dy = DIRECTIONS[move]
        nxt = (head[0] + dx, head[1] + dy)
        value = flood_space(board, nxt) * 6.0
        foods = [point(food) for food in board["food"]]
        if foods:
            value -= min(manhattan(nxt, food) for food in foods) * (12.0 if you["health"] < 60 else 4.0)
        if you["length"] >= enemy["length"]:
            value -= manhattan(nxt, enemy_head) * 9.0
        else:
            value += manhattan(nxt, enemy_head) * 8.0
        return value

    return max(safe, key=score)


def space_policy(state: Dict) -> str:
    safe = legal_moves(state)
    if not safe:
        return "up"
    board = state["board"]
    head = point(state["you"]["head"])
    return max(safe, key=lambda move: flood_space(board, move_point(head, move)))


def food_policy(state: Dict) -> str:
    safe = legal_moves(state)
    if not safe:
        return "up"
    board = state["board"]
    head = point(state["you"]["head"])
    foods = [point(food) for food in board["food"]]
    if not foods:
        return space_policy(state)
    return min(safe, key=lambda move: min(manhattan(move_point(head, move), food) for food in foods))


def legal_moves(state: Dict) -> List[str]:
    board = state["board"]
    you = state["you"]
    head = point(you["head"])
    blocked = set()
    for snake in board["snakes"]:
        body = [point(part) for part in snake["body"]]
        blocked.update(body[:-1])
    moves = []
    for move in MOVE_ORDER:
        nxt = move_point(head, move)
        if in_bounds(nxt, board["width"], board["height"]) and nxt not in blocked:
            moves.append(move)
    return moves


def flood_space(board: Dict, start: Point) -> int:
    blocked = set()
    for snake in board["snakes"]:
        body = [point(part) for part in snake["body"]]
        blocked.update(body[:-1])
    if start in blocked or not in_bounds(start, board["width"], board["height"]):
        return 0
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for dx, dy in DIRECTIONS.values():
            nxt = (cell[0] + dx, cell[1] + dy)
            if nxt in seen or nxt in blocked or not in_bounds(nxt, board["width"], board["height"]):
                continue
            seen.add(nxt)
            stack.append(nxt)
    return len(seen)


def ensure_food(game: Game) -> None:
    occupied = {part for snake in game.snakes.values() if snake.alive for part in snake.body}
    while len(game.food) < 3:
        candidates = [
            (x, y)
            for x in range(game.width)
            for y in range(game.height)
            if (x, y) not in occupied and (x, y) not in game.food
        ]
        if not candidates:
            return
        game.food.add(random.choice(candidates))


def move_point(point_: Point, move: str) -> Point:
    dx, dy = DIRECTIONS[move]
    return (point_[0] + dx, point_[1] + dy)


def point(item: Dict) -> Point:
    return (item["x"], item["y"])


def in_bounds(point_: Point, width: int, height: int) -> bool:
    return 0 <= point_[0] < width and 0 <= point_[1] < height


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


if __name__ == "__main__":
    main()
