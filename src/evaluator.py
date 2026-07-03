from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.moves import DIRECTIONS, apply_move
from src.safety import safe_moves
from src.state import GameState, Point, SnakeState

WIN_VALUE = 1_000_000.0
LOSS_VALUE = -1_000_000.0
DRAW_VALUE = 0.0


@dataclass(frozen=True)
class EvaluatorWeights:
    territory: float = 8.0
    open_space: float = 12.0
    safe_moves: float = 40.0
    tail_reachable: float = 60.0
    health: float = 1.2
    length_advantage: float = 80.0
    food_urgency: float = 4.0
    head_to_head_risk: float = 120.0
    trap_risk: float = 160.0


class HeuristicEvaluator:
    def __init__(self, weights: EvaluatorWeights | None = None) -> None:
        self.weights = weights or EvaluatorWeights()

    def evaluate(self, state: GameState, perspective_snake_id: str) -> float:
        terminal = self._terminal_value(state, perspective_snake_id)
        if terminal is not None:
            return terminal

        snake = _snake_by_id(state, perspective_snake_id)
        opponents = [other for other in state.snakes if other.id != perspective_snake_id and other.alive]
        occupied = _occupied_without_vacating_tails(state) - {other.head for other in state.snakes if other.alive}
        open_space = _flood_count(state, snake.head, occupied, limit=state.width * state.height)
        territory = _territory_score(state, snake, opponents, occupied)
        safe_count = len(safe_moves(state, snake.id))
        tail_reachable = 1.0 if snake.body and snake.body[-1] in _reachable(state, snake.head, occupied - {snake.body[-1]}) else 0.0
        length_advantage = snake.length - max((other.length for other in opponents), default=0)
        food_score = _food_urgency_score(state, snake, occupied)
        h2h_risk = _head_to_head_risk(state, snake)
        trap_risk = 1.0 if open_space < snake.length + 2 else 0.0

        weights = self.weights
        return (
            weights.open_space * open_space
            + weights.territory * territory
            + weights.safe_moves * safe_count
            + weights.tail_reachable * tail_reachable
            + weights.length_advantage * length_advantage
            + weights.health * snake.health
            + weights.food_urgency * food_score
            - weights.head_to_head_risk * h2h_risk
            - weights.trap_risk * trap_risk
        )

    def _terminal_value(self, state: GameState, perspective_snake_id: str) -> float | None:
        alive_ids = {snake.id for snake in state.snakes if snake.alive}
        if perspective_snake_id not in alive_ids:
            return DRAW_VALUE if not alive_ids else LOSS_VALUE
        opponents_alive = alive_ids - {perspective_snake_id}
        if not opponents_alive:
            return WIN_VALUE
        return None


def _snake_by_id(state: GameState, snake_id: str) -> SnakeState:
    for snake in state.snakes:
        if snake.id == snake_id and snake.alive:
            return snake
    raise ValueError(f"Snake {snake_id} not found")


def _occupied_without_vacating_tails(state: GameState) -> set[Point]:
    occupied: set[Point] = set()
    for snake in state.snakes:
        if snake.alive:
            occupied.update(snake.body[:-1])
    return occupied


def _reachable(state: GameState, start: Point, blocked: set[Point]) -> set[Point]:
    if not _in_bounds(state, start) or start in blocked:
        return set()
    seen = {start}
    queue: deque[Point] = deque([start])
    while queue:
        point = queue.popleft()
        for delta in DIRECTIONS.values():
            nxt = (point[0] + delta[0], point[1] + delta[1])
            if nxt in seen or nxt in blocked or not _in_bounds(state, nxt):
                continue
            seen.add(nxt)
            queue.append(nxt)
    return seen


def _flood_count(state: GameState, start: Point, blocked: set[Point], limit: int) -> int:
    seen = _reachable(state, start, blocked)
    return min(len(seen), limit)


def _territory_score(
    state: GameState,
    snake: SnakeState,
    opponents: list[SnakeState],
    blocked: set[Point],
) -> float:
    our_dist = _distances(state, [snake.head], blocked)
    enemy_dist = _distances(state, [opponent.head for opponent in opponents], blocked) if opponents else {}
    territory = 0.0
    for cell, dist in our_dist.items():
        enemy = enemy_dist.get(cell)
        if enemy is None or dist < enemy:
            territory += 1.0
        elif dist == enemy:
            territory += 0.25
    return territory


def _distances(state: GameState, starts: list[Point], blocked: set[Point]) -> dict[Point, int]:
    distances: dict[Point, int] = {}
    queue: deque[Point] = deque()
    for start in starts:
        if _in_bounds(state, start) and start not in blocked and start not in distances:
            distances[start] = 0
            queue.append(start)
    while queue:
        point = queue.popleft()
        for delta in DIRECTIONS.values():
            nxt = (point[0] + delta[0], point[1] + delta[1])
            if nxt in distances or nxt in blocked or not _in_bounds(state, nxt):
                continue
            distances[nxt] = distances[point] + 1
            queue.append(nxt)
    return distances


def _food_urgency_score(state: GameState, snake: SnakeState, blocked: set[Point]) -> float:
    if not state.food:
        return 0.0
    if snake.health > 45:
        return 0.0
    distances = _distances(state, [snake.head], blocked)
    nearest = min((distances[food] for food in state.food if food in distances), default=state.width + state.height)
    return max(0.0, (state.width + state.height - nearest) * (50 - snake.health) / 50.0)


def _head_to_head_risk(state: GameState, snake: SnakeState) -> float:
    risk = 0.0
    my_safe = safe_moves(state, snake.id)
    for move in my_safe:
        point = apply_move(snake.head, move)
        for other in state.snakes:
            if other.id == snake.id or other.length < snake.length:
                continue
            if any(apply_move(other.head, other_move) == point for other_move in DIRECTIONS):
                risk += 1.0
    return risk


def _in_bounds(state: GameState, point: Point) -> bool:
    return 0 <= point[0] < state.width and 0 <= point[1] < state.height
