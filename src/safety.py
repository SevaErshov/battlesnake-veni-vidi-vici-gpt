from __future__ import annotations

from typing import List

from src.moves import DIRECTIONS, apply_move
from src.state import GameState, SnakeState


def legal_moves(state: GameState, snake_id: str) -> List[str]:
    snake = _snake_by_id(state, snake_id)
    head = snake.head
    occupied = {segment for snake in state.snakes for segment in snake.body}
    legal: List[str] = []
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if 0 <= nxt[0] < state.width and 0 <= nxt[1] < state.height and nxt not in occupied:
            legal.append(move)
    return legal


def safe_moves(state: GameState, snake_id: str) -> List[str]:
    snake = _snake_by_id(state, snake_id)
    safe: List[str] = []
    occupied = {segment for other in state.snakes for segment in other.body}
    own_tail = snake.body[-1]
    for move in DIRECTIONS:
        nxt = apply_move(snake.head, move)
        ate = nxt in state.food
        if not _in_bounds(nxt, state.width, state.height):
            continue
        health_after = snake.health - 1
        if nxt in state.hazards:
            health_after -= state.hazard_damage
        if health_after <= 0 and not ate:
            continue
        if nxt in occupied:
            if nxt == own_tail and not ate:
                safe.append(move)
            continue
        if _risks_head_to_head(state, snake, nxt):
            continue
        safe.append(move)
    return safe


def _snake_by_id(state: GameState, snake_id: str) -> SnakeState:
    for snake in state.snakes:
        if snake.id == snake_id and snake.alive:
            return snake
    raise ValueError(f"Snake {snake_id} not found")


def _risks_head_to_head(state: GameState, snake: SnakeState, point: tuple[int, int]) -> bool:
    for other in state.snakes:
        if other.id == snake.id or not other.alive:
            continue
        if other.length < snake.length:
            continue
        for move in DIRECTIONS:
            if apply_move(other.head, move) == point:
                return True
    return False


def _in_bounds(point: tuple[int, int], width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height
