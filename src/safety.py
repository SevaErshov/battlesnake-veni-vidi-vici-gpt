from __future__ import annotations

from typing import Dict, List

from src.moves import DIRECTIONS, apply_move
from src.state import GameState, Point, SnakeState


def legal_moves(state: GameState) -> List[str]:
    head = state.our_snake.head
    occupied = {segment for snake in state.snakes for segment in snake.body}
    legal: List[str] = []
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if 0 <= nxt[0] < state.width and 0 <= nxt[1] < state.height and nxt not in occupied:
            legal.append(move)
    return legal


def safe_moves(state: GameState) -> List[str]:
    moves = legal_moves(state)
    if not moves:
        return []
    safe: List[str] = []
    body_cells = {segment for snake in state.snakes for segment in snake.body}
    tail = state.our_snake.body[-1]
    for move in moves:
        nxt = apply_move(state.our_snake.head, move)
        if nxt in state.hazards and state.our_snake.health <= state.hazard_damage:
            continue
        if nxt == tail:
            safe.append(move)
            continue
        if any(nxt == other.head and other.length >= state.our_snake.length for other in state.snakes if other.id != state.our_snake_id):
            continue
        safe.append(move)
    return safe
