from __future__ import annotations

from typing import Callable, Dict, List

from src.state import GameState
from src.simulator import simulate_turn
from src.safety import safe_moves


def expectimax_search(
    state: GameState,
    our_id: str,
    opponents: list[str],
    policy_order: Callable[[GameState, str, list[str]], list[str]],
    value_evaluator: Callable[[GameState, str], float],
    depth: int,
    beam_width: int,
    deadline: float,
) -> str:
    candidate_moves = policy_order(state, our_id, safe_moves(state) or [])[:beam_width]
    best_move = candidate_moves[0] if candidate_moves else "up"
    best_value = -float("inf")
    for move in candidate_moves:
        value = _expect_value(state, our_id, opponents, move, policy_order, value_evaluator, depth - 1, beam_width, deadline)
        if value > best_value:
            best_value = value
            best_move = move
        if _timeout(deadline):
            break
    return best_move


def _expect_value(
    state: GameState,
    our_id: str,
    opponents: list[str],
    move: str,
    policy_order: Callable[[GameState, str, list[str]], list[str]],
    value_evaluator: Callable[[GameState, str], float],
    depth: int,
    beam_width: int,
    deadline: float,
) -> float:
    if _timeout(deadline) or depth <= 0:
        return value_evaluator(state, our_id)
    opponent_id = opponents[0] if opponents else None
    if opponent_id is None:
        return value_evaluator(state, our_id)
    moves = safe_moves(state)
    if not moves:
        return -1_000_000.0
    ordered = policy_order(state, opponent_id, moves)[:beam_width]
    values = []
    for opp_move in ordered:
        next_state = simulate_turn(state, {our_id: move, opponent_id: opp_move})
        values.append(value_evaluator(next_state, our_id))
        if _timeout(deadline):
            break
    if not values:
        return -1_000_000.0
    return sum(values) / len(values)


def _timeout(deadline: float) -> bool:
    import time

    return time.monotonic() >= deadline
