from __future__ import annotations

import math
from typing import Callable, List

from src.simulator import simulate_turn
from src.state import GameState
from src.safety import legal_moves, safe_moves


def minimax_search(
    state: GameState,
    our_id: str,
    opponents: List[str],
    candidate_moves: List[str],
    policy_order: Callable[[GameState, str, List[str]], List[str]],
    value_evaluator: Callable[[GameState, str], float],
    depth: int,
    deadline: float,
) -> str:
    best_move = candidate_moves[0]
    best_score = -math.inf
    for move in policy_order(state, our_id, candidate_moves):
        score = _min_value(state, our_id, opponents, move, policy_order, value_evaluator, depth, deadline)
        if score > best_score:
            best_score = score
            best_move = move
        if _timeout(deadline):
            break
    return best_move


def _max_value(
    state: GameState,
    our_id: str,
    opponents: List[str],
    policy_order: Callable[[GameState, str, List[str]], List[str]],
    value_evaluator: Callable[[GameState, str], float],
    depth: int,
    deadline: float,
) -> float:
    if _timeout(deadline) or depth <= 0:
        return value_evaluator(state, our_id)
    our_moves = safe_moves(state) or legal_moves(state)
    if not our_moves:
        return -1_000_000.0
    best = -math.inf
    for move in policy_order(state, our_id, our_moves):
        score = _min_value(state, our_id, opponents, move, policy_order, value_evaluator, depth - 1, deadline)
        best = max(best, score)
        if _timeout(deadline):
            break
    return best


def _min_value(
    state: GameState,
    our_id: str,
    opponents: List[str],
    our_move: str,
    policy_order: Callable[[GameState, str, List[str]], List[str]],
    value_evaluator: Callable[[GameState, str], float],
    depth: int,
    deadline: float,
) -> float:
    if _timeout(deadline) or depth <= 0:
        return value_evaluator(state, our_id)
    if not opponents:
        return value_evaluator(state, our_id)
    opponent_id = opponents[0]
    opp_moves = safe_moves(state) or legal_moves(state)
    if not opp_moves:
        return -1_000_000.0
    worst = math.inf
    for opp_move in policy_order(state, opponent_id, opp_moves):
        next_state = simulate_turn(state, {our_id: our_move, opponent_id: opp_move})
        if _terminal(next_state, our_id):
            score = value_evaluator(next_state, our_id)
        else:
            remaining = opponents[1:]
            if remaining:
                score = _min_value(next_state, our_id, remaining, our_move, policy_order, value_evaluator, depth, deadline)
            else:
                score = _max_value(next_state, our_id, opponents, policy_order, value_evaluator, depth - 1, deadline)
        worst = min(worst, score)
        if _timeout(deadline):
            break
    return worst


def _terminal(state: GameState, our_id: str) -> bool:
    return all(snake.id != our_id for snake in state.snakes)


def _timeout(deadline: float) -> bool:
    import time

    return time.monotonic() >= deadline
