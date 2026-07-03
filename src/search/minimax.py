from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

from src.evaluator import HeuristicEvaluator
from src.moves import DIRECTIONS
from src.safety import legal_moves, safe_moves
from src.simulator import simulate_turn
from src.state import GameState


@dataclass(frozen=True)
class SearchResult:
    move: str
    score: float
    depth_completed: int
    timed_out: bool = False
    fallback_used: bool = False
    nodes: int = 0


class SearchTimeout(Exception):
    pass


def choose_move_minimax(
    state: GameState,
    our_snake_id: str,
    evaluator: HeuristicEvaluator,
    max_depth: int,
    deadline: float,
    fallback_move: str = "up",
    move_order: Callable[[GameState, str, list[str]], list[str]] | None = None,
) -> SearchResult:
    candidates = _candidate_moves(state, our_snake_id)
    if not candidates:
        return SearchResult(move=_valid_fallback(fallback_move), score=evaluator.evaluate(state, our_snake_id), depth_completed=0, fallback_used=True)

    order = move_order or _identity_order
    fallback = fallback_move if fallback_move in candidates else candidates[0]
    last_complete = SearchResult(move=fallback, score=-math.inf, depth_completed=0, fallback_used=True)
    nodes = 0
    timed_out = False

    for depth in range(1, max_depth + 1):
        if _timeout(deadline):
            timed_out = True
            break
        try:
            best_move = fallback
            best_score = -math.inf
            depth_nodes = 0
            for move in order(state, our_snake_id, candidates):
                _raise_if_timeout(deadline)
                score, visited = _score_root_move(state, our_snake_id, move, evaluator, depth, deadline, order)
                depth_nodes += visited
                if score > best_score:
                    best_score = score
                    best_move = move
            nodes += depth_nodes
            last_complete = SearchResult(move=best_move, score=best_score, depth_completed=depth, nodes=nodes)
        except SearchTimeout:
            timed_out = True
            break

    if timed_out:
        return SearchResult(
            move=last_complete.move,
            score=last_complete.score,
            depth_completed=last_complete.depth_completed,
            timed_out=True,
            fallback_used=last_complete.fallback_used,
            nodes=last_complete.nodes,
        )
    return last_complete


def minimax_search(
    state: GameState,
    our_id: str,
    opponents: list[str],
    candidate_moves: list[str],
    policy_order: Callable[[GameState, str, list[str]], list[str]],
    value_evaluator: Callable[[GameState, str], float],
    depth: int,
    deadline: float,
) -> str:
    class _AdapterEvaluator(HeuristicEvaluator):
        def evaluate(self, state: GameState, perspective_snake_id: str) -> float:
            return value_evaluator(state, perspective_snake_id)

    del opponents
    result = choose_move_minimax(
        state=state,
        our_snake_id=our_id,
        evaluator=_AdapterEvaluator(),
        max_depth=depth,
        deadline=deadline,
        fallback_move=candidate_moves[0] if candidate_moves else "up",
        move_order=policy_order,
    )
    return result.move


def _score_root_move(
    state: GameState,
    our_snake_id: str,
    our_move: str,
    evaluator: HeuristicEvaluator,
    depth: int,
    deadline: float,
    move_order: Callable[[GameState, str, list[str]], list[str]],
) -> tuple[float, int]:
    opponent_id = _single_opponent_id(state, our_snake_id)
    if opponent_id is None:
        return evaluator.evaluate(state, our_snake_id), 1
    opponent_moves = _candidate_moves(state, opponent_id)
    if not opponent_moves:
        return 1_000_000.0, 1

    worst = math.inf
    nodes = 0
    for opponent_move in move_order(state, opponent_id, opponent_moves):
        _raise_if_timeout(deadline)
        next_state = simulate_turn(state, {our_snake_id: our_move, opponent_id: opponent_move})
        value, visited = _max_value(next_state, our_snake_id, evaluator, depth - 1, deadline, move_order)
        nodes += visited + 1
        worst = min(worst, value)
    return worst, nodes


def _max_value(
    state: GameState,
    our_snake_id: str,
    evaluator: HeuristicEvaluator,
    depth: int,
    deadline: float,
    move_order: Callable[[GameState, str, list[str]], list[str]],
) -> tuple[float, int]:
    _raise_if_timeout(deadline)
    if _is_terminal(state, our_snake_id) or depth <= 0:
        return evaluator.evaluate(state, our_snake_id), 1

    our_moves = _candidate_moves(state, our_snake_id)
    if not our_moves:
        return -1_000_000.0, 1

    best = -math.inf
    nodes = 1
    for our_move in move_order(state, our_snake_id, our_moves):
        score, visited = _score_root_move(state, our_snake_id, our_move, evaluator, depth, deadline, move_order)
        nodes += visited
        best = max(best, score)
    return best, nodes


def _candidate_moves(state: GameState, snake_id: str) -> list[str]:
    try:
        return safe_moves(state, snake_id) or legal_moves(state, snake_id)
    except ValueError:
        return []


def _single_opponent_id(state: GameState, our_snake_id: str) -> str | None:
    opponents = [snake.id for snake in state.snakes if snake.alive and snake.id != our_snake_id]
    if not opponents:
        return None
    if len(opponents) > 1:
        return opponents[0]
    return opponents[0]


def _is_terminal(state: GameState, our_snake_id: str) -> bool:
    alive_ids = {snake.id for snake in state.snakes if snake.alive}
    return our_snake_id not in alive_ids or len(alive_ids - {our_snake_id}) == 0


def _identity_order(state: GameState, snake_id: str, moves: list[str]) -> list[str]:
    del state, snake_id
    return list(moves)


def _valid_fallback(move: str) -> str:
    return move if move in DIRECTIONS else "up"


def _raise_if_timeout(deadline: float) -> None:
    if _timeout(deadline):
        raise SearchTimeout


def _timeout(deadline: float) -> bool:
    return time.monotonic() >= deadline
