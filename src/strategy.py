from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

from src.baseline import choose_move as baseline_choose_move
from src.evaluator import HeuristicEvaluator
from src.features import FeatureBuilder
from src.models.registry import ModelRegistry
from src.moves import DIRECTIONS
from src.search.deadline import Deadline
from src.search.expectimax import expectimax_search
from src.search.minimax import choose_move_minimax, minimax_search
from src.safety import legal_moves, safe_moves
from src.state import GameState, parse_game_state

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_REGISTRY = ModelRegistry(ARTIFACTS_DIR)

SEARCH_BUDGET_MS = int(os.environ.get("SEARCH_BUDGET_MS", "250"))
SEARCH_MAX_DEPTH = int(os.environ.get("SEARCH_MAX_DEPTH", "3"))
SEARCH_BEAM_WIDTH = int(os.environ.get("SEARCH_BEAM_WIDTH", "4"))
HEURISTIC_SEARCH_DEPTH = int(os.environ.get("HEURISTIC_SEARCH_DEPTH", "2"))


def choose_move(game_state: Dict, fallback_move: str) -> str:
    """Production move selection preserving baseline fallback without ML artifacts."""
    fallback_move = _valid_move(fallback_move)
    if not game_state:
        return fallback_move
    try:
        state = parse_game_state(game_state)
        if not MODEL_REGISTRY.has_models():
            return fallback_move

        legal = legal_moves(state, state.our_snake_id)
        if not legal:
            return fallback_move
        safe = safe_moves(state, state.our_snake_id)
        candidates = safe or legal
        if len(candidates) == 1:
            return candidates[0]

        policy_scores = _rank_candidates(state, candidates)
        best_move = _search(state, candidates, policy_scores)
        return _valid_move(best_move, fallback_move)
    except Exception:
        logger.exception("Hybrid strategy failed, using baseline")
        return fallback_move


def choose_move_search_heuristic(game_state: Dict) -> str:
    """Benchmark candidate: hard safety + 1v1 minimax + heuristic evaluator."""
    fallback_move = _valid_move(baseline_choose_move(game_state))
    try:
        state = parse_game_state(game_state)
        legal = legal_moves(state, state.our_snake_id)
        if not legal:
            return fallback_move
        candidates = safe_moves(state, state.our_snake_id) or legal
        if len(candidates) == 1:
            return candidates[0]
        result = choose_move_minimax(
            state=state,
            our_snake_id=state.our_snake_id,
            evaluator=HeuristicEvaluator(),
            max_depth=HEURISTIC_SEARCH_DEPTH,
            deadline=Deadline(SEARCH_BUDGET_MS).deadline,
            fallback_move=fallback_move if fallback_move in candidates else candidates[0],
            move_order=_heuristic_order_moves,
        )
        return _valid_move(result.move, fallback_move)
    except Exception:
        logger.exception("Heuristic search strategy failed, using baseline")
        return fallback_move


def _rank_candidates(state: GameState, candidates: List[str]) -> List[tuple[str, float]]:
    builder = FeatureBuilder(state)
    features = [builder.policy_features(state.our_snake_id, move) for move in candidates]
    if not MODEL_REGISTRY.has_policy_model:
        return [(move, 0.0) for move in candidates]
    scores = MODEL_REGISTRY.policy.rank(features)
    return list(zip(candidates, scores))


def _order_moves(state: GameState, snake_id: str, moves: List[str]) -> List[str]:
    builder = FeatureBuilder(state)
    features = [builder.policy_features(snake_id, move) for move in moves]
    if not MODEL_REGISTRY.has_policy_model:
        return list(moves)
    scores = MODEL_REGISTRY.policy.rank(features)
    return [move for _, move in sorted(zip(scores, moves), reverse=True)]


def _heuristic_order_moves(state: GameState, snake_id: str, moves: List[str]) -> List[str]:
    evaluator = HeuristicEvaluator()
    scored: list[tuple[float, str]] = []
    for move in moves:
        score = FeatureBuilder(state).policy_features(snake_id, move).get("open_space_after_move", 0.0)
        try:
            score += 0.001 * evaluator.evaluate(state, snake_id)
        except Exception:
            pass
        scored.append((score, move))
    return [move for _, move in sorted(scored, reverse=True)]


def _evaluate(state: GameState, snake_id: str) -> float:
    builder = FeatureBuilder(state)
    features = [builder.value_features(snake_id)]
    if not MODEL_REGISTRY.has_value_model:
        return 0.0
    return MODEL_REGISTRY.value.predict(features)[0]


def _search(state: GameState, candidates: List[str], policy_scores: List[tuple[str, float]]) -> str:
    del policy_scores
    opponents = [snake.id for snake in state.snakes if snake.id != state.our_snake_id and snake.alive]
    deadline = Deadline(SEARCH_BUDGET_MS).deadline
    if len(opponents) <= 1:
        return minimax_search(
            state=state,
            our_id=state.our_snake_id,
            opponents=opponents,
            candidate_moves=candidates,
            policy_order=_order_moves,
            value_evaluator=_evaluate,
            depth=SEARCH_MAX_DEPTH,
            deadline=deadline,
        )
    return expectimax_search(
        state=state,
        our_id=state.our_snake_id,
        opponents=opponents,
        policy_order=_order_moves,
        value_evaluator=_evaluate,
        depth=SEARCH_MAX_DEPTH,
        beam_width=SEARCH_BEAM_WIDTH,
        deadline=deadline,
    )


def _valid_move(move: str, fallback: str = "up") -> str:
    if move in DIRECTIONS:
        return move
    if fallback in DIRECTIONS:
        return fallback
    return "up"
