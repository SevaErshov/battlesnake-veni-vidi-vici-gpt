"""Thin HTTP adapter for the Battlesnake service."""

from __future__ import annotations

from typing import Dict

from src.baseline import choose_move as baseline_choose_move, get_info
from src.moves import DIRECTIONS
from src.strategy import choose_move as hybrid_choose_move


def choose_move(game_state: Dict) -> str:
    fallback_move = _valid_move(baseline_choose_move(game_state))
    try:
        return _valid_move(hybrid_choose_move(game_state=game_state, fallback_move=fallback_move), fallback_move)
    except Exception:
        return fallback_move


def _valid_move(move: str, fallback: str = "up") -> str:
    if move in DIRECTIONS:
        return move
    if fallback in DIRECTIONS:
        return fallback
    return "up"
