import pytest

from src.state import GameState, SnakeState
from src.safety import legal_moves, safe_moves


def make_state() -> GameState:
    return GameState(
        width=3,
        height=3,
        turn=0,
        our_snake_id="our",
        snakes=(
            SnakeState(id="our", head=(1, 1), body=((1, 1), (1, 0)), health=5, length=2),
            SnakeState(id="enemy", head=(2, 2), body=((2, 2),), health=90, length=3),
        ),
        food=(),
        hazards=((1, 2),),
        hazard_damage=10,
        max_health=100,
    )


def test_legal_moves_excludes_walls_and_bodies():
    state = make_state()
    moves = legal_moves(state)
    assert "down" not in moves
    assert set(moves) == {"up", "left", "right"}


def test_safe_moves_excludes_hazard_when_low_health():
    state = make_state()
    safe = safe_moves(state)
    assert "up" not in safe
    assert "left" in safe


def test_safe_moves_allows_tail_move():
    original = make_state()
    state = GameState(
        width=original.width,
        height=original.height,
        turn=original.turn,
        our_snake_id=original.our_snake_id,
        snakes=(
            SnakeState(id="our", head=(1, 1), body=((1, 1), (1, 2)), health=90, length=2),
            original.snakes[1],
        ),
        food=original.food,
        hazards=original.hazards,
        hazard_damage=original.hazard_damage,
        max_health=original.max_health,
    )
    safe = safe_moves(state)
    assert "down" in safe
