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
    moves = legal_moves(state, "our")
    assert "down" not in moves
    assert set(moves) == {"up", "left", "right"}


def test_safe_moves_excludes_hazard_when_low_health():
    state = make_state()
    safe = safe_moves(state, "our")
    assert "up" not in safe
    assert "left" in safe


def test_safe_moves_allows_own_vacating_tail_move():
    state = GameState(
        width=3,
        height=3,
        turn=0,
        our_snake_id="our",
        snakes=(SnakeState(id="our", head=(1, 1), body=((1, 1), (1, 2)), health=90, length=2),),
        food=(),
        hazards=(),
        hazard_damage=0,
        max_health=100,
    )
    assert "up" in safe_moves(state, "our")


def test_legal_moves_are_snake_aware():
    state = GameState(
        width=5,
        height=5,
        turn=0,
        our_snake_id="our",
        snakes=(
            SnakeState(id="our", head=(0, 0), body=((0, 0), (1, 0), (1, 1)), health=90, length=3),
            SnakeState(id="enemy", head=(4, 4), body=((4, 4), (4, 3), (3, 3)), health=90, length=3),
        ),
        food=(),
        hazards=(),
        hazard_damage=0,
        max_health=100,
    )
    assert set(legal_moves(state, "our")) == {"up"}
    assert set(legal_moves(state, "enemy")) == {"left"}
