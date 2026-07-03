from src.state import GameState, Point, SnakeState
from src.simulator import simulate_turn


def make_state() -> GameState:
    return GameState(
        width=3,
        height=3,
        turn=0,
        our_snake_id="our",
        snakes=(
            SnakeState(id="our", head=(1, 1), body=((1, 1), (1, 0)), health=90, length=2),
            SnakeState(id="enemy", head=(0, 0), body=((0, 0), (0, 1)), health=90, length=2),
        ),
        food=((2, 2),),
        hazards=(),
        hazard_damage=0,
        max_health=100,
    )


def test_simulator_tail_releases_without_eating():
    state = make_state()
    next_state = simulate_turn(state, {"our": "up", "enemy": "right"})
    our = next(s for s in next_state.snakes if s.id == "our")
    assert our.body == ((1, 2), (1, 1))
    assert our.length == 2


def test_simulator_grows_after_eating():
    state = make_state()
    state = GameState(
        **{**state.__dict__, "food": ((1, 2),)}
    )
    next_state = simulate_turn(state, {"our": "up", "enemy": "right"})
    our = next(s for s in next_state.snakes if s.id == "our")
    assert our.body == ((1, 2), (1, 1), (1, 0))
    assert our.length == 3


def test_simulator_health_resets_on_food():
    state = make_state()
    state = GameState(
        **{**state.__dict__, "food": ((1, 2),)}
    )
    state = GameState(
        **{**state.__dict__, "snakes": (SnakeState(id="our", head=(1, 1), body=((1, 1), (1, 0)), health=5, length=2), state.snakes[1])}
    )
    next_state = simulate_turn(state, {"our": "up", "enemy": "right"})
    our = next(s for s in next_state.snakes if s.id == "our")
    assert our.health == 100


def test_simulator_head_to_head_resolution():
    state = GameState(
        width=3,
        height=3,
        turn=0,
        our_snake_id="our",
        snakes=(
            SnakeState(id="our", head=(1, 1), body=((1, 1),), health=90, length=1),
            SnakeState(id="enemy", head=(1, 3), body=((1, 3),), health=90, length=2),
        ),
        food=(),
        hazards=(),
        hazard_damage=0,
        max_health=100,
    )
    next_state = simulate_turn(state, {"our": "up", "enemy": "down"})
    assert all(snake.id != "our" for snake in next_state.snakes)
