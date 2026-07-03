import pytest

from src.state import GameState, SnakeState
from src.simulator import MissingMoveError, simulate_turn


def state(snakes, food=(), hazards=(), hazard_damage=0, width=7, height=7, turn=0):
    return GameState(
        width=width,
        height=height,
        turn=turn,
        our_snake_id=snakes[0].id if snakes else "our",
        snakes=tuple(snakes),
        food=tuple(food),
        hazards=tuple(hazards),
        hazard_damage=hazard_damage,
        max_health=100,
    )


def snake(snake_id, body, health=90):
    return SnakeState(id=snake_id, head=body[0], body=tuple(body), health=health, length=len(body))


def by_id(next_state, snake_id):
    return next(s for s in next_state.snakes if s.id == snake_id)


def test_health_decreases():
    next_state = simulate_turn(state([snake("our", [(2, 2), (2, 1)])]), {"our": "up"})
    assert by_id(next_state, "our").health == 89


def test_death_when_health_reaches_zero():
    next_state = simulate_turn(state([snake("our", [(2, 2)], health=1)]), {"our": "up"})
    assert next_state.snakes == ()


def test_hazard_damage():
    next_state = simulate_turn(
        state([snake("our", [(2, 2)], health=20)], hazards=((2, 3),), hazard_damage=10),
        {"our": "up"},
    )
    assert by_id(next_state, "our").health == 9


def test_hazard_causes_death():
    next_state = simulate_turn(
        state([snake("our", [(2, 2)], health=11)], hazards=((2, 3),), hazard_damage=10),
        {"our": "up"},
    )
    assert next_state.snakes == ()


def test_food_restores_health():
    next_state = simulate_turn(state([snake("our", [(2, 2)], health=1)], food=((2, 3),)), {"our": "up"})
    assert by_id(next_state, "our").health == 100


def test_food_causes_growth():
    next_state = simulate_turn(state([snake("our", [(2, 2), (2, 1)])], food=((2, 3),)), {"our": "up"})
    assert by_id(next_state, "our").body == ((2, 3), (2, 2), (2, 1))


def test_tail_moves_without_food():
    next_state = simulate_turn(state([snake("our", [(2, 2), (2, 1)])]), {"our": "up"})
    assert by_id(next_state, "our").body == ((2, 3), (2, 2))


def test_tail_stays_when_eating():
    next_state = simulate_turn(state([snake("our", [(2, 2), (2, 1)])], food=((2, 3),)), {"our": "up"})
    assert (2, 1) in by_id(next_state, "our").body


def test_can_enter_vacating_tail():
    next_state = simulate_turn(
        state([snake("our", [(1, 1), (1, 2), (2, 2), (2, 1)])]),
        {"our": "right"},
    )
    assert by_id(next_state, "our").head == (2, 1)


def test_cannot_enter_non_vacating_tail():
    next_state = simulate_turn(
        state([snake("our", [(1, 1), (1, 2), (2, 2), (2, 1)])], food=((2, 1),)),
        {"our": "right"},
    )
    assert next_state.snakes == ()


def test_wall_collision():
    next_state = simulate_turn(state([snake("our", [(2, 6)])]), {"our": "up"})
    assert next_state.snakes == ()


def test_self_collision():
    next_state = simulate_turn(
        state([snake("our", [(2, 2), (2, 1), (1, 1), (1, 2)])]),
        {"our": "down"},
    )
    assert next_state.snakes == ()


def test_body_collision():
    next_state = simulate_turn(
        state(
            [
                snake("our", [(2, 2), (1, 2)]),
                snake("enemy", [(5, 5), (3, 2), (5, 4)]),
            ]
        ),
        {"our": "right", "enemy": "left"},
    )
    assert all(s.id != "our" for s in next_state.snakes)


def test_head_to_head_longer_survives():
    next_state = simulate_turn(
        state([snake("our", [(2, 2), (2, 1), (2, 0)]), snake("enemy", [(4, 2), (4, 1)])]),
        {"our": "right", "enemy": "left"},
    )
    assert [s.id for s in next_state.snakes] == ["our"]


def test_head_to_head_shorter_dies():
    next_state = simulate_turn(
        state([snake("our", [(2, 2), (2, 1)]), snake("enemy", [(4, 2), (4, 1), (4, 0)])]),
        {"our": "right", "enemy": "left"},
    )
    assert [s.id for s in next_state.snakes] == ["enemy"]


def test_equal_length_head_to_head_kills_both():
    next_state = simulate_turn(
        state([snake("our", [(2, 2), (2, 1)]), snake("enemy", [(4, 2), (4, 1)])]),
        {"our": "right", "enemy": "left"},
    )
    assert next_state.snakes == ()


def test_three_snake_head_to_head():
    next_state = simulate_turn(
        state(
            [
                snake("a", [(2, 1), (2, 0), (1, 0)]),
                snake("b", [(2, 3), (2, 4), (1, 4)]),
                snake("c", [(1, 2), (0, 2)]),
            ],
            width=5,
            height=5,
        ),
        {"a": "up", "b": "down", "c": "right"},
    )
    assert next_state.snakes == ()


def test_missing_move_raises():
    with pytest.raises(MissingMoveError):
        simulate_turn(state([snake("our", [(2, 2)]), snake("enemy", [(4, 4)])]), {"our": "up"})


def test_food_removed_after_eating():
    next_state = simulate_turn(state([snake("our", [(2, 2)])], food=((2, 3), (0, 0))), {"our": "up"})
    assert next_state.food == ((0, 0),)


def test_turn_incremented():
    next_state = simulate_turn(state([snake("our", [(2, 2)])], turn=41), {"our": "up"})
    assert next_state.turn == 42
