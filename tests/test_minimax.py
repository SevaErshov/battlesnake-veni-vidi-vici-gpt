import time

from src.evaluator import HeuristicEvaluator
from src.search.minimax import choose_move_minimax
from src.state import GameState, SnakeState


def snake(snake_id, body, health=90):
    return SnakeState(id=snake_id, head=body[0], body=tuple(body), health=health, length=len(body))


def state(snakes, width=7, height=7):
    return GameState(
        width=width,
        height=height,
        turn=0,
        our_snake_id="our",
        snakes=tuple(snakes),
        food=(),
        hazards=(),
        hazard_damage=0,
        max_health=100,
    )


def search(game_state, depth=2, evaluator=None):
    return choose_move_minimax(
        state=game_state,
        our_snake_id="our",
        evaluator=evaluator or HeuristicEvaluator(),
        max_depth=depth,
        deadline=time.monotonic() + 1.0,
        fallback_move="up",
    )


def test_one_our_move_leads_to_immediate_death():
    game_state = state([snake("our", [(1, 1), (1, 0)]), snake("enemy", [(3, 1), (3, 0), (4, 0)])])
    result = search(game_state)
    assert result.move != "right"


def test_one_our_move_can_guarantee_killing_opponent():
    game_state = state(
        [
            snake("our", [(1, 1), (1, 0), (0, 0), (0, 1), (0, 2)]),
            snake("enemy", [(3, 1), (3, 2), (4, 1), (3, 0)]),
        ],
        width=5,
        height=3,
    )
    assert search(game_state).move == "right"


def test_one_move_loses_head_to_head():
    game_state = state(
        [
            snake("our", [(1, 1), (1, 0)]),
            snake("enemy", [(3, 1), (3, 2), (4, 1), (3, 0)]),
        ],
        width=5,
        height=3,
    )
    assert search(game_state).move != "right"


def test_minimax_chooses_safe_move():
    game_state = state([snake("our", [(1, 1), (1, 0)]), snake("enemy", [(3, 1), (3, 0), (4, 0)])])
    assert search(game_state).move in {"up", "left"}


def test_depth_two_sees_trap_that_depth_one_misses():
    class TrapLovingEvaluator(HeuristicEvaluator):
        def evaluate(self, game_state, perspective_snake_id):
            alive = {s.id for s in game_state.snakes}
            if perspective_snake_id not in alive:
                return -1_000_000.0
            our = next(s for s in game_state.snakes if s.id == perspective_snake_id)
            return 1_000.0 if our.head == (2, 1) else 0.0

    game_state = state(
        [
            snake("our", [(1, 1), (1, 0), (0, 0)]),
            snake("enemy", [(4, 4), (2, 2), (2, 0), (3, 1), (4, 3)]),
        ],
        width=5,
        height=5,
    )
    assert search(game_state, depth=1, evaluator=TrapLovingEvaluator()).move == "right"
    assert search(game_state, depth=2, evaluator=TrapLovingEvaluator()).move != "right"


def test_deadline_returns_last_correct_result():
    game_state = state([snake("our", [(1, 1), (1, 0)]), snake("enemy", [(5, 5), (5, 4)])])
    result = choose_move_minimax(
        game_state,
        "our",
        HeuristicEvaluator(),
        max_depth=3,
        deadline=time.monotonic() - 0.01,
        fallback_move="up",
    )
    assert result.move in {"up", "down", "left", "right"}
    assert result.timed_out


def test_absence_of_safe_moves_does_not_raise():
    game_state = state([snake("our", [(1, 1)]), snake("enemy", [(3, 1), (3, 0), (4, 0)])])
    result = search(game_state)
    assert result.move in {"up", "down", "left", "right"}


def test_search_always_returns_valid_string():
    game_state = state([snake("our", [(0, 0)]), snake("enemy", [(6, 6)])])
    assert search(game_state).move in {"up", "down", "left", "right"}
