from src.baseline import choose_move as baseline_choose_move
from src.game_runner import BotSpec, create_initial_state, run_game
from src.state import GameState, SnakeState
from src.strategy import choose_move_search_heuristic


def fixed(move):
    return lambda raw: move


def tiny_state():
    return GameState(
        width=5,
        height=3,
        turn=0,
        our_snake_id="candidate",
        snakes=(
            SnakeState("candidate", (1, 1), ((1, 1), (1, 0), (0, 0)), 90, 3),
            SnakeState("baseline", (3, 1), ((3, 1), (3, 0)), 90, 2),
        ),
        food=(),
        hazards=(),
        hazard_damage=0,
        max_health=100,
    )


def compact_trajectory(result):
    return [(item["turn"], item["moves"], item["state"]["board"]["food"]) for item in result.trajectory]


def test_same_seed_is_reproducible():
    bot_a = BotSpec("candidate", fixed("up"))
    bot_b = BotSpec("baseline", fixed("down"))
    initial_a = create_initial_state(42)
    initial_b = create_initial_state(42)
    result_a = run_game(bot_a, bot_b, initial_a, seed=42, max_turns=5)
    result_b = run_game(bot_a, bot_b, initial_b, seed=42, max_turns=5)
    assert (result_a.result, result_a.turns, compact_trajectory(result_a)) == (
        result_b.result,
        result_b.turns,
        compact_trajectory(result_b),
    )


def test_different_seed_changes_initial_state():
    assert create_initial_state(1).food != create_initial_state(2).food


def test_swap_positions():
    normal = create_initial_state(1)
    swapped = create_initial_state(1, swap_positions=True)
    assert normal.snakes[0].body == swapped.snakes[1].body
    assert normal.snakes[1].body == swapped.snakes[0].body


def test_runner_finishes_terminal_game():
    result = run_game(BotSpec("candidate", fixed("right")), BotSpec("baseline", fixed("left")), tiny_state(), seed=1)
    assert result.winner_id == "candidate"
    assert result.result == "bot_a_win"


def test_runner_respects_max_turns():
    result = run_game(
        BotSpec("candidate", fixed("up")),
        BotSpec("baseline", fixed("up")),
        create_initial_state(5),
        seed=5,
        max_turns=1,
    )
    assert result.turns == 1


def test_runner_records_trajectory():
    result = run_game(BotSpec("candidate", fixed("right")), BotSpec("baseline", fixed("left")), tiny_state(), seed=1)
    assert result.trajectory
    assert result.trajectory[0]["moves"] == {"candidate": "right", "baseline": "left"}


def test_runner_records_latency():
    result = run_game(BotSpec("candidate", fixed("right")), BotSpec("baseline", fixed("left")), tiny_state(), seed=1)
    assert result.latencies_ms["candidate"]
    assert result.latencies_ms["candidate"][0] >= 0


def test_runner_rejects_invalid_move():
    result = run_game(
        BotSpec("candidate", fixed("diagonal")),
        BotSpec("baseline", fixed("left")),
        tiny_state(),
        seed=1,
        max_turns=1,
    )
    assert result.invalid_moves["candidate"] == 1
    assert result.fallback_counts["candidate"] == 1


def test_runner_builds_correct_you_perspective():
    seen = []

    def record(raw):
        seen.append(raw["you"]["id"])
        return "up"

    run_game(BotSpec("candidate", record), BotSpec("baseline", record), create_initial_state(6), seed=6, max_turns=1)
    assert seen == ["candidate", "baseline"]


def test_full_candidate_vs_baseline_smoke_game():
    result = run_game(
        BotSpec("candidate", choose_move_search_heuristic, fallback_move=baseline_choose_move),
        BotSpec("baseline", baseline_choose_move, fallback_move=baseline_choose_move),
        create_initial_state(99),
        seed=99,
        max_turns=30,
    )
    assert result.result in {"bot_a_win", "bot_b_win", "draw"}
    assert result.trajectory
