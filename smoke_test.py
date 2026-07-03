from __future__ import annotations

import json
import os
import time

from logic import RUNTIME_MODEL, choose_move, compare_models, rank_moves

SAMPLE_STATE = {
    "game": {"id": "smoke", "ruleset": {"name": "standard", "settings": {}}, "timeout": 500},
    "turn": 37,
    "board": {
        "height": 11,
        "width": 11,
        "food": [{"x": 1, "y": 8}, {"x": 8, "y": 6}, {"x": 5, "y": 5}],
        "hazards": [],
        "snakes": [
            {
                "id": "you",
                "name": "you",
                "health": 42,
                "body": [
                    {"x": 3, "y": 5},
                    {"x": 3, "y": 4},
                    {"x": 3, "y": 3},
                    {"x": 2, "y": 3},
                    {"x": 1, "y": 3},
                ],
                "head": {"x": 3, "y": 5},
                "length": 5,
            },
            {
                "id": "small",
                "name": "small",
                "health": 70,
                "body": [
                    {"x": 6, "y": 5},
                    {"x": 6, "y": 4},
                    {"x": 7, "y": 4},
                ],
                "head": {"x": 6, "y": 5},
                "length": 3,
            },
            {
                "id": "big",
                "name": "big",
                "health": 80,
                "body": [
                    {"x": 8, "y": 8},
                    {"x": 8, "y": 9},
                    {"x": 7, "y": 9},
                    {"x": 6, "y": 9},
                    {"x": 5, "y": 9},
                ],
                "head": {"x": 8, "y": 8},
                "length": 5,
            },
        ],
    },
}
SAMPLE_STATE["you"] = SAMPLE_STATE["board"]["snakes"][0]


def _move_in_bounds_for_test(state, move):
    width = state["board"]["width"]
    height = state["board"]["height"]
    head = state["you"]["head"]
    delta = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}[move]
    nx, ny = head["x"] + delta[0], head["y"] + delta[1]
    return 0 <= nx < width and 0 <= ny < height


def _edge_state(x, y, width=5, height=5, blockers=None):
    blockers = blockers or []
    you_body = [{"x": x, "y": y}] + [{"x": bx, "y": by} for bx, by in blockers]
    state = {
        "game": {"id": "edge", "timeout": 500},
        "turn": 1,
        "board": {
            "height": height,
            "width": width,
            "food": [],
            "hazards": [],
            "snakes": [
                {
                    "id": "you",
                    "name": "you",
                    "health": 80,
                    "body": you_body,
                    "head": {"x": x, "y": y},
                    "length": len(you_body),
                }
            ],
        },
    }
    state["you"] = state["board"]["snakes"][0]
    return state


def run_oob_guard_tests():
    cases = [
        ("top_edge", _edge_state(2, 4)),
        ("bottom_edge", _edge_state(2, 0)),
        ("left_edge", _edge_state(0, 2)),
        ("right_edge", _edge_state(4, 2)),
        # Fully boxed except out-of-board direction would previously risk blind `up`.
        ("top_boxed", _edge_state(2, 4, blockers=[(2, 3), (1, 4), (3, 4)])),
        ("corner_boxed", _edge_state(0, 0, blockers=[(0, 1), (1, 0)])),
    ]
    for name, state in cases:
        move = choose_move(state)
        assert _move_in_bounds_for_test(state, move), f"{name}: {move} goes out of bounds"
    print("OOB guard tests: OK")


def main() -> None:
    print("ACTIVE_MODEL =", RUNTIME_MODEL)
    for model in ["expert", "ridge", "logistic", "svm", "perceptron", "mlp", "ensemble"]:
        ranked = rank_moves(SAMPLE_STATE, model)
        print("\n", model)
        for c in ranked:
            print(f"  {c.move:5s} final={c.final_score:9.2f} ml={c.model_score:7.3f} safe={c.safety_score:8.1f} {c.notes}")

    t0 = time.perf_counter()
    n = 1000
    moves = [choose_move(SAMPLE_STATE) for _ in range(n)]
    elapsed = (time.perf_counter() - t0) * 1000 / n
    print("\nChosen move:", moves[-1])
    print(f"Avg latency over {n} calls: {elapsed:.3f} ms")
    assert moves[-1] in {"up", "down", "left", "right"}
    assert elapsed < 20, "Local smoke latency is unexpectedly high"

    run_oob_guard_tests()

    print("\nCompact JSON compare:")
    print(json.dumps(compare_models(SAMPLE_STATE), ensure_ascii=False, indent=2)[:1500] + "...")


if __name__ == "__main__":
    main()
