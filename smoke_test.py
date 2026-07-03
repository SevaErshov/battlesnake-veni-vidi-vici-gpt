"""Tiny local checks for logic.py. Run: python smoke_test.py"""

from logic import choose_move, rank_moves


def snake(sid, body, health=90):
    return {
        "id": sid,
        "name": sid,
        "health": health,
        "body": [{"x": x, "y": y} for x, y in body],
        "head": {"x": body[0][0], "y": body[0][1]},
        "length": len(body),
    }


def state_basic():
    you = snake("me", [(5, 5), (5, 4), (5, 3)], health=80)
    return {
        "game": {"id": "test"},
        "turn": 10,
        "board": {
            "height": 11,
            "width": 11,
            "food": [{"x": 2, "y": 2}, {"x": 8, "y": 8}],
            "hazards": [],
            "snakes": [you],
        },
        "you": you,
    }


def state_predator_corner():
    # We are longer. The small enemy is near the right wall; the best moves
    # should tend to pressure / reduce its exits while staying safe.
    you = snake("me", [(6, 5), (5, 5), (4, 5), (4, 4), (4, 3)], health=88)
    prey = snake("prey", [(8, 5), (9, 5), (9, 4)], health=60)
    return {
        "game": {"id": "kill"},
        "turn": 20,
        "board": {
            "height": 11,
            "width": 11,
            "food": [{"x": 8, "y": 6}],
            "hazards": [],
            "snakes": [you, prey],
        },
        "you": you,
    }


if __name__ == "__main__":
    for name, factory in [("basic", state_basic), ("predator", state_predator_corner)]:
        st = factory()
        print("\n==", name)
        print("chosen:", choose_move(st))
        for row in rank_moves(st):
            print(row)
