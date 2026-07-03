from src.strategy import choose_move


def test_choose_move_returns_valid_direction():
    game_state = {
        "board": {
            "width": 3,
            "height": 3,
            "snakes": [
                {"id": "our", "head": {"x": 1, "y": 1}, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 0}], "health": 90, "length": 2},
                {"id": "enemy", "head": {"x": 2, "y": 2}, "body": [{"x": 2, "y": 2}], "health": 90, "length": 3},
            ],
            "food": [],
            "hazards": [],
            "width": 3,
            "height": 3,
            "hazardDamage": 10,
        },
        "you": {"id": "our", "head": {"x": 1, "y": 1}, "body": [{"x": 1, "y": 1}, {"x": 1, "y": 0}], "health": 90, "length": 2},
        "turn": 0,
    }
    move = choose_move(game_state, fallback_move="left")
    assert move in {"up", "down", "left", "right"}
