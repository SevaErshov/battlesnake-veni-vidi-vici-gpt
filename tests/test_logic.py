import unittest

from logic import choose_move, get_info


def snake(snake_id, body, health=90):
    return {
        "id": snake_id,
        "name": snake_id,
        "health": health,
        "body": [{"x": x, "y": y} for x, y in body],
        "head": {"x": body[0][0], "y": body[0][1]},
        "length": len(body),
    }


def state(you_body, enemies=None, food=None, health=90, width=7, height=7):
    you = snake("you", you_body, health=health)
    enemy_snakes = [snake(f"enemy-{idx}", body) for idx, body in enumerate(enemies or [])]
    return {
        "game": {
            "id": "test-game",
            "ruleset": {"settings": {"hazardDamagePerTurn": 15}},
        },
        "turn": 0,
        "board": {
            "height": height,
            "width": width,
            "food": [{"x": x, "y": y} for x, y in (food or [])],
            "hazards": [],
            "snakes": [you] + enemy_snakes,
        },
        "you": you,
    }


class LogicTests(unittest.TestCase):
    def test_info_contract(self):
        info = get_info()

        self.assertEqual(info["apiversion"], "1")
        self.assertIn("color", info)
        self.assertIn("head", info)
        self.assertIn("tail", info)

    def test_avoids_wall_and_body(self):
        game_state = state([(0, 0), (0, 1), (1, 1)], width=4, height=4)

        self.assertEqual(choose_move(game_state), "right")

    def test_chases_adjacent_food_when_starving(self):
        game_state = state(
            [(3, 3), (3, 2), (3, 1)],
            food=[(4, 3), (0, 0)],
            health=12,
            width=7,
            height=7,
        )

        self.assertEqual(choose_move(game_state), "right")

    def test_avoids_equal_length_head_to_head(self):
        game_state = state(
            [(1, 1), (1, 0), (0, 0)],
            enemies=[[(3, 1), (3, 0), (4, 0)]],
            food=[(6, 6)],
            width=7,
            height=7,
        )

        self.assertNotEqual(choose_move(game_state), "right")

    def test_avoids_contested_food_that_would_tie_after_growth(self):
        game_state = state(
            [(1, 1), (1, 0), (0, 0)],
            enemies=[[(3, 1), (3, 0), (4, 0)]],
            food=[(2, 1)],
            width=7,
            height=7,
        )

        self.assertNotEqual(choose_move(game_state), "right")

    def test_allows_following_own_tail_when_it_vacates(self):
        game_state = state(
            [(1, 1), (1, 2), (0, 2), (0, 1), (0, 0), (1, 0)],
            enemies=[[(2, 1), (2, 0), (2, 0)]],
            width=3,
            height=3,
        )

        self.assertEqual(choose_move(game_state), "down")


if __name__ == "__main__":
    unittest.main()
