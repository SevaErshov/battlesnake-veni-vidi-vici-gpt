from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from src.moves import DIRECTIONS

Point = Tuple[int, int]


def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    try:
        move = choose_move_model(game_state)
    except Exception:
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width = int(board["width"])
    height = int(board["height"])

    head = (you["head"]["x"], you["head"]["y"])
    my_length = int(you["length"])
    health = int(you["health"])

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board.get("food", [])]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= 10_000

        if foods and health < 50:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    return best_move or "up"


def choose_move_model(game_state: Dict) -> Optional[str]:
    legal = _legal_moves(game_state)
    if not legal:
        return None

    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]

    best_move = None
    best_score = float("-inf")
    for move in legal:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width = int(board["width"])
    height = int(board["height"])
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    occupied = set()
    for snake in snakes:
        for seg in snake.get("body", []):
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    danger = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if int(snake.get("length", len(snake.get("body", [])))) < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    seen = {start}
    stack = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _in_bounds(point: Point, width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


_MODEL = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}
