"""Model-backed move-selection logic for the Battlesnake.

The served policy uses a linear ranking model, scores each legal move,
and returns the highest-scoring direction. A compact heuristic remains as a
fallback so gameplay still returns a legal move if model scoring fails.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

import logging
import time
from collections import deque
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]
SnakeState = Tuple[str, Tuple[Point, ...], int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
VALID_MOVES: Tuple[str, ...] = tuple(DIRECTIONS.keys())
DEFAULT_MOVE = VALID_MOVES[0]

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50

MAX_SEARCH_DEPTH = 3
SEARCH_MAX_BUDGET_SECONDS = 0.21
SEARCH_TIMEOUT_RESERVE_SECONDS = 0.28
SEARCH_TIMEOUT_FRACTION = 0.42
STANDARD_FOOD_HEALTH = 100
WIN_SCORE = 1_000_000.0
LOSS_SCORE = -1_000_000.0

log = logging.getLogger("battlesnake.logic")
LAST_SEARCH_STATS: Dict[str, object] = {}


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move using a baseline-first tactical search."""
    started_at = time.perf_counter()
    baseline = _choose_move_baseline(game_state)
    _record_search_stats(
        {
            "completed_depth": 0,
            "expanded_states": 0,
            "fallback_used": True,
            "reason": "baseline",
            "move": baseline,
        }
    )

    try:
        if not _supports_search_rules(game_state):
            _record_search_stats(
                {
                    "completed_depth": 0,
                    "expanded_states": 0,
                    "fallback_used": True,
                    "reason": "unsupported_ruleset",
                    "move": baseline,
                }
            )
            return _finalize_move(game_state, baseline)

        deadline = _search_deadline(game_state, started_at)
        move, stats = _search_best_move(game_state, baseline, deadline)
        move = _finalize_move(game_state, move, baseline)
        stats["move"] = move
        stats["fallback_used"] = stats.get("completed_depth", 0) == 0
        _record_search_stats(stats)
        return move
    except _SearchTimeout:
        _record_search_stats(
            {
                "completed_depth": 0,
                "expanded_states": 0,
                "fallback_used": True,
                "reason": "timeout",
                "move": baseline,
            }
        )
        return _finalize_move(game_state, baseline)
    except Exception:  # noqa: BLE001 - search must never break the API response
        log.debug("Battlesnake search failed; using baseline move", exc_info=True)
        _record_search_stats(
            {
                "completed_depth": 0,
                "expanded_states": 0,
                "fallback_used": True,
                "reason": "search_exception",
                "move": baseline,
            }
        )
        return _finalize_move(game_state, baseline)


def _choose_move_baseline(game_state: Dict) -> str:
    """Return the existing model/heuristic choice, always as a valid direction."""
    for chooser in (choose_move_model, choose_move_heuristic):
        try:
            move = chooser(game_state)
        except Exception:  # noqa: BLE001 - baseline must stay fail-safe
            continue
        if move in DIRECTIONS:
            return _finalize_move(game_state, move)
    return _finalize_move(game_state, DEFAULT_MOVE)


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """All cells currently filled by any snake's body.

    We keep tails occupied too; they only free up *next* turn and treating them
    as solid is the conservative, safe choice for a base bot.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that are >= our length.

    Moving onto one of these risks a head-to-head collision we would lose or
    tie, so they are heavily penalized (but not forbidden — sometimes it's the
    only move).
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Used to avoid moves that would seal us into a small pocket.
    """
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
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


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Embedded model features -------------------------------------------------

_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def _bfs_dist(sources, blocked, width, height):
    """Shortest free-cell distances from seed cells."""
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    # Tail reachability is a useful anti-self-trap signal.
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD

    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1)),
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
    }


# --- Model -----------------------------------------------------
# Embedded standardized linear model.

_MODEL: Dict = {
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


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the trained model; return the best.

    Returns ``None`` (so the caller falls back to the heuristic) if the model
    isn't available or the snake is trapped with no legal move.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None

    best_move, best_score = None, float("-inf")
    for move in legal:
        score = _model_score_for_move(game_state, move)
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def _model_score_for_move(game_state: Dict, move: str) -> float:
    """Return the embedded linear model score for one candidate move."""
    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    score = float(_MODEL["intercept"])
    feats = _candidate_features(game_state, move)
    for i, name in enumerate(names):
        z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
        score += coef[i] * z
    return score


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]


# --- Deadline-aware tactical search -----------------------------------------


class _SearchTimeout(Exception):
    """Raised when the request-local search budget is exhausted."""


def _record_search_stats(stats: Dict[str, object]) -> None:
    LAST_SEARCH_STATS.clear()
    LAST_SEARCH_STATS.update(stats)


def _valid_direction(move: Optional[str], fallback: str = DEFAULT_MOVE) -> str:
    if move in DIRECTIONS:
        return move
    if fallback in DIRECTIONS:
        return fallback
    return DEFAULT_MOVE


def _in_bounds_moves(game_state: Dict) -> List[str]:
    """Directions that keep our next head inside the board, ignoring bodies."""
    try:
        board = game_state["board"]
        width, height = board["width"], board["height"]
        head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    except Exception:  # noqa: BLE001 - malformed states fall back to DEFAULT_MOVE
        return []
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
    ]


def _finalize_move(game_state: Dict, move: Optional[str], fallback: str = DEFAULT_MOVE) -> str:
    """Return a valid direction, avoiding out-of-bounds moves whenever possible."""
    in_bounds = _in_bounds_moves(game_state)
    if move in in_bounds:
        return move
    if fallback in in_bounds:
        return fallback
    legal = []
    try:
        legal = _legal_moves(game_state)
    except Exception:  # noqa: BLE001 - final safety must not raise
        legal = []
    if legal:
        return legal[0]
    if in_bounds:
        return in_bounds[0]
    return _valid_direction(move, fallback)


def _supports_search_rules(game_state: Dict) -> bool:
    """Return whether the lightweight simulator matches this request's rules."""
    try:
        board = game_state["board"]
        if board.get("hazards"):
            return False
        ruleset = (game_state.get("game") or {}).get("ruleset") or {}
        name = str(ruleset.get("name", "standard")).lower()
        return name in {"", "standard", "solo"}
    except Exception:
        return False


def _search_deadline(game_state: Dict, started_at: float) -> float:
    timeout_ms = ((game_state.get("game") or {}).get("timeout") or 500)
    try:
        timeout_seconds = max(float(timeout_ms) / 1000.0, 0.02)
    except (TypeError, ValueError):
        timeout_seconds = 0.5
    budget = min(
        SEARCH_MAX_BUDGET_SECONDS,
        max(0.015, timeout_seconds * SEARCH_TIMEOUT_FRACTION),
        max(0.015, timeout_seconds - SEARCH_TIMEOUT_RESERVE_SECONDS),
    )
    return started_at + budget


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise _SearchTimeout


def _snake_id(snake: SnakeState) -> str:
    return snake[0]


def _snake_body(snake: SnakeState) -> Tuple[Point, ...]:
    return snake[1]


def _snake_health(snake: SnakeState) -> int:
    return snake[2]


def _build_search_state(game_state: Dict) -> Dict[str, object]:
    board = game_state["board"]
    you_id = game_state["you"]["id"]
    snakes: List[SnakeState] = []
    for snake in board["snakes"]:
        body = tuple((seg["x"], seg["y"]) for seg in snake["body"])
        if not body:
            continue
        default_health = game_state["you"].get("health", STANDARD_FOOD_HEALTH) if snake["id"] == you_id else STANDARD_FOOD_HEALTH
        snakes.append((snake["id"], body, int(snake.get("health", default_health))))
    if not any(_snake_id(snake) == you_id for snake in snakes):
        raise ValueError("search state does not contain our snake")
    food = frozenset((f["x"], f["y"]) for f in board.get("food", []))
    return {
        "width": int(board["width"]),
        "height": int(board["height"]),
        "food": food,
        "snakes": tuple(snakes),
        "you_id": you_id,
        "had_opponents": any(_snake_id(snake) != you_id for snake in snakes),
    }


def _state_to_game_state(state: Dict[str, object], perspective_id: Optional[str] = None) -> Dict:
    snakes_json = []
    you_json = None
    for sid, body, health in state["snakes"]:
        head = body[0]
        snake_json = {
            "id": sid,
            "name": sid,
            "health": health,
            "body": [{"x": x, "y": y} for x, y in body],
            "head": {"x": head[0], "y": head[1]},
            "length": len(body),
            "latency": "0",
            "shout": "",
        }
        snakes_json.append(snake_json)
        if sid == (perspective_id or state["you_id"]):
            you_json = snake_json
    if you_json is None:
        you_json = snakes_json[0]
    return {
        "game": {"id": "search", "ruleset": {"name": "standard"}, "timeout": 500},
        "turn": 0,
        "board": {
            "height": state["height"],
            "width": state["width"],
            "food": [{"x": x, "y": y} for x, y in sorted(state["food"])],
            "hazards": [],
            "snakes": snakes_json,
        },
        "you": you_json,
    }


def _living_snake(state: Dict[str, object], snake_id: str) -> Optional[SnakeState]:
    for snake in state["snakes"]:
        if _snake_id(snake) == snake_id:
            return snake
    return None


def _enemy_snakes(state: Dict[str, object]) -> List[SnakeState]:
    you_id = state["you_id"]
    return [snake for snake in state["snakes"] if _snake_id(snake) != you_id]


def _move_point(point: Point, move: str) -> Point:
    dx, dy = DIRECTIONS[_valid_direction(move)]
    return point[0] + dx, point[1] + dy


def _body_cells_after_tail_release(state: Dict[str, object]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in state["snakes"]:
        for point in _snake_body(snake)[:-1]:
            occupied.add(point)
    return occupied


def _head_contest_danger(state: Dict[str, object], snake_id: str, dest: Point) -> bool:
    snake = _living_snake(state, snake_id)
    if snake is None:
        return False
    my_length = len(_snake_body(snake)) + (1 if dest in state["food"] else 0)
    for enemy in state["snakes"]:
        if _snake_id(enemy) == snake_id:
            continue
        enemy_head = _snake_body(enemy)[0]
        if _manhattan(enemy_head, dest) != 1:
            continue
        enemy_length = len(_snake_body(enemy)) + (1 if dest in state["food"] else 0)
        if enemy_length >= my_length:
            return True
    return False


def _classify_move_for_snake(state: Dict[str, object], snake_id: str, move: str) -> str:
    snake = _living_snake(state, snake_id)
    if snake is None or move not in DIRECTIONS:
        return "impossible"
    body = _snake_body(snake)
    nxt = _move_point(body[0], move)
    if not _in_bounds(nxt, state["width"], state["height"]):
        return "impossible"
    if nxt in _body_cells_after_tail_release(state):
        return "fatal_body"
    if _snake_health(snake) <= 1 and nxt not in state["food"]:
        return "fatal_starvation"
    if _head_contest_danger(state, snake_id, nxt):
        return "dangerous"
    return "survivable"


def _model_score_for_state_move(state: Dict[str, object], snake_id: str, move: str) -> float:
    if move not in DIRECTIONS:
        return -_BIG
    snake = _living_snake(state, snake_id)
    if snake is None:
        return -_BIG
    nxt = _move_point(_snake_body(snake)[0], move)
    if not _in_bounds(nxt, state["width"], state["height"]):
        return -_BIG
    try:
        return _model_score_for_move(_state_to_game_state(state, snake_id), move)
    except Exception:  # noqa: BLE001 - ordering should remain best-effort
        return 0.0


def _ordered_moves_for_snake(
    state: Dict[str, object],
    snake_id: str,
    baseline: Optional[str] = None,
) -> List[Dict[str, object]]:
    class_rank = {
        "survivable": 0,
        "dangerous": 1,
        "fatal_starvation": 2,
        "fatal_body": 3,
        "impossible": 4,
    }
    records = []
    for index, move in enumerate(VALID_MOVES):
        classification = _classify_move_for_snake(state, snake_id, move)
        model_score = _model_score_for_state_move(state, snake_id, move)
        if move == baseline:
            model_score += 0.001
        records.append(
            {
                "move": move,
                "classification": classification,
                "model_score": model_score,
                "direction_index": index,
                "rank": class_rank[classification],
            }
        )
    return sorted(records, key=lambda item: (item["rank"], -item["model_score"], item["direction_index"]))


def _playable_records(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Drop out-of-bounds moves when the snake has any in-bounds alternative."""
    in_bounds = [record for record in records if record["classification"] != "impossible"]
    return in_bounds or records


def _simulate_turn(state: Dict[str, object], moves: Dict[str, str]) -> Dict[str, object]:
    """Simulate one supported Standard turn without mutating the input state."""
    moved: List[Tuple[str, Tuple[Point, ...], int]] = []
    food = set(state["food"])

    for sid, body, health in state["snakes"]:
        move = _valid_direction(moves.get(sid))
        new_head = _move_point(body[0], move)
        new_body = (new_head,) + body[:-1]
        moved.append((sid, new_body, health - 1))

    eaten: Set[Point] = set()
    grown: List[Tuple[str, Tuple[Point, ...], int]] = []
    for sid, body, health in moved:
        if body[0] in food:
            tail = body[-1]
            body = body + (tail,)
            health = STANDARD_FOOD_HEALTH
            eaten.add(body[0])
        grown.append((sid, body, health))
    food.difference_update(eaten)

    eliminated: Set[str] = set()
    width, height = state["width"], state["height"]

    for sid, body, health in grown:
        head = body[0]
        if health <= 0:
            eliminated.add(sid)
        if not _in_bounds(head, width, height):
            eliminated.add(sid)
        body_after_head_stack = body[1:]
        while body_after_head_stack and body_after_head_stack[0] == head:
            body_after_head_stack = body_after_head_stack[1:]
        if head in body_after_head_stack:
            eliminated.add(sid)

    for sid, body, _health in grown:
        head = body[0]
        for other_sid, other_body, _other_health in grown:
            if other_sid == sid:
                continue
            if head in other_body[1:]:
                eliminated.add(sid)
                break

    heads: Dict[Point, List[Tuple[str, int]]] = {}
    for sid, body, _health in grown:
        heads.setdefault(body[0], []).append((sid, len(body)))
    for contenders in heads.values():
        if len(contenders) < 2:
            continue
        max_length = max(length for _sid, length in contenders)
        max_count = sum(1 for _sid, length in contenders if length == max_length)
        for sid, length in contenders:
            if length < max_length or max_count > 1:
                eliminated.add(sid)

    survivors = tuple((sid, body, health) for sid, body, health in grown if sid not in eliminated)
    return {
        "width": width,
        "height": height,
        "food": frozenset(food),
        "snakes": survivors,
        "you_id": state["you_id"],
        "had_opponents": state["had_opponents"],
    }


def _terminal_value(state: Dict[str, object], depth_remaining: int) -> Optional[float]:
    you = _living_snake(state, state["you_id"])
    if you is None:
        return LOSS_SCORE - depth_remaining * 1000.0
    if state["had_opponents"] and not _enemy_snakes(state):
        return WIN_SCORE + depth_remaining * 1000.0
    return None


def _select_relevant_opponents(state: Dict[str, object], our_move: str, depth: int) -> List[str]:
    you = _living_snake(state, state["you_id"])
    if you is None:
        return []
    our_dest = _move_point(_snake_body(you)[0], our_move)
    our_length = len(_snake_body(you)) + (1 if our_dest in state["food"] else 0)
    scored = []
    for enemy in _enemy_snakes(state):
        enemy_id = _snake_id(enemy)
        enemy_head = _snake_body(enemy)[0]
        distance = _manhattan(enemy_head, our_dest)
        can_contest = distance == 1
        score = 100.0 - distance * 8.0
        if can_contest:
            score += 1000.0
        if len(_snake_body(enemy)) >= our_length:
            score += 40.0
        for food in state["food"]:
            if _manhattan(enemy_head, food) <= depth + 1 and _manhattan(_snake_body(you)[0], food) <= depth + 1:
                score += 15.0
                break
        scored.append((0 if can_contest else 1, -score, enemy_id))
    scored.sort()
    return [enemy_id for _contest_rank, _score, enemy_id in scored[:2]]


def _predict_move_for_snake(state: Dict[str, object], snake_id: str) -> str:
    records = _playable_records(_ordered_moves_for_snake(state, snake_id))
    for record in records:
        if record["classification"] == "survivable":
            return record["move"]
    for record in records:
        if record["classification"] == "dangerous":
            return record["move"]
    return records[0]["move"]


def _search_best_move(game_state: Dict, baseline: str, deadline: float) -> Tuple[str, Dict[str, object]]:
    state = _build_search_state(game_state)
    root_moves = _playable_records(_ordered_moves_for_snake(state, state["you_id"], baseline=baseline))
    cache: Dict[Tuple[object, ...], float] = {}
    stats: Dict[str, object] = {
        "completed_depth": 0,
        "expanded_states": 0,
        "fallback_used": True,
        "reason": "no_completed_search",
    }
    completed_move: Optional[str] = None
    completed_score = float("-inf")

    for depth in range(1, MAX_SEARCH_DEPTH + 1):
        try:
            _check_deadline(deadline)
            best_move = None
            best_score = float("-inf")
            best_model_score = float("-inf")
            alpha = float("-inf")
            for record in root_moves:
                value = _move_value(state, record["move"], depth, deadline, cache, stats, alpha)
                model_score = float(record["model_score"])
                tie = (value, model_score, -int(record["direction_index"]))
                best_tie = (best_score, best_model_score, -VALID_MOVES.index(best_move)) if best_move else None
                if best_tie is None or tie > best_tie:
                    best_move = record["move"]
                    best_score = value
                    best_model_score = model_score
                alpha = max(alpha, best_score)
            completed_move = best_move
            completed_score = best_score
            stats["completed_depth"] = depth
            stats["best_score"] = completed_score
            stats["reason"] = "completed"
        except _SearchTimeout:
            stats["timed_out"] = True
            break

    if completed_move is None:
        return baseline, stats
    stats["fallback_used"] = False
    return completed_move, stats


def _move_value(
    state: Dict[str, object],
    our_move: str,
    depth: int,
    deadline: float,
    cache: Dict[Tuple[object, ...], float],
    stats: Dict[str, object],
    alpha: float,
) -> float:
    _check_deadline(deadline)
    you_id = state["you_id"]
    selected = _select_relevant_opponents(state, our_move, depth)
    option_lists = [
        [record["move"] for record in _playable_records(_ordered_moves_for_snake(state, enemy_id))]
        for enemy_id in selected
    ]
    worst = float("inf")

    for combo in product(*option_lists) if option_lists else [()]:
        _check_deadline(deadline)
        moves = {you_id: our_move}
        for enemy_id, enemy_move in zip(selected, combo):
            moves[enemy_id] = enemy_move
        for enemy in _enemy_snakes(state):
            enemy_id = _snake_id(enemy)
            if enemy_id not in moves:
                moves[enemy_id] = _predict_move_for_snake(state, enemy_id)
        next_state = _simulate_turn(state, moves)
        terminal = _terminal_value(next_state, depth - 1)
        if terminal is not None:
            value = terminal
        elif depth <= 1:
            value = _evaluate_leaf(next_state, deadline)
        else:
            value = _search_value(next_state, depth - 1, deadline, cache, stats, alpha, worst)
        worst = min(worst, value)
        if worst <= alpha:
            break
    return worst


def _search_value(
    state: Dict[str, object],
    depth: int,
    deadline: float,
    cache: Dict[Tuple[object, ...], float],
    stats: Dict[str, object],
    alpha: float,
    beta: float,
) -> float:
    _check_deadline(deadline)
    terminal = _terminal_value(state, depth)
    if terminal is not None:
        return terminal
    if depth <= 0:
        return _evaluate_leaf(state, deadline)

    key = _state_key(state, depth)
    cached = cache.get(key)
    if cached is not None:
        return cached

    stats["expanded_states"] = int(stats.get("expanded_states", 0)) + 1
    best = float("-inf")
    for record in _playable_records(_ordered_moves_for_snake(state, state["you_id"])):
        value = _move_value(state, record["move"], depth, deadline, cache, stats, alpha)
        best = max(best, value)
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    cache[key] = best
    return best


def _state_key(state: Dict[str, object], depth: int) -> Tuple[object, ...]:
    return (
        state["width"],
        state["height"],
        state["you_id"],
        depth,
        tuple(sorted(state["food"])),
        tuple((sid, body, health) for sid, body, health in state["snakes"]),
    )


def _release_times(state: Dict[str, object]) -> Dict[Point, int]:
    release: Dict[Point, int] = {}
    for _sid, body, _health in state["snakes"]:
        length = len(body)
        for index, point in enumerate(body):
            turns = length - index
            release[point] = max(release.get(point, 0), turns)
    return release


def _time_aware_dist(start: Point, state: Dict[str, object], deadline: float) -> Dict[Point, int]:
    width, height = state["width"], state["height"]
    release = _release_times(state)
    dist = {start: 0}
    dq = deque([start])
    checks = 0
    while dq:
        point = dq.popleft()
        d = dist[point]
        checks += 1
        if checks % 64 == 0:
            _check_deadline(deadline)
        for dx, dy in _NEIGHBORS:
            nxt = (point[0] + dx, point[1] + dy)
            if nxt in dist or not _in_bounds(nxt, width, height):
                continue
            arrival = d + 1
            if release.get(nxt, 0) > arrival:
                continue
            dist[nxt] = arrival
            dq.append(nxt)
    return dist


def _safe_exit_count(state: Dict[str, object], snake_id: str) -> int:
    return sum(
        1
        for move in VALID_MOVES
        if _classify_move_for_snake(state, snake_id, move) == "survivable"
    )


def _best_model_score_for_state(state: Dict[str, object]) -> float:
    game_state = _state_to_game_state(state, state["you_id"])
    try:
        legal = _legal_moves(game_state)
        if not legal:
            return -1000.0
        return max(_model_score_for_move(game_state, move) for move in legal)
    except Exception:  # noqa: BLE001 - leaf evaluation remains heuristic if model features fail
        return 0.0


def _evaluate_leaf(state: Dict[str, object], deadline: float) -> float:
    _check_deadline(deadline)
    terminal = _terminal_value(state, 0)
    if terminal is not None:
        return terminal

    you = _living_snake(state, state["you_id"])
    if you is None:
        return LOSS_SCORE

    my_body = _snake_body(you)
    my_head = my_body[0]
    my_dist = _time_aware_dist(my_head, state, deadline)
    enemies = _enemy_snakes(state)
    enemy_dists = [(_snake_id(enemy), len(_snake_body(enemy)), _time_aware_dist(_snake_body(enemy)[0], state, deadline)) for enemy in enemies]

    territory = 0.0
    strongest_enemy_space = 0
    for _enemy_id, _enemy_length, enemy_dist in enemy_dists:
        strongest_enemy_space = max(strongest_enemy_space, len(enemy_dist))

    for x in range(state["width"]):
        for y in range(state["height"]):
            point = (x, y)
            my_arrival = my_dist.get(point, _BIG)
            enemy_arrivals = [(dist.get(point, _BIG), length) for _enemy_id, length, dist in enemy_dists]
            if not enemy_arrivals:
                if my_arrival < _BIG:
                    territory += 1.0
                continue
            enemy_arrival = min(arrival for arrival, _length in enemy_arrivals)
            if my_arrival < enemy_arrival:
                territory += 1.0
            elif enemy_arrival < my_arrival:
                territory -= 0.75
            elif my_arrival < _BIG:
                max_enemy_length = max(length for arrival, length in enemy_arrivals if arrival == enemy_arrival)
                territory += 0.25 if len(my_body) > max_enemy_length else -0.25

    exits = _safe_exit_count(state, state["you_id"])
    tail_reachable = 1.0 if my_body[-1] in my_dist else 0.0
    health = _snake_health(you)
    length_advantage = len(my_body) - max((len(_snake_body(enemy)) for enemy in enemies), default=len(my_body))

    food_score = 0.0
    if state["food"]:
        food_distances = [my_dist.get(food, _BIG) for food in state["food"]]
        nearest_food = min(food_distances)
        if nearest_food < _BIG:
            food_score += max(0.0, health - nearest_food) * 1.5
            food_score += max(0.0, HUNGRY_THRESHOLD - health) * 0.8 / max(nearest_food, 1)
            for food in state["food"]:
                my_arrival = my_dist.get(food, _BIG)
                if my_arrival >= _BIG:
                    continue
                enemy_arrival = min((dist.get(food, _BIG) for _enemy_id, _length, dist in enemy_dists), default=_BIG)
                food_score += 12.0 if my_arrival < enemy_arrival else -6.0
        elif health < HUNGRY_THRESHOLD:
            food_score -= 75.0

    bottleneck_penalty = 60.0 if exits <= 1 else 0.0
    return (
        len(my_dist) * 3.0
        + territory * 4.0
        - strongest_enemy_space * 0.35
        + exits * 25.0
        + tail_reachable * 35.0
        + health * 0.45
        + food_score
        + length_advantage * 22.0
        - bottleneck_penalty
    )
