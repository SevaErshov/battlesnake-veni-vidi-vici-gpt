"""Tournament-grade move-selection logic for Battlesnake.

The bot is intentionally deterministic and dependency-free: every turn it
scores the four possible moves with survival-first heuristics that fit well
inside the 500 ms turn budget from the hackathon brief.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

MOVE_ORDER = ("up", "right", "down", "left")

_BIG = 1_000_000
_H2H_DEATH = 200_000
_BODY_DEATH = 1_000_000
_WALL_DEATH = 1_000_000

# Below these values the food term becomes increasingly aggressive.
HUNGRY_THRESHOLD = 55
STARVING_THRESHOLD = 25

# Standard Battlesnake hazard damage is 15, but rulesets may override it.
DEFAULT_HAZARD_DAMAGE = 15


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#0B8F79",
        "head": "all-seeing",
        "tail": "bolt",
        "version": "1.0.0-tournament",
    }


def choose_move(game_state: Dict) -> str:
    """Return the best move for the current turn."""
    board = game_state["board"]
    width: int = board["width"]
    height: int = board["height"]
    you = game_state["you"]
    head = _head(you)

    scored = []
    for move in MOVE_ORDER:
        dx, dy = DIRECTIONS[move]
        nxt = (head[0] + dx, head[1] + dy)
        scored.append((_score_move(game_state, move, nxt), move))

    scored.sort(reverse=True)
    best_score, best_move = scored[0]

    # If everything is truly dead, at least prefer a move that stays on board.
    if best_score <= -_BODY_DEATH:
        for move in MOVE_ORDER:
            dx, dy = DIRECTIONS[move]
            nxt = (head[0] + dx, head[1] + dy)
            if _in_bounds(nxt, width, height):
                return move
    return best_move


def _score_move(game_state: Dict, move: str, nxt: Point) -> float:
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]
    food: Set[Point] = {_point(f) for f in board.get("food", [])}
    hazards: Set[Point] = {_point(h) for h in board.get("hazards", [])}

    if not _in_bounds(nxt, width, height):
        return -_WALL_DEATH

    body_blockers = _solid_body_cells(board["snakes"])
    if nxt in body_blockers:
        return -_BODY_DEATH

    health: int = you["health"]
    my_body = _body(you)
    my_length = you["length"]
    eating = nxt in food
    next_length = my_length + (1 if eating else 0)
    next_body = [nxt] + (my_body if eating else my_body[:-1])

    h2h_risk, h2h_kill = _head_to_head_status(board, you, nxt, next_length)
    if h2h_risk:
        return -_H2H_DEATH + _space_tiebreak(board, next_body, nxt)

    blocked_after_move = _projected_blockers(board, you["id"], next_body)
    blocked_for_space = blocked_after_move - {nxt}

    space = _flood_fill(nxt, blocked_for_space, width, height, limit=width * height)
    exits = _exit_count(nxt, blocked_for_space, width, height)
    tail_reachable = _reachable_tail(nxt, next_body, blocked_for_space, width, height)
    voronoi = _voronoi_control(board, you, nxt, blocked_for_space)
    wall_distance = _wall_distance(nxt, width, height)
    hazard_damage = _hazard_damage(game_state)
    health_after = 100 if eating else health - 1
    if nxt in hazards:
        health_after -= hazard_damage

    score = 0.0

    # Space wins tournaments: do not voluntarily enter a pocket smaller than us.
    score += min(space, width * height) * 18.0
    if space < next_length:
        score -= 16_000.0 + (next_length - space) * 1_200.0
    elif space < next_length * 2:
        score -= (next_length * 2 - space) * 160.0

    if exits == 0:
        score -= 8_000.0
    elif exits == 1 and space < next_length * 3:
        score -= 1_800.0
    else:
        score += exits * 80.0

    if tail_reachable:
        score += 700.0

    score += voronoi * 7.0
    score += wall_distance * 28.0
    score += _straight_bonus(game_state, move)

    if h2h_kill and space >= next_length:
        score += 650.0

    score += _food_score(game_state, nxt, food, blocked_for_space, health_after, eating)
    score += _enemy_pressure_score(board, you, nxt, next_length)

    if nxt in hazards:
        score -= hazard_damage * 55.0
        if health_after <= 0:
            score -= _BODY_DEATH

    # Moving into any tail is legal when it vacates, but still carries extra
    # uncertainty for enemy tails. Prefer cleaner cells when the board allows it.
    if nxt in _enemy_tail_cells(board, you["id"]):
        score -= 140.0

    return score


def _food_score(
    game_state: Dict,
    nxt: Point,
    food: Set[Point],
    blocked: Set[Point],
    health_after: int,
    eating: bool,
) -> float:
    if not food:
        return 0.0

    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]
    health = you["health"]
    dist = _bfs_dist([nxt], blocked, width, height)
    food_distances = [dist[f] for f in food if f in dist]

    if not food_distances:
        return -2_500.0 if health < HUNGRY_THRESHOLD else -150.0

    nearest_food = min(food_distances)
    urgency = max(0, HUNGRY_THRESHOLD - health)
    starvation = max(0, STARVING_THRESHOLD - health)
    score = 0.0

    if eating:
        score += 170.0
        if health < 80:
            score += 520.0 + (80 - health) * 36.0

    # Prefer moves that keep food reachable before health runs out.
    score += max(0, width + height - nearest_food) * (8.0 + urgency * 0.75)
    if health_after - nearest_food <= 8:
        score += 1_500.0 + starvation * 70.0

    race_penalty = _food_race_penalty(game_state, dist, food)
    if health > STARVING_THRESHOLD:
        score -= race_penalty
    else:
        score -= race_penalty * 0.35

    return score


def _food_race_penalty(game_state: Dict, my_dist: Dict[Point, int], food: Set[Point]) -> float:
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]
    enemies = [snake for snake in board["snakes"] if snake["id"] != you["id"]]
    if not enemies:
        return 0.0

    enemy_heads = [_head(snake) for snake in enemies]
    enemy_blocked = _solid_body_cells(board["snakes"])
    enemy_dist = _bfs_dist(enemy_heads, enemy_blocked, width, height)

    penalty = 0.0
    my_length = you["length"]
    for item in food:
        md = my_dist.get(item)
        ed = enemy_dist.get(item)
        if md is None or ed is None:
            continue
        if ed <= md:
            close_big_enemy = any(
                snake["length"] >= my_length and _manhattan(_head(snake), item) <= md + 1
                for snake in enemies
            )
            penalty += 850.0 if close_big_enemy else 260.0
    return penalty


def _enemy_pressure_score(board: Dict, you: Dict, nxt: Point, next_length: int) -> float:
    score = 0.0
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            continue
        distance = _manhattan(nxt, _head(snake))
        if snake["length"] >= next_length:
            if distance == 1:
                score -= 950.0
            elif distance == 2:
                score -= 180.0
        else:
            if distance == 1:
                score += 170.0
    return score


def _head_to_head_status(board: Dict, you: Dict, target: Point, next_length: int) -> Tuple[bool, bool]:
    """Return ``(risk, kill_opportunity)`` for a target cell."""
    risk = False
    kill = False
    food = {_point(item) for item in board.get("food", [])}
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            continue
        if target not in _legal_enemy_targets(board, snake):
            continue
        enemy_next_length = snake["length"] + (1 if target in food else 0)
        if enemy_next_length >= next_length:
            risk = True
        else:
            kill = True
    return risk, kill


def _legal_enemy_targets(board: Dict, snake: Dict) -> Set[Point]:
    width: int = board["width"]
    height: int = board["height"]
    head = _head(snake)
    blockers = _solid_body_cells(board["snakes"])
    targets = set()
    for dx, dy in DIRECTIONS.values():
        target = (head[0] + dx, head[1] + dy)
        if _in_bounds(target, width, height) and target not in blockers:
            targets.add(target)
    return targets


def _projected_blockers(board: Dict, my_id: str, next_body: List[Point]) -> Set[Point]:
    blockers = set(next_body)
    for snake in board["snakes"]:
        if snake["id"] == my_id:
            continue
        body = _body(snake)
        if not body:
            continue
        tail_vacates = not _tail_is_stacked(body)
        blockers.update(body[:-1] if tail_vacates else body)
    return blockers


def _solid_body_cells(snakes: List[Dict]) -> Set[Point]:
    """Cells that should be treated as occupied before this turn resolves."""
    occupied: Set[Point] = set()
    for snake in snakes:
        body = _body(snake)
        if not body:
            continue
        tail_vacates = not _tail_is_stacked(body)
        occupied.update(body[:-1] if tail_vacates else body)
    return occupied


def _enemy_tail_cells(board: Dict, my_id: str) -> Set[Point]:
    tails = set()
    for snake in board["snakes"]:
        if snake["id"] == my_id:
            continue
        body = _body(snake)
        if body and not _tail_is_stacked(body):
            tails.add(body[-1])
    return tails


def _tail_is_stacked(body: List[Point]) -> bool:
    return len(body) >= 2 and body[-1] == body[-2]


def _reachable_tail(start: Point, next_body: List[Point], blocked: Set[Point], width: int, height: int) -> bool:
    if not next_body:
        return False
    tail = next_body[-1]
    dist = _bfs_dist([start], blocked - {tail}, width, height)
    return tail in dist


def _voronoi_control(board: Dict, you: Dict, start: Point, blocked: Set[Point]) -> int:
    width: int = board["width"]
    height: int = board["height"]
    enemies = [snake for snake in board["snakes"] if snake["id"] != you["id"]]
    if not enemies:
        return width * height

    my_dist = _bfs_dist([start], blocked, width, height)
    enemy_dist = _bfs_dist([_head(snake) for snake in enemies], blocked, width, height)
    return sum(1 for cell, dist in my_dist.items() if dist < enemy_dist.get(cell, _BIG))


def _space_tiebreak(board: Dict, next_body: List[Point], start: Point) -> float:
    width: int = board["width"]
    height: int = board["height"]
    blocked = set(next_body) - {start}
    return float(_flood_fill(start, blocked, width, height, limit=width * height))


def _exit_count(start: Point, blocked: Set[Point], width: int, height: int) -> int:
    count = 0
    for dx, dy in DIRECTIONS.values():
        nxt = (start[0] + dx, start[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in blocked:
            count += 1
    return count


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
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


def _bfs_dist(sources: Iterable[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    dist: Dict[Point, int] = {}
    queue = deque()
    for source in sources:
        if source in dist or not _in_bounds(source, width, height):
            continue
        dist[source] = 0
        queue.append(source)

    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in dist:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in blocked:
                continue
            dist[nbr] = dist[(x, y)] + 1
            queue.append(nbr)
    return dist


def _straight_bonus(game_state: Dict, move: str) -> float:
    direction = _current_direction(game_state["you"])
    return 22.0 if direction == move else 0.0


def _current_direction(you: Dict) -> Optional[str]:
    body = _body(you)
    if len(body) < 2:
        return None
    head = body[0]
    neck = body[1]
    delta = (head[0] - neck[0], head[1] - neck[1])
    for move, vector in DIRECTIONS.items():
        if vector == delta:
            return move
    return None


def _hazard_damage(game_state: Dict) -> int:
    return int(
        game_state.get("game", {})
        .get("ruleset", {})
        .get("settings", {})
        .get("hazardDamagePerTurn", DEFAULT_HAZARD_DAMAGE)
    )


def _head(snake: Dict) -> Point:
    return _point(snake["head"])


def _body(snake: Dict) -> List[Point]:
    return [_point(part) for part in snake.get("body", [])]


def _point(item: Dict) -> Point:
    return (item["x"], item["y"])


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _wall_distance(p: Point, width: int, height: int) -> int:
    return min(p[0], width - 1 - p[0], p[1], height - 1 - p[1])
