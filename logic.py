"""Tournament-grade move-selection logic for Battlesnake.

The policy is deterministic and dependency-free. It ranks all four moves with
survival-first heuristics, then layers in food selection, territory control, and
pressure against smaller snakes.

Board coordinates: (0, 0) is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

MOVE_ORDER = ("up", "right", "down", "left")

BIG = 1_000_000
FATAL = -1_000_000.0
H2H_DEATH = 220_000.0
TRAP_DEATH = 35_000.0

CRITICAL_HEALTH = 22
HUNGRY_HEALTH = 48
FOOD_OK_HEALTH = 68
DEFAULT_HAZARD_DAMAGE = 15

DEBUG = os.getenv("BATTLESNAKE_DEBUG", "0") == "1"


@dataclass(frozen=True)
class MoveScore:
    move: str
    score: float
    notes: str = ""


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from GET /."""
    return {
        "apiversion": "1",
        "author": "veni-vidi-vici-gpt",
        "color": "#00B894",
        "head": "evil",
        "tail": "sharp",
        "version": "2.0.0-seva",
    }


def choose_move(game_state: Dict) -> str:
    """Return the best move for this turn. Never raise into the web handler."""
    try:
        ranked = rank_moves(game_state)
        if ranked:
            if DEBUG:
                print(" | ".join(f"{item.move}:{item.score:.1f}:{item.notes}" for item in ranked))
            return ranked[0].move
    except Exception as exc:  # noqa: BLE001 - gameplay must continue
        if DEBUG:
            print(f"choose_move failed: {exc!r}")
    return _emergency_move(game_state)


def rank_moves(game_state: Dict) -> List[MoveScore]:
    """Score all non-immediate-death moves in descending order."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)

    ranked: List[MoveScore] = []
    for move in MOVE_ORDER:
        dx, dy = DIRECTIONS[move]
        nxt = (head[0] + dx, head[1] + dy)
        scored = _score_move(game_state, move, nxt, width, height)
        if scored is not None:
            ranked.append(scored)

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _score_move(game_state: Dict, move: str, nxt: Point, width: int, height: int) -> Optional[MoveScore]:
    board = game_state["board"]
    you = game_state["you"]
    my_id = you["id"]
    my_body = _body(you)
    my_len = int(you["length"])
    health = int(you["health"])
    foods = _foods(board)
    food_set = set(foods)
    hazards = _hazards(board)
    enemies = _enemies(board, my_id)
    notes: List[str] = []

    if not _in_bounds(nxt, width, height):
        return None

    entry_blockers = _entry_blockers(board["snakes"], my_id, nxt, food_set)
    if nxt in entry_blockers:
        return None

    eating = nxt in food_set
    next_len = my_len + (1 if eating else 0)
    next_body = [nxt] + (my_body if eating else my_body[:-1])

    health_after = 100 if eating else health - 1
    if nxt in hazards:
        health_after -= _hazard_damage(game_state)
        notes.append("hazard")
    if health_after <= 0:
        return MoveScore(move, FATAL, "starves")

    h2h_risk, h2h_kill = _head_to_head_status(board, you, nxt, next_len, food_set)
    if h2h_risk:
        return MoveScore(move, -H2H_DEATH + _space_tiebreak(board, my_id, next_body, nxt), "h2h-risk")

    projected_blockers = _projected_blockers(board, my_id, next_body, food_set)
    blockers_for_space = projected_blockers - {nxt}

    space = _flood_fill(nxt, blockers_for_space, width, height, limit=width * height)
    exits = _exit_count(nxt, blockers_for_space, width, height)
    second_steps = _second_step_options(nxt, blockers_for_space, width, height)
    tail_reachable = _reachable_tail(nxt, next_body, blockers_for_space, width, height)
    voronoi = _voronoi_control(board, you, nxt, blockers_for_space)
    wall = _wall_distance(nxt, width, height)
    center = abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2)

    score = 0.0

    # First priority: do not enter pockets that cannot hold us.
    score += space * 20.0
    min_space = max(next_len + 2, int(next_len * 1.35))
    if space < next_len:
        score -= TRAP_DEATH + (next_len - space) * 2_000.0
        notes.append(f"tiny-space={space}")
    elif space < min_space:
        score -= 5_000.0 + (min_space - space) * 550.0
        notes.append(f"tight-space={space}")
    elif space < next_len * 2:
        score -= (next_len * 2 - space) * 130.0

    if exits == 0:
        score -= 12_000.0
        notes.append("dead-end")
    elif exits == 1 and space < next_len * 3:
        score -= 1_900.0
        notes.append("corridor")
    else:
        score += exits * 95.0
    score += second_steps * 38.0

    if tail_reachable:
        score += 850.0
        notes.append("tail")

    score += voronoi * 9.0
    score += wall * 30.0
    score -= center * (9.0 if health > HUNGRY_HEALTH else 3.5)
    score += _straight_bonus(you, move)

    if h2h_kill and space >= next_len:
        score += 850.0
        notes.append("h2h-kill")

    score += _food_score(
        game_state=game_state,
        nxt=nxt,
        foods=foods,
        food_set=food_set,
        blocked=blockers_for_space,
        health=health,
        health_after=health_after,
        eating=eating,
        space=space,
    )
    score += _enemy_pressure_score(
        board=board,
        you=you,
        my_next=nxt,
        next_body=next_body,
        next_len=next_len,
        food_set=food_set,
        blockers_after=projected_blockers,
    )

    if nxt in hazards:
        damage = _hazard_damage(game_state)
        score -= damage * (70.0 if health > CRITICAL_HEALTH else 25.0)

    if nxt in _enemy_tail_cells(board, my_id):
        score -= 150.0

    score += {"up": 0.04, "right": 0.03, "down": 0.02, "left": 0.01}[move]
    return MoveScore(move, score, ",".join(notes))


def _food_score(
    *,
    game_state: Dict,
    nxt: Point,
    foods: Sequence[Point],
    food_set: Set[Point],
    blocked: Set[Point],
    health: int,
    health_after: int,
    eating: bool,
    space: int,
) -> float:
    if not foods:
        return 0.0

    board = game_state["board"]
    you = game_state["you"]
    width, height = int(board["width"]), int(board["height"])
    dist = _bfs_dist([nxt], blocked, width, height)
    reachable_food = [(food, dist[food]) for food in foods if food in dist]
    if not reachable_food:
        return -3_500.0 if health < HUNGRY_HEALTH else -220.0

    nearest_dist = min(distance for _, distance in reachable_food)
    urgency = max(0.0, (FOOD_OK_HEALTH - health) / FOOD_OK_HEALTH)
    starving = max(0.0, (CRITICAL_HEALTH - health) / max(1, CRITICAL_HEALTH))

    score = 0.0
    if eating:
        score += 220.0
        score += max(0, 90 - health) * 34.0
        if space < int(you["length"]) + 5 and health > CRITICAL_HEALTH:
            score -= 650.0

    score += max(0, width + height - nearest_dist) * (7.0 + urgency * 34.0)
    if health_after - nearest_dist <= 7:
        score += 1_900.0 + starving * 1_400.0

    race_penalty = _food_race_penalty(game_state, dist, food_set)
    if health <= CRITICAL_HEALTH:
        score -= race_penalty * 0.25
    elif health <= HUNGRY_HEALTH:
        score -= race_penalty * 0.65
    else:
        score -= race_penalty

    # When healthy, avoid letting food tunnel vision beat territory.
    if health > FOOD_OK_HEALTH:
        score *= 0.35
    return score


def _food_race_penalty(game_state: Dict, my_dist: Dict[Point, int], foods: Set[Point]) -> float:
    board = game_state["board"]
    you = game_state["you"]
    width, height = int(board["width"]), int(board["height"])
    enemies = _enemies(board, you["id"])
    if not enemies:
        return 0.0

    enemy_heads = [_head(enemy) for enemy in enemies]
    enemy_blocked = _solid_cells(board["snakes"], foods) - set(enemy_heads)
    enemy_dist = _bfs_dist(enemy_heads, enemy_blocked, width, height)
    my_len = int(you["length"])

    penalty = 0.0
    for food in foods:
        md = my_dist.get(food)
        ed = enemy_dist.get(food)
        if md is None or ed is None:
            continue
        if ed <= md:
            bigger_close = any(
                int(enemy["length"]) >= my_len and _manhattan(_head(enemy), food) <= md + 1 for enemy in enemies
            )
            penalty += 950.0 if bigger_close else 300.0
    return penalty


def _enemy_pressure_score(
    *,
    board: Dict,
    you: Dict,
    my_next: Point,
    next_body: Sequence[Point],
    next_len: int,
    food_set: Set[Point],
    blockers_after: Set[Point],
) -> float:
    score = 0.0
    my_id = you["id"]
    width, height = int(board["width"]), int(board["height"])

    for enemy in _enemies(board, my_id):
        enemy_len = int(enemy["length"])
        enemy_head = _head(enemy)
        distance = _manhattan(my_next, enemy_head)

        if enemy_len >= next_len:
            if distance == 1:
                score -= 1_100.0
            elif distance == 2:
                score -= 240.0
            continue

        if distance == 1:
            score += 230.0
        elif distance == 2:
            score += 75.0

        before = _enemy_targets(board["snakes"], enemy, width, height, food_set, override_my_body=None, my_id=my_id)
        after = _enemy_targets(
            board["snakes"], enemy, width, height, food_set, override_my_body=next_body, my_id=my_id
        )
        reduction = max(0, len(before) - len(after))
        score += reduction * 170.0

        if len(after) == 0:
            score += 1_250.0
        elif len(after) == 1:
            score += 470.0
        elif len(after) == 2:
            score += 135.0

        if after:
            enemy_blocked = _blockers_for_enemy(board["snakes"], enemy["id"], food_set, next_body, my_id)
            enemy_areas = [
                _flood_fill(target, enemy_blocked - {target}, width, height, limit=width * height)
                for target in after
            ]
            best_area = max(enemy_areas) if enemy_areas else 0
            if best_area < enemy_len + 3:
                score += 520.0
            score += max(0, 20 - best_area) * 20.0

        score += _cutoff_score(my_next, enemy_head, blockers_after, width, height)

    return score


def _cutoff_score(my_next: Point, enemy_head: Point, blocked: Set[Point], width: int, height: int) -> float:
    if _manhattan(my_next, enemy_head) > 4:
        return 0.0
    my_dist = _bfs_dist([my_next], blocked - {my_next}, width, height)
    enemy_dist = _bfs_dist([enemy_head], blocked - {enemy_head}, width, height)
    contested = 0
    for cell, ed in enemy_dist.items():
        md = my_dist.get(cell)
        if md is not None and md <= ed and ed <= 5:
            contested += 1
    return contested * 18.0


def _head_to_head_status(
    board: Dict, you: Dict, target: Point, my_next_len: int, food_set: Set[Point]
) -> Tuple[bool, bool]:
    risk = False
    kill = False
    width, height = int(board["width"]), int(board["height"])
    for enemy in _enemies(board, you["id"]):
        if target not in _enemy_targets(
            board["snakes"], enemy, width, height, food_set, override_my_body=None, my_id=you["id"]
        ):
            continue
        enemy_next_len = int(enemy["length"]) + (1 if target in food_set else 0)
        if enemy_next_len >= my_next_len:
            risk = True
        else:
            kill = True
    return risk, kill


def _entry_blockers(snakes: Sequence[Dict], my_id: str, my_next: Point, food_set: Set[Point]) -> Set[Point]:
    blocked = _all_cells(snakes)
    for snake in snakes:
        body = _body(snake)
        if not body:
            continue
        tail = body[-1]
        if snake["id"] == my_id:
            if my_next not in food_set and not _tail_is_stacked(body):
                blocked.discard(tail)
        elif not _can_eat_next_turn(snake, food_set) and not _tail_is_stacked(body):
            blocked.discard(tail)
    return blocked


def _projected_blockers(board: Dict, my_id: str, next_body: Sequence[Point], food_set: Set[Point]) -> Set[Point]:
    blocked = set(next_body)
    for snake in board["snakes"]:
        if snake["id"] == my_id:
            continue
        body = _body(snake)
        if not body:
            continue
        blocked.update(body)
        if not _can_eat_next_turn(snake, food_set) and not _tail_is_stacked(body):
            blocked.discard(body[-1])
    return blocked


def _blockers_for_enemy(
    snakes: Sequence[Dict],
    enemy_id: str,
    food_set: Set[Point],
    override_my_body: Sequence[Point],
    my_id: str,
) -> Set[Point]:
    blocked: Set[Point] = set()
    for snake in snakes:
        sid = snake["id"]
        if sid == my_id:
            blocked.update(override_my_body)
            continue
        body = _body(snake)
        blocked.update(body)
        if sid != enemy_id and body and not _can_eat_next_turn(snake, food_set) and not _tail_is_stacked(body):
            blocked.discard(body[-1])
        if sid == enemy_id and body and not _tail_is_stacked(body):
            blocked.discard(body[-1])
    return blocked


def _solid_cells(snakes: Sequence[Dict], food_set: Set[Point]) -> Set[Point]:
    blocked = _all_cells(snakes)
    for snake in snakes:
        body = _body(snake)
        if body and not _can_eat_next_turn(snake, food_set) and not _tail_is_stacked(body):
            blocked.discard(body[-1])
    return blocked


def _enemy_targets(
    snakes: Sequence[Dict],
    enemy: Dict,
    width: int,
    height: int,
    food_set: Set[Point],
    *,
    override_my_body: Optional[Sequence[Point]],
    my_id: str,
) -> List[Point]:
    if override_my_body is None:
        blocked = _solid_cells(snakes, food_set)
    else:
        blocked = _blockers_for_enemy(snakes, enemy["id"], food_set, override_my_body, my_id)
    head = _head(enemy)
    targets = []
    for dx, dy in DIRECTIONS.values():
        target = (head[0] + dx, head[1] + dy)
        if _in_bounds(target, width, height) and target not in blocked:
            targets.append(target)
    return targets


def _voronoi_control(board: Dict, you: Dict, start: Point, blocked: Set[Point]) -> int:
    width, height = int(board["width"]), int(board["height"])
    enemies = _enemies(board, you["id"])
    if not enemies:
        return width * height
    my_dist = _bfs_dist([start], blocked, width, height)
    enemy_heads = [_head(enemy) for enemy in enemies]
    enemy_dist = _bfs_dist(enemy_heads, blocked - set(enemy_heads), width, height)
    return sum(1 for cell, dist in my_dist.items() if dist < enemy_dist.get(cell, BIG))


def _space_tiebreak(board: Dict, my_id: str, next_body: Sequence[Point], start: Point) -> float:
    width, height = int(board["width"]), int(board["height"])
    food_set = set(_foods(board))
    blocked = _projected_blockers(board, my_id, next_body, food_set) - {start}
    return float(_flood_fill(start, blocked, width, height, limit=width * height))


def _emergency_move(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    occupied = _all_cells(board["snakes"])
    for move in MOVE_ORDER:
        dx, dy = DIRECTIONS[move]
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in occupied:
            return move
    for move in MOVE_ORDER:
        dx, dy = DIRECTIONS[move]
        if _in_bounds((head[0] + dx, head[1] + dy), width, height):
            return move
    return "up"


def _foods(board: Dict) -> List[Point]:
    return [_point(food) for food in board.get("food", [])]


def _hazards(board: Dict) -> Set[Point]:
    return {_point(hazard) for hazard in board.get("hazards", [])}


def _enemies(board: Dict, my_id: str) -> List[Dict]:
    return [snake for snake in board.get("snakes", []) if snake.get("id") != my_id]


def _head(snake: Dict) -> Point:
    return _point(snake["head"])


def _body(snake: Dict) -> List[Point]:
    return [_point(part) for part in snake.get("body", [])]


def _point(item: Dict) -> Point:
    return (int(item["x"]), int(item["y"]))


def _all_cells(snakes: Iterable[Dict]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in snakes:
        occupied.update(_body(snake))
    return occupied


def _tail_is_stacked(body: Sequence[Point]) -> bool:
    return len(body) >= 2 and body[-1] == body[-2]


def _can_eat_next_turn(snake: Dict, food_set: Set[Point]) -> bool:
    if not food_set:
        return False
    head = _head(snake)
    return any(_manhattan(head, food) == 1 for food in food_set)


def _enemy_tail_cells(board: Dict, my_id: str) -> Set[Point]:
    tails: Set[Point] = set()
    for snake in _enemies(board, my_id):
        body = _body(snake)
        if body and not _tail_is_stacked(body):
            tails.add(body[-1])
    return tails


def _reachable_tail(start: Point, next_body: Sequence[Point], blocked: Set[Point], width: int, height: int) -> bool:
    if not next_body:
        return False
    tail = next_body[-1]
    dist = _bfs_dist([start], blocked - {tail}, width, height)
    return tail in dist


def _exit_count(start: Point, blocked: Set[Point], width: int, height: int) -> int:
    return sum(
        1
        for dx, dy in DIRECTIONS.values()
        if _in_bounds((start[0] + dx, start[1] + dy), width, height)
        and (start[0] + dx, start[1] + dy) not in blocked
    )


def _second_step_options(start: Point, blocked: Set[Point], width: int, height: int) -> int:
    count = 0
    for dx, dy in DIRECTIONS.values():
        nxt = (start[0] + dx, start[1] + dy)
        if not _in_bounds(nxt, width, height) or nxt in blocked:
            continue
        count += 1 + min(2, _exit_count(nxt, blocked | {start}, width, height))
    return count


def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: int) -> int:
    if not _in_bounds(start, width, height) or start in blocked:
        return 0
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if nxt in seen or nxt in blocked or not _in_bounds(nxt, width, height):
                continue
            seen.add(nxt)
            stack.append(nxt)
    return count


def _bfs_dist(sources: Iterable[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    dist: Dict[Point, int] = {}
    queue: deque[Point] = deque()
    for source in sources:
        if source in dist or not _in_bounds(source, width, height) or source in blocked:
            continue
        dist[source] = 0
        queue.append(source)
    while queue:
        x, y = queue.popleft()
        current = dist[(x, y)]
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if nxt in dist or nxt in blocked or not _in_bounds(nxt, width, height):
                continue
            dist[nxt] = current + 1
            queue.append(nxt)
    return dist


def _straight_bonus(you: Dict, move: str) -> float:
    direction = _current_direction(you)
    return 24.0 if direction == move else 0.0


def _current_direction(you: Dict) -> Optional[str]:
    body = _body(you)
    if len(body) < 2:
        return None
    delta = (body[0][0] - body[1][0], body[0][1] - body[1][1])
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


def _wall_distance(p: Point, width: int, height: int) -> int:
    return min(p[0], width - 1 - p[0], p[1], height - 1 - p[1])


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
