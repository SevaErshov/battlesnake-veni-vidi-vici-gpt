"""Predator/Kill-Mode Battlesnake logic.

The bot is intentionally pure Python and deterministic: it must answer the
Battlesnake `/move` request quickly and never crash the server.

Main idea:
1. Safety shield first: walls, bodies, equal/bigger enemy head-to-head danger,
   trap detection, tail-aware movement.
2. If we are hungry, choose safe food, not merely nearest food.
3. If we are longer than at least one enemy, activate Predator Mode: reduce the
   prey's legal moves, block exits, pressure smaller heads and steal their food.

Board coordinates: (0, 0) is the bottom-left corner.
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

BIG = 10_000
CRITICAL_HEALTH = 25
HUNGRY_HEALTH = 45
LOW_HEALTH = 60

# Hard safety penalties. These should dominate any hunting reward.
FATAL = -1_000_000.0
HEAD_TO_HEAD_FATAL_PENALTY = 50_000.0
TRAP_PENALTY = 2_500.0

DEBUG = os.getenv("BATTLESNAKE_DEBUG", "0") == "1"


@dataclass(frozen=True)
class MoveScore:
    move: str
    score: float
    mode: str
    notes: str = ""


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from GET /."""
    return {
        "apiversion": "1",
        "author": "veni-vidi-vici-gpt",
        "color": "#00B894",
        "head": "evil",
        "tail": "sharp",
        "version": "1.0.0-killmode",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move. The function must never raise."""
    try:
        ranked = rank_moves(game_state)
        if ranked:
            if DEBUG:
                print(" | ".join(f"{r.move}:{r.score:.1f}:{r.mode}:{r.notes}" for r in ranked))
            return ranked[0].move
    except Exception as exc:  # noqa: BLE001 - gameplay must continue
        if DEBUG:
            print(f"choose_move failed: {exc!r}")

    # Last resort: choose any move that is not immediately inside wall/body.
    return _emergency_move(game_state)


def rank_moves(game_state: Dict) -> List[MoveScore]:
    """Score all legal-ish moves and return them in descending order."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = int(board["width"]), int(board["height"])
    foods = _foods(board)
    head = _head(you)
    my_len = int(you["length"])
    health = int(you["health"])
    enemies = _enemies(board, you["id"])

    mode = _choose_mode(you, enemies)
    ranked: List[MoveScore] = []

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        scored = _score_candidate(
            game_state=game_state,
            move=move,
            nxt=nxt,
            mode=mode,
            width=width,
            height=height,
            foods=foods,
            enemies=enemies,
            my_len=my_len,
            health=health,
        )
        if scored is not None:
            ranked.append(scored)

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked


def _choose_mode(you: Dict, enemies: Sequence[Dict]) -> str:
    health = int(you["health"])
    my_len = int(you["length"])
    if health <= CRITICAL_HEALTH:
        return "HEALTH_CRISIS"
    if any(int(enemy["length"]) < my_len for enemy in enemies):
        return "PREDATOR"
    if health <= HUNGRY_HEALTH:
        return "FOOD_SAFE"
    if len(enemies) <= 1:
        return "ENDGAME_CONTROL"
    return "CONTROL"


def _score_candidate(
    *,
    game_state: Dict,
    move: str,
    nxt: Point,
    mode: str,
    width: int,
    height: int,
    foods: Sequence[Point],
    enemies: Sequence[Dict],
    my_len: int,
    health: int,
) -> Optional[MoveScore]:
    board = game_state["board"]
    you = game_state["you"]
    head = _head(you)

    if not _in_bounds(nxt, width, height):
        return None

    # Candidate-specific tail-aware body map: if we do not eat, our tail frees;
    # enemy tails free only when they are unlikely to eat next turn.
    blocked_for_entry = _blocked_for_my_entry(board["snakes"], you["id"], nxt, foods)
    if nxt in blocked_for_entry:
        return None

    my_body_after = _my_body_after_move(you, nxt, foods)
    blocked_after = _blocked_after_my_move(board["snakes"], you["id"], my_body_after, foods)
    blocked_after_without_head = set(blocked_after)
    blocked_after_without_head.discard(nxt)

    score = 0.0
    notes: List[str] = []

    # 1) Safety / survival. This layer should dominate everything.
    risk_cells = _head_to_head_risk_cells(board["snakes"], you["id"], my_len, width, height, foods)
    kill_cells = _head_to_head_kill_cells(board["snakes"], you["id"], my_len, width, height, foods)

    if nxt in risk_cells:
        score -= HEAD_TO_HEAD_FATAL_PENALTY
        notes.append("h2h-risk")

    open_area = _flood_fill(nxt, blocked_after_without_head, width, height, limit=width * height)
    capped_area = min(open_area, my_len + 8)
    exits = _exit_count(nxt, blocked_after_without_head, width, height)
    reaches_tail = _can_reach_own_tail(nxt, you, blocked_after_without_head, width, height)
    second_step_options = _second_step_options(nxt, blocked_after_without_head, width, height)

    score += capped_area * 14.0
    score += open_area * 1.8
    score += exits * 35.0
    score += second_step_options * 30.0
    if reaches_tail:
        score += 140.0
        notes.append("tail")

    # Pocket detector: entering too-small areas usually kills us later.
    min_required_space = min(width * height, max(my_len + 2, int(my_len * 1.25)))
    if open_area < min_required_space:
        score -= TRAP_PENALTY + (min_required_space - open_area) * 120.0
        notes.append(f"trap-area={open_area}")
    if exits == 0:
        score += FATAL / 2
        notes.append("no-exit")
    elif exits == 1 and open_area < my_len + 5:
        score -= 650.0
        notes.append("thin-corridor")

    # Avoid hugging equal/bigger heads even when not stepping into direct h2h.
    nearest_big_head = min(
        (_manhattan(nxt, _head(enemy)) for enemy in enemies if int(enemy["length"]) >= my_len),
        default=BIG,
    )
    if nearest_big_head == 1:
        score -= 500.0
    elif nearest_big_head == 2:
        score -= 130.0

    # Center/territory helps when no urgent food/kill exists.
    center_dist = abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2)
    wall_dist = min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
    score += wall_dist * 12.0
    score -= center_dist * (7.0 if mode in {"CONTROL", "ENDGAME_CONTROL"} else 3.0)

    # 2) Food logic. Nearest food is not always good: we check race and space.
    food_score = _safe_food_score(
        head=head,
        nxt=nxt,
        foods=foods,
        enemies=enemies,
        my_len=my_len,
        health=health,
        open_area=open_area,
        width=width,
        height=height,
    )
    score += food_score
    if food_score > 150:
        notes.append("food")

    # 3) Predator/Kill Mode: if we are longer, reduce smaller snakes' options.
    predator_score = _predator_score(
        game_state=game_state,
        my_next=nxt,
        my_body_after=my_body_after,
        blocked_after=blocked_after,
        foods=foods,
        width=width,
        height=height,
    )
    if mode == "HEALTH_CRISIS":
        # Do not throw the game for a kill when starving.
        predator_score *= 0.25
    elif mode == "FOOD_SAFE":
        predator_score *= 0.55
    elif mode == "ENDGAME_CONTROL":
        predator_score *= 1.20
    else:
        predator_score *= 1.00
    score += predator_score
    if predator_score > 120:
        notes.append("kill")

    # If the move itself is a winning h2h bait against smaller snake, reward it,
    # but only after survival checks have been applied above.
    if nxt in kill_cells:
        score += 260.0
        notes.append("h2h-kill-bait")

    # Tiny deterministic tie-breaker: prefer continuing direction less randomly
    # (Battlesnake sends latency-sensitive requests; deterministic is easier to debug).
    score += {"up": 0.04, "right": 0.03, "down": 0.02, "left": 0.01}[move]

    return MoveScore(move=move, score=score, mode=mode, notes=",".join(notes))


# ---------------------------------------------------------------------------
# Food strategy


def _safe_food_score(
    *,
    head: Point,
    nxt: Point,
    foods: Sequence[Point],
    enemies: Sequence[Dict],
    my_len: int,
    health: int,
    open_area: int,
    width: int,
    height: int,
) -> float:
    if not foods:
        return 0.0

    board_span = width + height
    urgency = max(0.0, (LOW_HEALTH - health) / LOW_HEALTH)
    critical = health <= CRITICAL_HEALTH
    best = -BIG

    for food in foods:
        dist_now = _manhattan(head, food)
        dist_next = _manhattan(nxt, food)
        improves = dist_now - dist_next

        enemy_bigger_or_equal_dist = min(
            (_manhattan(_head(enemy), food) for enemy in enemies if int(enemy["length"]) >= my_len),
            default=BIG,
        )
        enemy_any_dist = min((_manhattan(_head(enemy), food) for enemy in enemies), default=BIG)

        # Food race: avoid food where equal/bigger enemy arrives no later.
        race_penalty = 0.0
        if enemy_bigger_or_equal_dist <= dist_next:
            race_penalty += 180.0
        elif enemy_any_dist < dist_next:
            race_penalty += 40.0

        # Food in a tiny pocket is bait, especially if we are not starving.
        trap_penalty = 0.0
        if dist_next == 0 and open_area < my_len + 4:
            trap_penalty = 450.0 if not critical else 160.0

        value = 0.0
        value += improves * (90.0 + urgency * 160.0)
        value += max(0, board_span - dist_next) * (6.0 + urgency * 18.0)
        if dist_next == 0:
            value += 180.0 + urgency * 420.0
        value -= race_penalty
        value -= trap_penalty
        best = max(best, value)

    # When full, food should not dominate territory/kill. When starving, it must.
    if health > LOW_HEALTH:
        best *= 0.25
    elif health > HUNGRY_HEALTH:
        best *= 0.60
    return best


# ---------------------------------------------------------------------------
# Predator / Kill Mode


def _predator_score(
    *,
    game_state: Dict,
    my_next: Point,
    my_body_after: Sequence[Point],
    blocked_after: Set[Point],
    foods: Sequence[Point],
    width: int,
    height: int,
) -> float:
    board = game_state["board"]
    you = game_state["you"]
    my_len = int(you["length"])
    prey = [enemy for enemy in _enemies(board, you["id"]) if int(enemy["length"]) < my_len]
    if not prey:
        return 0.0

    total = 0.0
    for enemy in prey:
        ehead = _head(enemy)
        distance = _manhattan(my_next, ehead)

        before_moves = _enemy_safe_moves(
            board["snakes"], enemy["id"], width, height, foods, override_my_body=None, my_id=you["id"]
        )
        after_moves = _enemy_safe_moves(
            board["snakes"], enemy["id"], width, height, foods, override_my_body=my_body_after, my_id=you["id"]
        )

        before_count = len(before_moves)
        after_count = len(after_moves)
        reduction = max(0, before_count - after_count)

        # Closing distance matters, but should not overpower safety.
        total += max(0, 8 - distance) * 18.0

        # The strongest kill signal: fewer legal moves for a smaller snake.
        total += reduction * 155.0
        if after_count == 0:
            total += 1_000.0
        elif after_count == 1:
            total += 380.0
        elif after_count == 2:
            total += 120.0

        # Occupying a square adjacent to prey head blocks one of its exits and
        # creates a head-to-head threat that we win because we are longer.
        if distance == 1:
            total += 260.0
        elif distance == 2:
            total += 90.0

        # Force prey into small territory.
        if after_moves:
            prey_blocked = _blocked_for_enemy_after_my_move(
                board["snakes"], enemy["id"], foods, override_my_body=my_body_after, my_id=you["id"]
            )
            areas = [
                _flood_fill(move, prey_blocked - {move}, width, height, limit=width * height)
                for move in after_moves
            ]
            best_prey_area = max(areas) if areas else 0
            if best_prey_area < int(enemy["length"]) + 3:
                total += 420.0
            total += max(0, 18 - best_prey_area) * 18.0

        # Steal or block food that a small/hungry enemy likely wants.
        if foods and int(enemy.get("health", 100)) <= LOW_HEALTH:
            target_food = min(foods, key=lambda food: _manhattan(ehead, food))
            enemy_food_dist = _manhattan(ehead, target_food)
            my_food_dist = _manhattan(my_next, target_food)
            if my_food_dist <= enemy_food_dist + 1:
                total += 95.0
            if my_next == target_food:
                total += 160.0

    return total


# ---------------------------------------------------------------------------
# Board / movement helpers


def _emergency_move(game_state: Dict) -> str:
    board = game_state["board"]
    you = game_state["you"]
    width, height = int(board["width"]), int(board["height"])
    head = _head(you)
    occupied = _all_occupied(board["snakes"])
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in occupied:
            return move
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, width, height):
            return move
    return "up"


def _foods(board: Dict) -> List[Point]:
    return [(int(food["x"]), int(food["y"])) for food in board.get("food", [])]


def _enemies(board: Dict, my_id: str) -> List[Dict]:
    return [snake for snake in board.get("snakes", []) if snake.get("id") != my_id]


def _head(snake: Dict) -> Point:
    return (int(snake["head"]["x"]), int(snake["head"]["y"]))


def _body(snake: Dict) -> List[Point]:
    return [(int(part["x"]), int(part["y"])) for part in snake.get("body", [])]


def _tail(snake: Dict) -> Optional[Point]:
    body = _body(snake)
    return body[-1] if body else None


def _all_occupied(snakes: Iterable[Dict]) -> Set[Point]:
    occupied: Set[Point] = set()
    for snake in snakes:
        occupied.update(_body(snake))
    return occupied


def _blocked_for_my_entry(snakes: Sequence[Dict], my_id: str, my_next: Point, foods: Sequence[Point]) -> Set[Point]:
    blocked = _all_occupied(snakes)
    for snake in snakes:
        tail = _tail(snake)
        if tail is None:
            continue
        if snake.get("id") == my_id:
            # If we eat on this move, our tail does not disappear.
            if my_next not in foods:
                blocked.discard(tail)
        else:
            # Conservative enemy tail logic: enemy tail is free only if the
            # enemy probably cannot eat now.
            if not _can_eat_next_turn(snake, foods):
                blocked.discard(tail)
    return blocked


def _blocked_after_my_move(
    snakes: Sequence[Dict], my_id: str, my_body_after: Sequence[Point], foods: Sequence[Point]
) -> Set[Point]:
    blocked: Set[Point] = set(my_body_after)
    for snake in snakes:
        if snake.get("id") == my_id:
            continue
        enemy_body = _body(snake)
        blocked.update(enemy_body)
        # For territory estimation, allow enemy tails to clear when they are
        # unlikely to eat. This avoids being overly afraid of paths that open.
        tail = enemy_body[-1] if enemy_body else None
        if tail is not None and not _can_eat_next_turn(snake, foods):
            blocked.discard(tail)
    return blocked


def _blocked_for_enemy_after_my_move(
    snakes: Sequence[Dict], enemy_id: str, foods: Sequence[Point], *, override_my_body: Sequence[Point], my_id: str
) -> Set[Point]:
    blocked: Set[Point] = set()
    for snake in snakes:
        sid = snake.get("id")
        if sid == my_id:
            blocked.update(override_my_body)
            continue
        body = _body(snake)
        blocked.update(body)
        tail = body[-1] if body else None
        if tail is not None and not _can_eat_next_turn(snake, foods):
            blocked.discard(tail)
    return blocked


def _my_body_after_move(you: Dict, nxt: Point, foods: Sequence[Point]) -> List[Point]:
    body = _body(you)
    if nxt in foods:
        return [nxt] + body
    return [nxt] + body[:-1]


def _can_eat_next_turn(snake: Dict, foods: Sequence[Point]) -> bool:
    if not foods:
        return False
    head = _head(snake)
    return any(_manhattan(head, food) == 1 for food in foods)


def _head_to_head_risk_cells(
    snakes: Sequence[Dict], my_id: str, my_len: int, width: int, height: int, foods: Sequence[Point]
) -> Set[Point]:
    risk: Set[Point] = set()
    for enemy in snakes:
        if enemy.get("id") == my_id or int(enemy["length"]) < my_len:
            continue
        for cell in _enemy_adjacent_cells(enemy, width, height):
            risk.add(cell)
    return risk


def _head_to_head_kill_cells(
    snakes: Sequence[Dict], my_id: str, my_len: int, width: int, height: int, foods: Sequence[Point]
) -> Set[Point]:
    kill: Set[Point] = set()
    for enemy in snakes:
        if enemy.get("id") == my_id or int(enemy["length"]) >= my_len:
            continue
        for cell in _enemy_adjacent_cells(enemy, width, height):
            kill.add(cell)
    return kill


def _enemy_adjacent_cells(enemy: Dict, width: int, height: int) -> Set[Point]:
    head = _head(enemy)
    cells: Set[Point] = set()
    for dx, dy in DIRECTIONS.values():
        cell = (head[0] + dx, head[1] + dy)
        if _in_bounds(cell, width, height):
            cells.add(cell)
    return cells


def _enemy_safe_moves(
    snakes: Sequence[Dict],
    enemy_id: str,
    width: int,
    height: int,
    foods: Sequence[Point],
    *,
    override_my_body: Optional[Sequence[Point]],
    my_id: str,
) -> List[Point]:
    enemy = next(snake for snake in snakes if snake.get("id") == enemy_id)
    ehead = _head(enemy)

    if override_my_body is None:
        blocked = _all_occupied(snakes)
        for snake in snakes:
            tail = _tail(snake)
            if tail is not None and not _can_eat_next_turn(snake, foods):
                blocked.discard(tail)
    else:
        blocked = _blocked_for_enemy_after_my_move(
            snakes, enemy_id, foods, override_my_body=override_my_body, my_id=my_id
        )

    moves: List[Point] = []
    for dx, dy in DIRECTIONS.values():
        nxt = (ehead[0] + dx, ehead[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in blocked:
            moves.append(nxt)
    return moves


def _exit_count(cell: Point, blocked: Set[Point], width: int, height: int) -> int:
    return sum(
        1
        for dx, dy in DIRECTIONS.values()
        if _in_bounds((cell[0] + dx, cell[1] + dy), width, height)
        and (cell[0] + dx, cell[1] + dy) not in blocked
    )


def _second_step_options(cell: Point, blocked: Set[Point], width: int, height: int) -> int:
    """Approximate number of non-fatal second moves after entering cell."""
    count = 0
    for dx, dy in DIRECTIONS.values():
        nxt = (cell[0] + dx, cell[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in blocked:
            # Give extra credit if the second cell itself has a way out.
            count += 1 + min(2, _exit_count(nxt, blocked | {cell}, width, height))
    return count


def _can_reach_own_tail(cell: Point, you: Dict, blocked: Set[Point], width: int, height: int) -> bool:
    tail = _tail(you)
    if tail is None:
        return False
    # Allow the tail cell as target even if it is currently in blocked.
    dist = _bfs_dist([cell], blocked - {tail}, width, height)
    return tail in dist


def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: int) -> int:
    if not _in_bounds(start, width, height):
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
            nbr = (x + dx, y + dy)
            if nbr in seen or nbr in blocked or not _in_bounds(nbr, width, height):
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _bfs_dist(sources: Iterable[Point], blocked: Set[Point], width: int, height: int) -> Dict[Point, int]:
    dist: Dict[Point, int] = {}
    dq: deque[Point] = deque()
    for source in sources:
        if _in_bounds(source, width, height) and source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in DIRECTIONS.values():
            nb = (x + dx, y + dy)
            if not _in_bounds(nb, width, height) or nb in blocked or nb in dist:
                continue
            dist[nb] = d + 1
            dq.append(nb)
    return dist


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
