"""Mega-hard ML Ensemble Battlesnake logic.

Runtime design:
- Safety shield is always first. The ML models can rank only moves that are not
  immediately fatal.
- Five exported lightweight models score candidate moves in pure Python:
  ridge_regression, logistic_best, linear_svm, perceptron, tiny_mlp.
- BATTLE_MODEL controls the model used at runtime:
  ensemble | ridge | logistic | svm | perceptron | mlp | expert
- Multiple fallbacks are included so /move never crashes.

The models were trained offline to imitate a stronger expert heuristic generated
from synthetic Battlesnake positions. Re-train with training/train_models.py.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from model_weights import MODEL_WEIGHTS
except Exception:  # noqa: BLE001 - safe in training/dev if weights are absent
    MODEL_WEIGHTS = {}

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
MOVE_ORDER = ["up", "down", "left", "right"]

FEATURE_NAMES = [
    "health_norm",
    "my_len_norm",
    "enemy_count_norm",
    "len_advantage_norm",
    "is_health_crisis",
    "is_hungry",
    "is_endgame",
    "is_predator",
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    "candidate_is_food",
    "dist_food_norm",
    "food_race_adv_norm",
    "open_area_norm",
    "open_area_vs_len",
    "exit_count_norm",
    "second_step_norm",
    "reaches_tail",
    "tail_dist_norm",
    "h2h_risk",
    "h2h_kill",
    "near_big_head_inv",
    "near_small_head_inv",
    "wall_dist_norm",
    "center_closeness",
    "trap_flag",
    "thin_corridor",
    "prey_dist_norm",
    "prey_options_reduced_norm",
    "hazard_flag",
]

FATAL = -1_000_000.0
SAFETY_DOMINANCE = 100_000.0
DEFAULT_TIMEOUT_MS = 500
DEBUG = os.getenv("BATTLESNAKE_DEBUG", "0") == "1"
RUNTIME_MODEL = os.getenv("BATTLE_MODEL", "expert").lower()


@dataclass(frozen=True)
class Candidate:
    move: str
    point: Point
    features: List[float]
    safety_score: float
    expert_score: float
    model_score: float = 0.0
    final_score: float = 0.0
    notes: str = ""


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from GET /."""
    model_name = RUNTIME_MODEL if RUNTIME_MODEL in _available_models() else "ensemble"
    return {
        "apiversion": "1",
        "author": "veni-vidi-vici-gpt",
        "color": "#111827",
        "head": "evil",
        "tail": "fat-rattle",
        "version": f"2.0.0-hardml-{model_name}",
    }


def choose_move(game_state: Dict) -> str:
    """Return next Battlesnake move. This function must never raise."""
    started = time.perf_counter()
    try:
        ranked = rank_moves(game_state, model_name=RUNTIME_MODEL)
        if ranked:
            elapsed_ms = (time.perf_counter() - started) * 1000
            if DEBUG:
                print(
                    f"model={RUNTIME_MODEL} elapsed={elapsed_ms:.2f}ms | "
                    + " | ".join(
                        f"{c.move}:final={c.final_score:.1f}:ml={c.model_score:.2f}:safe={c.safety_score:.1f}:{c.notes}"
                        for c in ranked
                    )
                )
            return ranked[0].move
    except Exception as exc:  # noqa: BLE001 - timeout/error must not kill us
        if DEBUG:
            print(f"choose_move failed: {exc!r}")

    return fallback_move(game_state)


def rank_moves(game_state: Dict, model_name: str = "ensemble") -> List[Candidate]:
    candidates = build_candidates(game_state)
    if not candidates:
        return []

    selected_model = model_name if model_name in _available_models() else "ensemble"
    scored: List[Candidate] = []
    for c in candidates:
        ml = score_features(c.features, selected_model)
        # Safety remains dominant. The ML/expert part separates safe alternatives.
        if selected_model == "expert":
            final = c.safety_score + c.expert_score
        else:
            # Blend exported ML with a small expert prior to reduce synthetic-label weirdness.
            final = c.safety_score + 120.0 * ml + 0.18 * c.expert_score
        scored.append(
            Candidate(
                move=c.move,
                point=c.point,
                features=c.features,
                safety_score=c.safety_score,
                expert_score=c.expert_score,
                model_score=ml,
                final_score=final,
                notes=c.notes,
            )
        )
    scored.sort(key=lambda item: item.final_score, reverse=True)
    return scored


def compare_models(game_state: Dict) -> Dict[str, List[Dict[str, float | str]]]:
    """Debug helper used by smoke_test.py; not called by Battlesnake engine."""
    output: Dict[str, List[Dict[str, float | str]]] = {}
    for model in ["expert", "ridge", "logistic", "svm", "perceptron", "mlp", "ensemble"]:
        ranked = rank_moves(game_state, model)
        output[model] = [
            {
                "move": c.move,
                "final": round(c.final_score, 3),
                "ml": round(c.model_score, 3),
                "safety": round(c.safety_score, 3),
                "notes": c.notes,
            }
            for c in ranked
        ]
    return output


def build_candidates(game_state: Dict) -> List[Candidate]:
    board = game_state.get("board", {})
    you = game_state.get("you", {})
    width = int(board.get("width", 11))
    height = int(board.get("height", 11))
    head = _head(you)
    food = _foods(board)
    hazards = set(_points(board.get("hazards", [])))
    snakes = board.get("snakes", [])
    enemies = _enemies(board, you.get("id"))
    my_len = int(you.get("length", len(you.get("body", [])) or 1))
    health = int(you.get("health", 100))

    out: List[Candidate] = []
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        candidate = _candidate_for_move(
            game_state=game_state,
            move=move,
            nxt=nxt,
            width=width,
            height=height,
            food=food,
            hazards=hazards,
            snakes=snakes,
            enemies=enemies,
            my_len=my_len,
            health=health,
        )
        if candidate is not None:
            out.append(candidate)
    # If all legal candidates are h2h-risky, keep them; otherwise remove the worst direct risks.
    safer = [c for c in out if "h2h-risk" not in c.notes]
    return safer if safer else out


def _candidate_for_move(
    *,
    game_state: Dict,
    move: str,
    nxt: Point,
    width: int,
    height: int,
    food: Sequence[Point],
    hazards: Set[Point],
    snakes: Sequence[Dict],
    enemies: Sequence[Dict],
    my_len: int,
    health: int,
) -> Optional[Candidate]:
    board = game_state.get("board", {})
    you = game_state.get("you", {})
    if not _in_bounds(nxt, width, height):
        return None

    blocked_entry = _blocked_for_entry(snakes, you.get("id"), nxt, food)
    if nxt in blocked_entry:
        return None

    my_body_after = _my_body_after_move(you, nxt, food)
    blocked_after = _blocked_after_my_move(snakes, you.get("id"), my_body_after, food)
    blocked_without_head = set(blocked_after)
    blocked_without_head.discard(nxt)

    max_area = width * height
    open_area = _flood_fill(nxt, blocked_without_head, width, height, limit=max_area)
    exits = _exit_count(nxt, blocked_without_head, width, height)
    second_options = _second_step_options(nxt, blocked_without_head, width, height)
    reaches_tail = _can_reach_tail(nxt, you, blocked_without_head, width, height)
    tail_dist = _dist_to_tail(nxt, you, blocked_without_head, width, height)
    wall_dist = min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
    center_dist = abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2)
    board_diag = max(1, width + height - 2)

    risk_cells = _head_to_head_risk_cells(snakes, you.get("id"), my_len, width, height, food)
    kill_cells = _head_to_head_kill_cells(snakes, you.get("id"), my_len, width, height, food)
    h2h_risk = 1.0 if nxt in risk_cells else 0.0
    h2h_kill = 1.0 if nxt in kill_cells else 0.0

    nearest_big_head = min(
        (_manhattan(nxt, _head(enemy)) for enemy in enemies if int(enemy.get("length", 0)) >= my_len),
        default=width + height,
    )
    nearest_small_head = min(
        (_manhattan(nxt, _head(enemy)) for enemy in enemies if int(enemy.get("length", 0)) < my_len),
        default=width + height,
    )
    prey_options_before, prey_options_after, prey_dist = _prey_pressure_stats(
        enemies=enemies,
        my_len=my_len,
        my_next=nxt,
        blocked_after=blocked_after,
        width=width,
        height=height,
        food=food,
    )
    prey_reduced = max(0, prey_options_before - prey_options_after)

    dist_food = min((_manhattan(nxt, f) for f in food), default=width + height)
    food_race_adv = _food_race_advantage(nxt, food, enemies)
    len_advantage = my_len - max((int(e.get("length", 0)) for e in enemies), default=0)

    trap_flag = 1.0 if open_area < max(my_len + 2, int(my_len * 1.2)) else 0.0
    thin_corridor = 1.0 if exits <= 1 and open_area < my_len + 6 else 0.0
    hazard_flag = 1.0 if nxt in hazards else 0.0
    candidate_is_food = 1.0 if nxt in food else 0.0

    # Direction one-hot.
    move_oh = [1.0 if move == m else 0.0 for m in MOVE_ORDER]
    features = [
        _clip01(health / 100.0),
        _clip01(my_len / 25.0),
        _clip01(len(enemies) / 7.0),
        _clip(len_advantage / 15.0, -1.0, 1.0),
        1.0 if health <= 25 else 0.0,
        1.0 if health <= 45 else 0.0,
        1.0 if len(enemies) <= 1 else 0.0,
        1.0 if any(int(e.get("length", 0)) < my_len for e in enemies) else 0.0,
        *move_oh,
        candidate_is_food,
        _clip01(dist_food / board_diag),
        _clip(food_race_adv / 8.0, -1.0, 1.0),
        _clip01(open_area / max_area),
        _clip(open_area / max(1, my_len + 8), 0.0, 2.0),
        _clip01(exits / 4.0),
        _clip01(second_options / 4.0),
        1.0 if reaches_tail else 0.0,
        _clip01(tail_dist / board_diag),
        h2h_risk,
        h2h_kill,
        1.0 / (1.0 + nearest_big_head),
        1.0 / (1.0 + nearest_small_head),
        _clip01(wall_dist / max(1, min(width, height) / 2)),
        1.0 - _clip01(center_dist / board_diag),
        trap_flag,
        thin_corridor,
        _clip01(prey_dist / board_diag),
        _clip01(prey_reduced / 4.0),
        hazard_flag,
    ]

    safety = 0.0
    notes: List[str] = []
    if h2h_risk:
        safety -= 0.65 * SAFETY_DOMINANCE
        notes.append("h2h-risk")
    if hazard_flag:
        # Only enter hazards in emergency; big negative, but not impossible.
        safety -= 900.0
        notes.append("hazard")
    if trap_flag:
        safety -= 1500.0 + (max(my_len + 2, int(my_len * 1.2)) - open_area) * 100.0
        notes.append(f"trap={open_area}")
    if thin_corridor:
        safety -= 650.0
        notes.append("thin")
    if exits == 0:
        safety -= 0.5 * SAFETY_DOMINANCE
        notes.append("no-exit")
    if reaches_tail:
        notes.append("tail")
    if candidate_is_food:
        notes.append("food")
    if h2h_kill:
        notes.append("kill-cell")

    expert = expert_score_from_features(features)
    return Candidate(
        move=move,
        point=nxt,
        features=features,
        safety_score=safety,
        expert_score=expert,
        notes=",".join(notes),
    )


def expert_score_from_features(x: Sequence[float]) -> float:
    """Strong heuristic used for labels and as a runtime fallback prior."""
    f = dict(zip(FEATURE_NAMES, x))
    score = 0.0
    # Survival and space.
    score += 900.0 * f["open_area_norm"]
    score += 300.0 * f["open_area_vs_len"]
    score += 220.0 * f["exit_count_norm"]
    score += 180.0 * f["second_step_norm"]
    score += 150.0 * f["reaches_tail"]
    score += 90.0 * f["wall_dist_norm"]
    score += 70.0 * f["center_closeness"]
    # Food: urgent only when health is low, but punish dangerous food via trap features.
    hunger = 1.0 - f["health_norm"]
    score += 520.0 * hunger * (1.0 - f["dist_food_norm"])
    score += 220.0 * f["candidate_is_food"] * (0.5 + hunger)
    score += 130.0 * f["food_race_adv_norm"] * max(0.2, hunger)
    # Predator: useful if we are longer and not starving.
    predator_weight = f["is_predator"] * max(0.0, f["health_norm"] - 0.25)
    score += predator_weight * 300.0 * (1.0 - f["prey_dist_norm"])
    score += predator_weight * 260.0 * f["prey_options_reduced_norm"]
    score += predator_weight * 260.0 * f["h2h_kill"]
    # Danger dominates.
    score -= 2500.0 * f["h2h_risk"]
    score -= 1300.0 * f["trap_flag"]
    score -= 550.0 * f["thin_corridor"]
    score -= 550.0 * f["hazard_flag"]
    score -= 450.0 * f["near_big_head_inv"]
    return score


def score_features(features: Sequence[float], model_name: str = "ensemble") -> float:
    if not MODEL_WEIGHTS:
        return expert_score_from_features(features) / 1000.0
    model_name = model_name.lower()
    if model_name in {"ridge_regression", "ridge"}:
        return _linear_score(features, MODEL_WEIGHTS["ridge"])
    if model_name in {"logistic_best", "logistic", "logit"}:
        z = _linear_score(features, MODEL_WEIGHTS["logistic"], raw=True)
        return _sigmoid(z) * 2.0 - 1.0
    if model_name in {"linear_svm", "svm"}:
        return _linear_score(features, MODEL_WEIGHTS["svm"])
    if model_name == "perceptron":
        return _linear_score(features, MODEL_WEIGHTS["perceptron"])
    if model_name in {"tiny_mlp", "mlp"}:
        return _mlp_score(features, MODEL_WEIGHTS["mlp"])
    if model_name == "expert":
        return expert_score_from_features(features) / 1000.0
    # Ensemble: average robust normalized model outputs.
    parts = [
        (0.08, _linear_score(features, MODEL_WEIGHTS["ridge"])),
        (0.22, _sigmoid(_linear_score(features, MODEL_WEIGHTS["logistic"], raw=True)) * 2.0 - 1.0),
        (0.32, _linear_score(features, MODEL_WEIGHTS["svm"])),
        (0.08, _linear_score(features, MODEL_WEIGHTS["perceptron"])),
        (0.30, _mlp_score(features, MODEL_WEIGHTS["mlp"])),
    ]
    return sum(w * _clip(v, -5.0, 5.0) for w, v in parts)


def _linear_score(features: Sequence[float], weights: Dict, raw: bool = False) -> float:
    mean = weights.get("mean", [0.0] * len(features))
    scale = weights.get("scale", [1.0] * len(features))
    coef = weights["coef"]
    intercept = float(weights.get("intercept", 0.0))
    z = intercept
    for val, mu, sd, w in zip(features, mean, scale, coef):
        if sd == 0:
            sd = 1.0
        z += ((float(val) - float(mu)) / float(sd)) * float(w)
    return z if raw else _clip(z, -10.0, 10.0)


def _mlp_score(features: Sequence[float], weights: Dict) -> float:
    mean = weights.get("mean", [0.0] * len(features))
    scale = weights.get("scale", [1.0] * len(features))
    x = [((float(v) - float(m)) / (float(s) or 1.0)) for v, m, s in zip(features, mean, scale)]
    hidden: List[float] = []
    for row, bias in zip(weights["hidden_weights"], weights["hidden_bias"]):
        z = float(bias) + sum(float(w) * xi for w, xi in zip(row, x))
        hidden.append(math.tanh(z))
    out = float(weights.get("out_bias", 0.0)) + sum(float(w) * h for w, h in zip(weights["out_weights"], hidden))
    return _clip(out, -10.0, 10.0)


def fallback_move(game_state: Dict) -> str:
    """Four-layer backup: expert -> tail path -> legal -> deterministic default."""
    try:
        ranked = rank_moves(game_state, "expert")
        if ranked:
            return ranked[0].move
    except Exception:
        pass
    move = _tail_chase_fallback(game_state)
    if move:
        return move
    move = _basic_legal_fallback(game_state)
    if move:
        return move
    return "up"


def _tail_chase_fallback(game_state: Dict) -> Optional[str]:
    board = game_state.get("board", {})
    you = game_state.get("you", {})
    width, height = int(board.get("width", 11)), int(board.get("height", 11))
    body = _body(you)
    if not body:
        return None
    head, tail = body[0], body[-1]
    blocked = _blocked_for_entry(board.get("snakes", []), you.get("id"), tail, _foods(board))
    for move in _shortest_first_moves(head, tail, blocked, width, height):
        return move
    return None


def _basic_legal_fallback(game_state: Dict) -> Optional[str]:
    board = game_state.get("board", {})
    you = game_state.get("you", {})
    width, height = int(board.get("width", 11)), int(board.get("height", 11))
    head = _head(you)
    blocked = _blocked_for_entry(board.get("snakes", []), you.get("id"), head, _foods(board))
    best_move, best_area = None, -1
    for move, d in DIRECTIONS.items():
        nxt = (head[0] + d[0], head[1] + d[1])
        if _in_bounds(nxt, width, height) and nxt not in blocked:
            area = _flood_fill(nxt, blocked, width, height, limit=width * height)
            if area > best_area:
                best_move, best_area = move, area
    return best_move


# ----------------------------- board helpers -----------------------------


def _available_models() -> Set[str]:
    return {"expert", "ridge", "logistic", "svm", "perceptron", "mlp", "ensemble"}


def _head(snake: Dict) -> Point:
    head = snake.get("head") or (snake.get("body") or [{"x": 0, "y": 0}])[0]
    return int(head["x"]), int(head["y"])


def _body(snake: Dict) -> List[Point]:
    return _points(snake.get("body", []))


def _points(items: Iterable[Dict]) -> List[Point]:
    return [(int(p["x"]), int(p["y"])) for p in items]


def _foods(board: Dict) -> List[Point]:
    return _points(board.get("food", []))


def _enemies(board: Dict, my_id: Optional[str]) -> List[Dict]:
    return [s for s in board.get("snakes", []) if s.get("id") != my_id]


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _neighbors(p: Point) -> Iterable[Point]:
    x, y = p
    for dx, dy in DIRECTIONS.values():
        yield x + dx, y + dy


def _blocked_for_entry(snakes: Sequence[Dict], my_id: Optional[str], candidate: Point, food: Sequence[Point]) -> Set[Point]:
    """Cells that are unsafe to enter now. Tail-aware for tails likely to move."""
    food_set = set(food)
    blocked: Set[Point] = set()
    for snake in snakes:
        body = _body(snake)
        if not body:
            continue
        # Every body except tail is occupied. Tail may free if that snake is unlikely to eat.
        blocked.update(body[:-1])
        tail = body[-1]
        # Keep own tail blocked only if candidate eats; otherwise it frees.
        if snake.get("id") == my_id:
            if candidate in food_set:
                blocked.add(tail)
        else:
            # Enemy tail is blocked if enemy head can eat next move, otherwise it likely frees.
            enemy_head = body[0]
            enemy_can_eat = any(_manhattan(enemy_head, f) == 1 for f in food_set)
            if enemy_can_eat:
                blocked.add(tail)
    return blocked


def _blocked_after_my_move(
    snakes: Sequence[Dict], my_id: Optional[str], my_body_after: Sequence[Point], food: Sequence[Point]
) -> Set[Point]:
    blocked: Set[Point] = set(my_body_after)
    food_set = set(food)
    for snake in snakes:
        if snake.get("id") == my_id:
            continue
        body = _body(snake)
        if not body:
            continue
        blocked.update(body[:-1])
        enemy_can_eat = any(_manhattan(body[0], f) == 1 for f in food_set)
        if enemy_can_eat:
            blocked.add(body[-1])
    return blocked


def _my_body_after_move(you: Dict, nxt: Point, food: Sequence[Point]) -> List[Point]:
    body = _body(you)
    if not body:
        return [nxt]
    after = [nxt] + body
    if nxt not in set(food):
        after = after[:-1]
    return after


def _flood_fill(start: Point, blocked: Set[Point], width: int, height: int, limit: int) -> int:
    if start in blocked or not _in_bounds(start, width, height):
        return 0
    q = deque([start])
    seen = {start}
    while q and len(seen) < limit:
        p = q.popleft()
        for nb in _neighbors(p):
            if nb in seen or nb in blocked or not _in_bounds(nb, width, height):
                continue
            seen.add(nb)
            q.append(nb)
    return len(seen)


def _exit_count(p: Point, blocked: Set[Point], width: int, height: int) -> int:
    return sum(1 for nb in _neighbors(p) if _in_bounds(nb, width, height) and nb not in blocked)


def _second_step_options(p: Point, blocked: Set[Point], width: int, height: int) -> int:
    options = 0
    for nb in _neighbors(p):
        if not _in_bounds(nb, width, height) or nb in blocked:
            continue
        if _exit_count(nb, blocked | {p}, width, height) > 0:
            options += 1
    return options


def _can_reach_tail(start: Point, you: Dict, blocked: Set[Point], width: int, height: int) -> bool:
    body = _body(you)
    if not body:
        return False
    tail = body[-1]
    blocked2 = set(blocked)
    blocked2.discard(tail)
    return _shortest_distance(start, tail, blocked2, width, height) is not None


def _dist_to_tail(start: Point, you: Dict, blocked: Set[Point], width: int, height: int) -> int:
    body = _body(you)
    if not body:
        return width + height
    tail = body[-1]
    blocked2 = set(blocked)
    blocked2.discard(tail)
    dist = _shortest_distance(start, tail, blocked2, width, height)
    return dist if dist is not None else width + height


def _shortest_distance(start: Point, goal: Point, blocked: Set[Point], width: int, height: int) -> Optional[int]:
    if start == goal:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        p, dist = q.popleft()
        for nb in _neighbors(p):
            if nb == goal:
                return dist + 1
            if nb in seen or nb in blocked or not _in_bounds(nb, width, height):
                continue
            seen.add(nb)
            q.append((nb, dist + 1))
    return None


def _shortest_first_moves(start: Point, goal: Point, blocked: Set[Point], width: int, height: int) -> List[str]:
    moves: List[Tuple[int, str]] = []
    for move, d in DIRECTIONS.items():
        nxt = (start[0] + d[0], start[1] + d[1])
        if not _in_bounds(nxt, width, height) or nxt in blocked:
            continue
        dist = _shortest_distance(nxt, goal, blocked, width, height)
        if dist is not None:
            moves.append((dist, move))
    moves.sort()
    return [m for _, m in moves]


def _head_to_head_risk_cells(
    snakes: Sequence[Dict], my_id: Optional[str], my_len: int, width: int, height: int, food: Sequence[Point]
) -> Set[Point]:
    cells: Set[Point] = set()
    occupied = _all_bodies(snakes)
    for snake in snakes:
        if snake.get("id") == my_id or int(snake.get("length", 0)) < my_len:
            continue
        h = _head(snake)
        for nb in _neighbors(h):
            if _in_bounds(nb, width, height) and (nb not in occupied or nb in set(food)):
                cells.add(nb)
    return cells


def _head_to_head_kill_cells(
    snakes: Sequence[Dict], my_id: Optional[str], my_len: int, width: int, height: int, food: Sequence[Point]
) -> Set[Point]:
    cells: Set[Point] = set()
    occupied = _all_bodies(snakes)
    for snake in snakes:
        if snake.get("id") == my_id or int(snake.get("length", 0)) >= my_len:
            continue
        h = _head(snake)
        for nb in _neighbors(h):
            if _in_bounds(nb, width, height) and (nb not in occupied or nb in set(food)):
                cells.add(nb)
    return cells


def _all_bodies(snakes: Sequence[Dict]) -> Set[Point]:
    out: Set[Point] = set()
    for s in snakes:
        out.update(_body(s))
    return out


def _food_race_advantage(nxt: Point, food: Sequence[Point], enemies: Sequence[Dict]) -> float:
    if not food:
        return 0.0
    best = 0.0
    for f in food:
        my_dist = _manhattan(nxt, f)
        enemy_dist = min((_manhattan(_head(e), f) for e in enemies), default=my_dist + 4)
        adv = enemy_dist - my_dist
        # favor close food that we can win or tie when we are hungry
        best = max(best, adv / max(1, my_dist))
    return best


def _prey_pressure_stats(
    *,
    enemies: Sequence[Dict],
    my_len: int,
    my_next: Point,
    blocked_after: Set[Point],
    width: int,
    height: int,
    food: Sequence[Point],
) -> Tuple[int, int, int]:
    prey = [e for e in enemies if int(e.get("length", 0)) < my_len]
    if not prey:
        return 0, 0, width + height
    target = min(prey, key=lambda e: _manhattan(my_next, _head(e)))
    h = _head(target)
    before_blocked = set(blocked_after)
    before_blocked.discard(my_next)
    before = _exit_count(h, before_blocked, width, height)
    after_blocked = set(before_blocked)
    after_blocked.add(my_next)
    after = _exit_count(h, after_blocked, width, height)
    return before, after, _manhattan(my_next, h)


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _clip01(v: float) -> float:
    return _clip(v, 0.0, 1.0)


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)
