from __future__ import annotations

from typing import Mapping

from src.moves import DIRECTIONS, apply_move
from src.state import GameState, Point, SnakeState


class MissingMoveError(ValueError):
    pass


def simulate_turn(state: GameState, moves: Mapping[str, str]) -> GameState:
    """Simulate one full simultaneous Battlesnake-like turn."""
    alive_snakes = [snake for snake in state.snakes if snake.alive]
    next_heads: dict[str, Point] = {}
    for snake in alive_snakes:
        if snake.id not in moves:
            raise MissingMoveError(f"Missing move for snake {snake.id}")
        move = moves[snake.id]
        if move not in DIRECTIONS:
            raise ValueError(f"Invalid move {move!r} for snake {snake.id}")
        next_heads[snake.id] = apply_move(snake.head, move)

    food_set = set(state.food)
    ate_food = {snake.id: next_heads[snake.id] in food_set for snake in alive_snakes}
    deaths: set[str] = set()

    next_health: dict[str, int] = {}
    for snake in alive_snakes:
        head = next_heads[snake.id]
        health = snake.health - 1
        if head in state.hazards:
            health -= state.hazard_damage
        if ate_food[snake.id]:
            health = state.max_health
        next_health[snake.id] = health
        if not _in_bounds(head, state.width, state.height):
            deaths.add(snake.id)
        elif health <= 0:
            deaths.add(snake.id)

    body_occupied = _occupied_body_cells_after_tail_release(alive_snakes, ate_food)
    for snake in alive_snakes:
        if snake.id in deaths:
            continue
        if next_heads[snake.id] in body_occupied:
            deaths.add(snake.id)

    heads_by_cell: dict[Point, list[SnakeState]] = {}
    for snake in alive_snakes:
        if snake.id in deaths:
            continue
        heads_by_cell.setdefault(next_heads[snake.id], []).append(snake)

    for contenders in heads_by_cell.values():
        if len(contenders) < 2:
            continue
        max_length = max(snake.length for snake in contenders)
        winners = [snake for snake in contenders if snake.length == max_length]
        if len(winners) == 1:
            deaths.update(snake.id for snake in contenders if snake.id != winners[0].id)
        else:
            deaths.update(snake.id for snake in contenders)

    next_snakes: list[SnakeState] = []
    for snake in alive_snakes:
        if snake.id in deaths:
            continue
        head = next_heads[snake.id]
        body = (head,) + snake.body
        if not ate_food[snake.id]:
            body = body[:-1]
        next_snakes.append(
            SnakeState(
                id=snake.id,
                head=head,
                body=body,
                health=next_health[snake.id],
                length=len(body),
                alive=True,
            )
        )

    eaten_food = {next_heads[snake.id] for snake in alive_snakes if ate_food[snake.id]}
    return GameState(
        width=state.width,
        height=state.height,
        turn=state.turn + 1,
        our_snake_id=state.our_snake_id,
        snakes=tuple(next_snakes),
        food=tuple(point for point in state.food if point not in eaten_food),
        hazards=state.hazards,
        hazard_damage=state.hazard_damage,
        max_health=state.max_health,
    )


def _occupied_body_cells_after_tail_release(
    snakes: list[SnakeState],
    ate_food: dict[str, bool],
) -> set[Point]:
    occupied: set[Point] = set()
    for snake in snakes:
        body = snake.body if ate_food[snake.id] else snake.body[:-1]
        occupied.update(body)
    return occupied


def _in_bounds(point: Point, width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height
