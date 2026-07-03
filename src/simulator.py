from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from src.moves import apply_move, DIRECTIONS
from src.state import GameState, Point, SnakeState


def simulate_turn(state: GameState, moves: Mapping[str, str]) -> GameState:
    """Simulate one simultaneous turn for all snakes."""
    next_positions: dict[str, Point] = {}
    for snake in state.snakes:
        if not snake.alive:
            continue
        move = moves.get(snake.id, "up")
        if move not in DIRECTIONS:
            raise ValueError(f"Invalid move {move} for snake {snake.id}")
        next_positions[snake.id] = apply_move(snake.head, move)

    survivors: List[SnakeState] = []
    occupied: set[Point] = {pos for pos in next_positions.values()}
    current_occupied: set[Point] = {segment for snake in state.snakes for segment in snake.body}
    food_set = set(state.food)

    eaten: set[Point] = set()
    for snake in state.snakes:
        if not snake.alive:
            continue
        head_next = next_positions[snake.id]
        if head_next in food_set:
            eaten.add(head_next)

    winner_lengths: dict[Point, int] = {}
    for snake in state.snakes:
        if not snake.alive:
            continue
        head_next = next_positions[snake.id]
        winner_lengths.setdefault(head_next, 0)
        winner_lengths[head_next] = max(winner_lengths[head_next], snake.length)

    deaths: set[str] = set()
    for snake in state.snakes:
        if not snake.alive:
            continue
        head_next = next_positions[snake.id]
        if not _in_bounds(head_next, state.width, state.height):
            deaths.add(snake.id)
            continue
        if head_next in state.hazards and snake.health <= state.hazard_damage:
            deaths.add(snake.id)
            continue
        if head_next in current_occupied and head_next != snake.body[-1]:
            deaths.add(snake.id)
            continue

    # Head-to-head collision rules.
    for position, length in winner_lengths.items():
        contenders = [snake for snake in state.snakes if next_positions.get(snake.id) == position and snake.alive]
        if len(contenders) < 2:
            continue
        max_len = max(snake.length for snake in contenders)
        winners = [snake for snake in contenders if snake.length == max_len]
        if len(winners) == 1:
            for snake in contenders:
                if snake.id != winners[0].id:
                    deaths.add(snake.id)
        else:
            for snake in contenders:
                deaths.add(snake.id)

    next_snakes: List[SnakeState] = []

    for snake in state.snakes:
        if not snake.alive or snake.id in deaths:
            continue
        head_next = next_positions[snake.id]
        ate = head_next in eaten
        health = snake.health - 1
        if ate:
            health = state.max_health
        if head_next in state.hazards:
            health -= state.hazard_damage
        body = (head_next,) + snake.body
        if not ate:
            body = body[:-1]
        next_snakes.append(
            SnakeState(
                id=snake.id,
                head=head_next,
                body=body,
                health=health,
                length=len(body),
                alive=True,
            )
        )

    next_food = tuple(pt for pt in state.food if pt not in eaten)
    return GameState(
        width=state.width,
        height=state.height,
        turn=state.turn + 1,
        our_snake_id=state.our_snake_id,
        snakes=tuple(next_snakes),
        food=next_food,
        hazards=state.hazards,
        hazard_damage=state.hazard_damage,
        max_health=state.max_health,
    )


def _in_bounds(point: Point, width: int, height: int) -> bool:
    return 0 <= point[0] < width and 0 <= point[1] < height
