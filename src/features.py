from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from src.moves import apply_move, DIRECTIONS
from src.state import GameState, Point, SnakeState
from src.simulator import simulate_turn


@dataclass(frozen=True)
class FeatureBuilder:
    state: GameState

    def policy_features(self, snake_id: str, move: str) -> Dict[str, float]:
        snake = self._snake(snake_id)
        next_head = apply_move(snake.head, move)
        food_dist = self._distance_to_nearest_food(next_head)
        open_space = self._flood_fill_count(next_head, snake)
        safe_moves_next = self._count_safe_moves_after_move(snake_id, move)
        return {
            "candidate_move": float(list(DIRECTIONS.keys()).index(move)),
            "open_space_after_move": float(open_space),
            "distance_to_nearest_food": float(food_dist),
            "health_after_move": float(snake.health - 1 if next_head not in self.state.food else self.state.max_health),
            "is_food": float(next_head in self.state.food),
            "wall_distance": float(min(next_head[0], self.state.width - 1 - next_head[0], next_head[1], self.state.height - 1 - next_head[1])),
            "distance_to_center": abs(next_head[0] - (self.state.width - 1) / 2) + abs(next_head[1] - (self.state.height - 1) / 2),
            "enemy_count": float(len([s for s in self.state.snakes if s.id != snake_id and s.alive])),
            "length_advantage": float(snake.length - max((other.length for other in self.state.snakes if other.id != snake_id), default=0)),
            "safe_moves_next_turn": float(safe_moves_next),
        }

    def value_features(self, snake_id: str) -> Dict[str, float]:
        snake = self._snake(snake_id)
        return {
            "health": float(snake.health),
            "length": float(snake.length),
            "length_advantage": float(snake.length - max((other.length for other in self.state.snakes if other.id != snake_id), default=0)),
            "enemy_count": float(len([s for s in self.state.snakes if s.id != snake_id and s.alive])),
            "turn": float(self.state.turn),
            "food_distance": float(self._distance_to_nearest_food(snake.head)),
            "safe_moves": float(self._count_safe_moves(snake_id)),
            "has_immediate_threat": float(self._has_immediate_threat(snake_id)),
        }

    def _snake(self, snake_id: str) -> SnakeState:
        for snake in self.state.snakes:
            if snake.id == snake_id:
                return snake
        raise ValueError(f"Snake {snake_id} not found")

    def _distance_to_nearest_food(self, point: Point) -> int:
        if not self.state.food:
            return self.state.width + self.state.height
        return min(abs(point[0] - f[0]) + abs(point[1] - f[1]) for f in self.state.food)

    def _flood_fill_count(self, start: Point, snake: SnakeState) -> int:
        visited = {start}
        stack = [start]
        limit = self.state.width * self.state.height
        occupied = {segment for s in self.state.snakes for segment in s.body}
        count = 0
        while stack and count < limit:
            current = stack.pop()
            count += 1
            for dx, dy in DIRECTIONS.values():
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in visited or nxt in occupied or not (0 <= nxt[0] < self.state.width and 0 <= nxt[1] < self.state.height):
                    continue
                visited.add(nxt)
                stack.append(nxt)
        return count

    def _count_safe_moves(self, snake_id: str) -> int:
        snake = self._snake(snake_id)
        safe = 0
        occupied = {segment for s in self.state.snakes for segment in s.body}
        for move in DIRECTIONS:
            nxt = apply_move(snake.head, move)
            if not (0 <= nxt[0] < self.state.width and 0 <= nxt[1] < self.state.height):
                continue
            if nxt in occupied:
                continue
            if nxt in self.state.hazards and snake.health <= self.state.hazard_damage:
                continue
            safe += 1
        return safe

    def _count_safe_moves_after_move(self, snake_id: str, move: str) -> int:
        snake = self._snake(snake_id)
        next_head = apply_move(snake.head, move)
        if next_head not in self.state.food:
            next_body = (next_head,) + snake.body[:-1]
        else:
            next_body = (next_head,) + snake.body
        occupied = {segment for s in self.state.snakes if s.id != snake_id for segment in s.body} | set(next_body)
        safe = 0
        for candidate in DIRECTIONS:
            nxt = apply_move(next_head, candidate)
            if not (0 <= nxt[0] < self.state.width and 0 <= nxt[1] < self.state.height):
                continue
            if nxt in occupied:
                continue
            if nxt in self.state.hazards and snake.health - 1 <= self.state.hazard_damage:
                continue
            safe += 1
        return safe

    def _has_immediate_threat(self, snake_id: str) -> bool:
        snake = self._snake(snake_id)
        for move in DIRECTIONS:
            nxt = apply_move(snake.head, move)
            if not (0 <= nxt[0] < self.state.width and 0 <= nxt[1] < self.state.height):
                continue
            if nxt in {segment for s in self.state.snakes for segment in s.body}:
                continue
            return False
        return True
