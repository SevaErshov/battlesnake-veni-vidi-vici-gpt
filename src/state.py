from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}


@dataclass(frozen=True)
class SnakeState:
    id: str
    head: Point
    body: Tuple[Point, ...]
    health: int
    length: int
    alive: bool = True


@dataclass(frozen=True)
class GameState:
    width: int
    height: int
    turn: int
    our_snake_id: str
    snakes: Tuple[SnakeState, ...]
    food: Tuple[Point, ...]
    hazards: Tuple[Point, ...]
    hazard_damage: int = 0
    max_health: int = 100

    @property
    def our_snake(self) -> SnakeState:
        for snake in self.snakes:
            if snake.id == self.our_snake_id:
                return snake
        raise ValueError(f"Our snake {self.our_snake_id} not found")


def _parse_point(raw: Dict[str, int]) -> Point:
    return raw["x"], raw["y"]


def parse_game_state(raw_state: Dict) -> GameState:
    """Parse Battlesnake JSON into a typed GameState."""
    board = raw_state["board"]
    width = int(board["width"])
    height = int(board["height"])
    turn = int(raw_state.get("turn", 0))
    you = raw_state["you"]
    our_id = you["id"]
    food = tuple(_parse_point(item) for item in board.get("food", []))
    hazards = tuple(_parse_point(item) for item in board.get("hazards", []))
    hazard_damage = int(board.get("hazardDamage", 0)) if board.get("hazardDamage") is not None else 0
    max_health = int(board.get("health" , 100)) if board.get("health") is not None else 100
    snakes = []
    for raw_snake in board.get("snakes", []):
        snake_id = raw_snake["id"]
        head = _parse_point(raw_snake["head"])
        body = tuple(_parse_point(seg) for seg in raw_snake.get("body", []))
        health = int(raw_snake.get("health", 0))
        length = int(raw_snake.get("length", len(body)))
        snakes.append(SnakeState(id=snake_id, head=head, body=body, health=health, length=length))
    return GameState(
        width=width,
        height=height,
        turn=turn,
        our_snake_id=our_id,
        snakes=tuple(snakes),
        food=food,
        hazards=hazards,
        hazard_damage=hazard_damage,
        max_health=max_health,
    )
