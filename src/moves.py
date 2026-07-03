from __future__ import annotations

from typing import Dict, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

REVERSE: Dict[str, str] = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def apply_move(point: Point, move: str) -> Point:
    dx, dy = DIRECTIONS[move]
    return point[0] + dx, point[1] + dy


def all_moves() -> tuple[str, ...]:
    return tuple(DIRECTIONS.keys())
