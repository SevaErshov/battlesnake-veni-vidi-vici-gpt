from __future__ import annotations

import time


class Deadline:
    def __init__(self, budget_ms: int = 250) -> None:
        self.deadline = time.monotonic() + budget_ms / 1000.0

    def expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def remaining_ms(self) -> int:
        return max(0, int((self.deadline - time.monotonic()) * 1000))
