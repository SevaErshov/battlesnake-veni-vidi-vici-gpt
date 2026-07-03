from __future__ import annotations

from pathlib import Path
from typing import Sequence

from catboost import CatBoostRanker


class PolicyModel:
    def __init__(self, model: CatBoostRanker) -> None:
        self.model = model

    def rank(self, features: list[dict[str, float]]) -> list[float]:
        if not features:
            return []
        return list(self.model.predict(features))

    @classmethod
    def load(cls, path: Path) -> "PolicyModel":
        model = CatBoostRanker()
        model.load_model(str(path))
        return cls(model)

    def save(self, path: Path) -> None:
        self.model.save_model(str(path))
