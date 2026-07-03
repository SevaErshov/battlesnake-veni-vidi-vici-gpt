from __future__ import annotations

from pathlib import Path

from catboost import CatBoostRegressor


class ValueModel:
    def __init__(self, model: CatBoostRegressor) -> None:
        self.model = model

    def predict(self, features: list[dict[str, float]]) -> list[float]:
        if not features:
            return []
        return list(self.model.predict(features))

    @classmethod
    def load(cls, path: Path) -> "ValueModel":
        model = CatBoostRegressor()
        model.load_model(str(path))
        return cls(model)

    def save(self, path: Path) -> None:
        self.model.save_model(str(path))
