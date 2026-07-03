from __future__ import annotations

from pathlib import Path

from training.build_datasets import main as build_datasets
from training.train_policy import main as train_policy
from training.train_value import main as train_value


def main() -> None:
    build_datasets(Path("data/dataset.csv"))
    train_policy(Path("artifacts/policy_model.cbm"))
    train_value(Path("artifacts/value_model.cbm"))


if __name__ == "__main__":
    main()
