from __future__ import annotations

import argparse
from pathlib import Path


def main(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Dataset building pipeline is not implemented yet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/dataset.csv"))
    args = parser.parse_args()
    main(args.output)
