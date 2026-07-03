from __future__ import annotations

import argparse
import json
import requests
from pathlib import Path


def main(url: str, timeout: int) -> None:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    assert "apiversion" in data
    assert "author" in data
    print("GET / OK", data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()
    main(args.url, args.timeout)
