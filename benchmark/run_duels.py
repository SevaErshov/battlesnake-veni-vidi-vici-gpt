from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.baseline import choose_move as baseline_choose_move
from src.game_runner import BotSpec, create_initial_state, run_game, write_trajectory_jsonl
from src.strategy import choose_move_search_heuristic


def run_duels(
    output: Path,
    games: int,
    seed: int,
    swap_positions: bool,
    max_turns: int,
    trajectory_output: Path | None = None,
) -> dict:
    candidate = BotSpec(name="candidate", choose_move=choose_move_search_heuristic, fallback_move=baseline_choose_move)
    baseline = BotSpec(name="baseline", choose_move=baseline_choose_move, fallback_move=baseline_choose_move)
    trajectory_path = trajectory_output or output.with_suffix(".jsonl")
    if trajectory_path.exists():
        trajectory_path.unlink()

    results = []
    current_seed = seed
    while len(results) < games:
        state = create_initial_state(current_seed, candidate.name, baseline.name, swap_positions=False)
        result = run_game(candidate, baseline, state, seed=current_seed, max_turns=max_turns)
        write_trajectory_jsonl(result, trajectory_path)
        results.append((current_seed, "normal", result))
        if swap_positions and len(results) < games:
            swapped = create_initial_state(current_seed, candidate.name, baseline.name, swap_positions=True)
            swapped_result = run_game(candidate, baseline, swapped, seed=current_seed, max_turns=max_turns)
            write_trajectory_jsonl(swapped_result, trajectory_path)
            results.append((current_seed, "swapped", swapped_result))
        current_seed += 1

    summary = _summarize(results, candidate.name, baseline.name, games, seed, swap_positions, max_turns, trajectory_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(summary)
    return summary


def _summarize(
    results: list[tuple[int, str, object]],
    candidate_name: str,
    baseline_name: str,
    games: int,
    seed: int,
    swap_positions: bool,
    max_turns: int,
    trajectory_path: Path,
) -> dict:
    wins = sum(1 for _, _, result in results if result.winner_id == candidate_name)
    losses = sum(1 for _, _, result in results if result.winner_id == baseline_name)
    draws = len(results) - wins - losses
    turns = [result.turns for _, _, result in results]
    candidate_latencies = [
        latency
        for _, _, result in results
        for latency in result.latencies_ms.get(candidate_name, [])
    ]
    fallback_count = sum(result.fallback_counts.get(candidate_name, 0) for _, _, result in results)
    invalid_moves = sum(result.invalid_moves.get(candidate_name, 0) for _, _, result in results)
    timeouts = sum(result.timeouts.get(candidate_name, 0) for _, _, result in results)

    return {
        "candidate_name": candidate_name,
        "baseline_name": baseline_name,
        "games": len(results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(results) if results else 0.0,
        "average_turns": statistics.fmean(turns) if turns else 0.0,
        "median_turns": statistics.median(turns) if turns else 0.0,
        "median_latency_ms": statistics.median(candidate_latencies) if candidate_latencies else 0.0,
        "p95_latency_ms": _percentile(candidate_latencies, 95),
        "max_latency_ms": max(candidate_latencies) if candidate_latencies else 0.0,
        "timeouts": timeouts,
        "invalid_moves": invalid_moves,
        "fallback_count": fallback_count,
        "fallback_counts_by_bot": _sum_dicts(result.fallback_counts for _, _, result in results),
        "invalid_moves_by_bot": _sum_dicts(result.invalid_moves for _, _, result in results),
        "timeouts_by_bot": _sum_dicts(result.timeouts for _, _, result in results),
        "results_by_seed": [
            {
                "seed": seed_value,
                "position": position,
                "game_id": result.game_id,
                "winner_id": result.winner_id,
                "result": result.result,
                "turns": result.turns,
                "death_reasons": result.death_reasons,
            }
            for seed_value, position, result in results
        ],
        "configuration": {
            "requested_games": games,
            "initial_seed": seed,
            "swap_positions": swap_positions,
            "max_turns": max_turns,
            "trajectory_output": str(trajectory_path),
            "local_runner": "simplified deterministic 1v1 runner, not the official Battlesnake engine",
        },
    }


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def _sum_dicts(dicts) -> dict[str, int]:
    total: dict[str, int] = {}
    for item in dicts:
        for key, value in item.items():
            total[key] = total.get(key, 0) + int(value)
    return total


def _print_summary(summary: dict) -> None:
    print(
        "Duels: "
        f"{summary['wins']}W/{summary['losses']}L/{summary['draws']}D "
        f"over {summary['games']} games, win_rate={summary['win_rate']:.3f}"
    )
    print(
        "Latency candidate ms: "
        f"median={summary['median_latency_ms']:.2f}, "
        f"p95={summary['p95_latency_ms']:.2f}, "
        f"max={summary['max_latency_ms']:.2f}"
    )
    print(
        "Quality counters: "
        f"fallbacks={summary['fallback_count']}, "
        f"invalid_moves={summary['invalid_moves']}, "
        f"timeouts={summary['timeouts']}"
    )
    print(f"Results written to {summary['configuration']['trajectory_output']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--swap-positions", action="store_true")
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("benchmark/results.json"))
    parser.add_argument("--trajectory-output", type=Path, default=None)
    args = parser.parse_args()
    run_duels(args.output, args.games, args.seed, args.swap_positions, args.max_turns, args.trajectory_output)
