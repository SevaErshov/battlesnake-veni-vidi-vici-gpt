"""Generate a synthetic Battlesnake imitation-learning dataset and export weights.

Why synthetic? Public Battlesnake move-level replay datasets are not consistently
available in a stable downloadable tabular format. This script creates many legal-ish
positions, asks the stronger expert heuristic in logic.py to rank moves, and trains
small models to imitate it.

Usage from project root:
    python training/train_models.py --states 12000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge, SGDClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logic  # noqa: E402

Point = Tuple[int, int]
MOVES = ["up", "down", "left", "right"]
DIRS = [(0, 1), (0, -1), (-1, 0), (1, 0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "model_weights.py")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: List[List[float]] = []
    y_best: List[int] = []
    y_reg: List[float] = []
    state_ids: List[int] = []
    moves: List[str] = []
    raw_scores: List[float] = []

    generated = 0
    attempts = 0
    while generated < args.states and attempts < args.states * 20:
        attempts += 1
        state = generate_state(rng)
        candidates = logic.build_candidates(state)
        if len(candidates) < 2:
            continue
        ranked = logic.rank_moves(state, "expert")
        if not ranked:
            continue
        best_move = ranked[0].move
        for c in candidates:
            score = c.safety_score + c.expert_score
            rows.append(list(c.features))
            y_best.append(1 if c.move == best_move else 0)
            y_reg.append(score / 1000.0)
            state_ids.append(generated)
            moves.append(c.move)
            raw_scores.append(score)
        generated += 1

    X = np.asarray(rows, dtype=np.float64)
    y_cls = np.asarray(y_best, dtype=np.int64)
    y_r = np.asarray(y_reg, dtype=np.float64)
    state_ids_np = np.asarray(state_ids, dtype=np.int64)
    moves_np = np.asarray(moves)

    train_state_ids, test_state_ids = train_test_split(
        np.unique(state_ids_np), test_size=0.25, random_state=args.seed
    )
    train_mask = np.isin(state_ids_np, train_state_ids)
    test_mask = np.isin(state_ids_np, test_state_ids)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])

    models = {}
    report: Dict[str, object] = {
        "states_generated": int(generated),
        "candidate_rows": int(len(rows)),
        "features": logic.FEATURE_NAMES,
        "seed": args.seed,
    }

    ridge = Ridge(alpha=2.0, random_state=args.seed)
    ridge.fit(X_train, y_r[train_mask])
    models["ridge"] = ridge

    logistic = LogisticRegression(max_iter=1000, C=1.25, class_weight="balanced", random_state=args.seed)
    logistic.fit(X_train, y_cls[train_mask])
    models["logistic"] = logistic

    svm = SGDClassifier(
        loss="hinge",
        alpha=0.0002,
        max_iter=2500,
        tol=1e-4,
        class_weight="balanced",
        random_state=args.seed,
    )
    svm.fit(X_train, y_cls[train_mask])
    models["svm"] = svm

    perceptron = Perceptron(
        penalty="l2",
        alpha=0.0003,
        max_iter=2500,
        tol=1e-4,
        class_weight="balanced",
        random_state=args.seed,
    )
    perceptron.fit(X_train, y_cls[train_mask])
    models["perceptron"] = perceptron

    mlp = MLPRegressor(
        hidden_layer_sizes=(10,),
        activation="tanh",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.006,
        max_iter=500,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
    )
    mlp.fit(X_train, y_r[train_mask])
    models["mlp"] = mlp

    # Metrics.
    for name, model in models.items():
        if name in {"ridge", "mlp"}:
            pred_score = model.predict(X_test)
            report[name] = {
                "top1_acc": top1_accuracy(pred_score, state_ids_np[test_mask], moves_np[test_mask], y_cls[test_mask]),
                "mse": float(mean_squared_error(y_r[test_mask], pred_score)),
            }
        else:
            if hasattr(model, "decision_function"):
                pred_score = model.decision_function(X_test)
            else:
                pred_score = model.predict_proba(X_test)[:, 1]
            metric = {
                "top1_acc": top1_accuracy(pred_score, state_ids_np[test_mask], moves_np[test_mask], y_cls[test_mask]),
                "row_accuracy": float(accuracy_score(y_cls[test_mask], model.predict(X_test))),
            }
            try:
                metric["row_auc"] = float(roc_auc_score(y_cls[test_mask], pred_score))
            except Exception:
                pass
            report[name] = metric

    # Ensemble metric.
    ensemble_scores = (
        normalize(models["ridge"].predict(X_test))
        + normalize(models["logistic"].decision_function(X_test))
        + normalize(models["svm"].decision_function(X_test))
        + normalize(models["perceptron"].decision_function(X_test))
        + normalize(models["mlp"].predict(X_test))
    ) / 5.0
    report["ensemble"] = {
        "top1_acc": top1_accuracy(ensemble_scores, state_ids_np[test_mask], moves_np[test_mask], y_cls[test_mask])
    }

    weights = export_weights(models, scaler)
    write_weights(args.out, weights)
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Exported weights to {args.out}")


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    if sd < 1e-9:
        return v * 0
    return (v - v.mean()) / sd


def top1_accuracy(pred_scores: Sequence[float], state_ids: Sequence[int], moves: Sequence[str], y_best: Sequence[int]) -> float:
    pred_scores = np.asarray(pred_scores)
    state_ids = np.asarray(state_ids)
    moves = np.asarray(moves)
    y_best = np.asarray(y_best)
    correct = 0
    total = 0
    for sid in np.unique(state_ids):
        idx = np.where(state_ids == sid)[0]
        if len(idx) == 0:
            continue
        chosen_idx = idx[int(np.argmax(pred_scores[idx]))]
        correct += int(y_best[chosen_idx] == 1)
        total += 1
    return float(correct / max(1, total))


def export_linear(model, scaler: StandardScaler) -> Dict[str, object]:
    coef = model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
    intercept = float(np.asarray(model.intercept_).ravel()[0]) if hasattr(model, "intercept_") else 0.0
    return {
        "mean": scaler.mean_.round(8).tolist(),
        "scale": scaler.scale_.round(8).tolist(),
        "coef": np.asarray(coef).round(8).tolist(),
        "intercept": float(np.round(intercept, 8)),
    }


def export_weights(models: Dict, scaler: StandardScaler) -> Dict[str, object]:
    mlp = models["mlp"]
    return {
        "feature_names": logic.FEATURE_NAMES,
        "ridge": export_linear(models["ridge"], scaler),
        "logistic": export_linear(models["logistic"], scaler),
        "svm": export_linear(models["svm"], scaler),
        "perceptron": export_linear(models["perceptron"], scaler),
        "mlp": {
            "mean": scaler.mean_.round(8).tolist(),
            "scale": scaler.scale_.round(8).tolist(),
            "hidden_weights": np.asarray(mlp.coefs_[0]).T.round(8).tolist(),
            "hidden_bias": np.asarray(mlp.intercepts_[0]).round(8).tolist(),
            "out_weights": np.asarray(mlp.coefs_[1]).ravel().round(8).tolist(),
            "out_bias": float(np.asarray(mlp.intercepts_[1]).ravel()[0].round(8)),
        },
    }


def write_weights(path: Path, weights: Dict[str, object]) -> None:
    payload = "# Auto-generated by training/train_models.py\nMODEL_WEIGHTS = " + repr(weights) + "\n"
    path.write_text(payload, encoding="utf-8")


def generate_state(rng: random.Random) -> Dict:
    width = rng.choice([7, 11, 11, 11, 13])
    height = width
    snake_count = rng.randint(2, 4)
    occupied: set[Point] = set()
    snakes = []
    for i in range(snake_count):
        length = rng.randint(3, min(12, width * height // 7))
        body = random_snake_body(rng, width, height, length, occupied)
        if not body:
            continue
        occupied.update(body)
        snakes.append(
            {
                "id": "you" if i == 0 else f"enemy-{i}",
                "name": "you" if i == 0 else f"enemy-{i}",
                "health": rng.randint(12, 100),
                "body": [{"x": x, "y": y} for x, y in body],
                "head": {"x": body[0][0], "y": body[0][1]},
                "length": len(body),
            }
        )
    if not snakes or snakes[0]["id"] != "you":
        # Rare generation failure: deterministic fallback.
        snakes = [
            snake_dict("you", [(3, 3), (3, 2), (3, 1)], 80),
            snake_dict("enemy-1", [(6, 6), (6, 5), (6, 4)], 80),
        ]
        width = height = 11
        occupied = {(3, 3), (3, 2), (3, 1), (6, 6), (6, 5), (6, 4)}

    empty = [(x, y) for x in range(width) for y in range(height) if (x, y) not in occupied]
    rng.shuffle(empty)
    food_count = rng.randint(1, min(6, len(empty)))
    food = empty[:food_count]
    hazards = []
    if rng.random() < 0.08:
        hazards = empty[food_count : food_count + rng.randint(1, min(5, max(1, len(empty) - food_count)))]
    you = snakes[0]
    return {
        "game": {"id": "synthetic", "ruleset": {"name": "standard", "settings": {}}, "timeout": 500},
        "turn": rng.randint(1, 250),
        "board": {
            "height": height,
            "width": width,
            "food": [{"x": x, "y": y} for x, y in food],
            "hazards": [{"x": x, "y": y} for x, y in hazards],
            "snakes": snakes,
        },
        "you": you,
    }


def snake_dict(sid: str, body: List[Point], health: int) -> Dict:
    return {
        "id": sid,
        "name": sid,
        "health": health,
        "body": [{"x": x, "y": y} for x, y in body],
        "head": {"x": body[0][0], "y": body[0][1]},
        "length": len(body),
    }


def random_snake_body(
    rng: random.Random, width: int, height: int, length: int, occupied_global: set[Point]
) -> Optional[List[Point]]:
    for _ in range(80):
        head = (rng.randrange(width), rng.randrange(height))
        if head in occupied_global:
            continue
        body = [head]
        local = {head}
        ok = True
        for _step in range(length - 1):
            candidates = []
            for dx, dy in DIRS:
                nb = (body[-1][0] + dx, body[-1][1] + dy)
                if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in local and nb not in occupied_global:
                    candidates.append(nb)
            if not candidates:
                ok = False
                break
            body.append(rng.choice(candidates))
            local.add(body[-1])
        if ok and len(body) == length:
            return body
    return None


if __name__ == "__main__":
    main()
