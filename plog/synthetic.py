"""Explicitly synthetic mechanism experiments, never a football performance benchmark."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from evaluation import possession_metrics as pm
from starter.baseline import smooth
from .config import Config, config_dict
from .data import Tracking
from .inference import infer


def fixture(seed=0, corrupt=True):
    """Programmed possessions, visible still stoppage, block dropout and a short recovery.

    This intentionally isolates known failure mechanisms and is not a simulator of match
    statistics. Labels and synthetic observations are generated together by code.
    """
    rng = np.random.default_rng(seed)
    n = 1400
    truth = np.array(list("H" * 350 + "A" * 250 + "D" * 180 + "H" * 230 + "A" * 25 + "H" * 365))
    teams = np.array([0, 0, 1, 1], dtype=np.int8)
    ball = np.column_stack((0.5 + 0.03 * np.sin(np.arange(n) / 12), np.full(n, 0.5)))
    players = np.tile(np.array([[0.2, .3], [.25, .7], [.8, .3], [.75, .7]]), (n, 1, 1))
    speeds = np.full((n, 4), 2.0)
    for side, label in ((0, "H"), (2, "A")):
        mask = truth == label
        players[mask, side] = ball[mask] + np.array([0.003, 0])
    dead = truth == "D"
    ball[dead] = [0.55, 0.5]
    players[dead] = [[.3, .3], [.4, .7], [.65, .3], [.7, .7]]
    speeds[dead] = .05
    if corrupt:
        # Missingness and erroneous nearest-player evidence are observation failures.
        ball[80:260] = np.nan
        ball[405:495] = np.nan
        ball[1100:1250] = np.nan
        for start in rng.integers(20, 570, size=12):
            if truth[start] == "H":
                players[start:start + 2, 2] = ball[start:start + 2]
                players[start:start + 2, 0] += .04
        players[300:330, 1] = np.nan
    return Tracking(players, ball, teams, speeds), truth


def nearest_codes(tracking):
    dist = np.linalg.norm((tracking.players - tracking.ball[:, None]) * np.asarray(tracking.pitch), axis=2)
    dist[~np.isfinite(dist)] = np.inf
    slot = dist.argmin(axis=1)
    valid = np.isfinite(dist[np.arange(len(dist)), slot])
    return np.where(valid, tracking.teams[slot], -1).astype(np.int8)


def run(output=None, seeds=(0, 1, 2, 3, 4)):
    cfg = Config()
    rows = []
    for seed in seeds:
        tracking, truth = fixture(seed)
        codes = nearest_codes(tracking)
        models = {"official_raw": lambda: smooth(codes, 0),
                  "official_smooth50": lambda: smooth(codes, 50),
                  "geometry_without_decoder": lambda: infer(tracking, replace(cfg, use_decoder=False)),
                  "plog_candidate": lambda: infer(tracking, cfg)}
        for name, predict in models.items():
            start = time.perf_counter()
            labels = predict()
            elapsed = time.perf_counter() - start
            result = pm.evaluate(pm.label_codes(truth), pm.label_codes(labels))
            rows.append({"seed": seed, "model": name, "spell_f1_mean": result["spell_f1_mean"],
                         "macro_f1": result["macro_f1"], "possessions": result["n_pred_spells"],
                         "gt_possessions": result["n_gt_spells"], "seconds": elapsed})
    record = {"domain": "synthetic_mechanism_only", "config": config_dict(cfg), "rows": rows,
              "warning": "Designed mechanism fixtures; cannot establish football or competition improvement."}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/synthetic/results.json")
    args = parser.parse_args()
    record = run(args.out)
    for name in sorted({r["model"] for r in record["rows"]}):
        rows = [r for r in record["rows"] if r["model"] == name]
        print(f"{name}: synthetic spellF1={np.mean([r['spell_f1_mean'] for r in rows]):.4f}; "
              f"macroF1={np.mean([r['macro_f1'] for r in rows]):.4f}")
    print("Synthetic mechanisms only; no claim of football generalization.")


if __name__ == "__main__":
    main()
