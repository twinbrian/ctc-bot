"""Train the in-storm models for the live bot and save everything it needs to bot/models.pkl.

    /opt/anaconda3/bin/python3 -I bot/train.py                 # train on all 600 seasons (for live)
    /opt/anaconda3/bin/python3 -I bot/train.py --holdout 100   # train on seasons 1-500 (for an honest replay test)

Features are keyed on storm AGE, not clock time, because the server's round timing (20 s pre-storm,
180 s aftermath) differs from the research data (60 s / 240 s).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
os.chdir(ROOT / "analysis")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from load import STOCKS, ISLANDS
from features import OUTLOOKS
from montecarlo import Library, MU, SIG
from damage import COMP_IDX, W

BASE = ["age", "round", "x", "y", "vx", "vy", "radius", "intensity"] + [f"post_{k}" for k in OUTLOOKS] \
    + [f"p_{i}" for i in ISLANDS]


def design(F: pd.DataFrame, s: str) -> pd.DataFrame:
    """Same features as analysis/backtest.py, with age = t - 60 instead of clock time."""
    X = F.assign(age=F.t - 60)[BASE].copy()
    for sym in STOCKS:
        X[f"gexp_{sym}"] = F[f"Vexp_{sym}"] - F[f"H_{sym}"]
    X["gpre"] = F[f"Hpre_{s}"] - F[f"H_{s}"]
    X["glock"] = F[f"Vlock_{s}"] - F[f"H_{s}"]
    X["mom"] = F[f"H_{s}"] - F[f"Hlag_{s}"]
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=0, help="leave the last N seasons out of training")
    args = ap.parse_args()
    F = pd.read_pickle("nowcast_grid.pkl").reset_index(drop=True)
    train = F[F.season <= 600 - args.holdout]
    models = {}
    for s in STOCKS:
        X = design(train, s)
        y = (train[f"Hend_{s}"] - train[f"H_{s}"]).values
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
                                          min_samples_leaf=60, l2_regularization=1.0, random_state=0)
        models[s] = m.fit(X.values, y)
        print(f"trained {s}: {len(y)} rows")
    lib = Library()
    consts = {
        "v": lib.v,                                                         # site vulnerabilities (17)
        "q": (1 - np.exp(-lib.v * lib.X)).mean(0) * np.exp(MU + SIG ** 2 / 2),   # expected damage of a future storm
        "cfut": {c: float((lib.closed[:, COMP_IDX[c]] * W[COMP_IDX[c]]).sum(1).mean()) for c in ("PORT", "GRID", "FERRY")},
        "decision_ages": list(range(10, 471, 20)),                          # research grid t = 70..530
        "feature_names": list(design(F.head(1), "PORT").columns),
        "holdout": args.holdout,
    }
    out = ROOT / "bot" / ("models.pkl" if args.holdout == 0 else f"models_holdout{args.holdout}.pkl")
    out.write_bytes(pickle.dumps({"models": models, "consts": consts}))
    print("saved", out)


if __name__ == "__main__":
    main()
