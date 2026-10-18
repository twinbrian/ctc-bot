"""Out-of-sample in-storm strategy: predict the house move to the end of the storm, trade toward it.

Model:   y = H_end(t=542) - H(t), per symbol, gradient boosting on public features at t.
CV:      5 folds by season (a season never appears in both train and test).
Trading: every 20 s from t=70 to t=530, target position = limit * clip(pred / scale, -1, 1);
         pay `cost` per share traded (half-spread); positions are closed at H_end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from load import CASE, STOCKS
from features import OUTLOOKS

SYMS = STOCKS
LIM = CASE["limits"]


def design(F: pd.DataFrame, s: str) -> pd.DataFrame:
    base = ["t", "round", "x", "y", "vx", "vy", "radius", "intensity"] + [f"post_{k}" for k in OUTLOOKS] \
        + [c for c in F.columns if c.startswith("p_")]
    X = F[base].copy()
    for sym in SYMS:      # cross-symbol physical gaps are informative (common storm)
        X[f"gexp_{sym}"] = F[f"Vexp_{sym}"] - F[f"H_{sym}"]
    X["gpre"] = F[f"Hpre_{s}"] - F[f"H_{s}"]
    X["glock"] = F[f"Vlock_{s}"] - F[f"H_{s}"]
    X["mom"] = F[f"H_{s}"] - F[f"Hlag_{s}"]
    return X


def oos_predictions(F: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=F.index)
    for s in SYMS:
        X = design(F, s).values
        y = (F[f"Hend_{s}"] - F[f"H_{s}"]).values
        pred = np.zeros_like(y)
        for tr, te in GroupKFold(5).split(X, y, F.season):
            m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
                                              min_samples_leaf=60, l2_regularization=1.0)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        out[s] = pred
    return out


def simulate(F: pd.DataFrame, pred: pd.DataFrame, cost: float, scale: dict) -> pd.DataFrame:
    """PnL per storm and symbol."""
    res = []
    for (season, rnd), g in F.groupby(["season", "round"]):
        g = g.sort_values("t")
        pr = pred.loc[g.index]
        row = {"season": season, "round": rnd}
        for s in SYMS:
            pos, pnl = 0.0, 0.0
            H = g[f"H_{s}"].values
            for k in range(len(g)):
                target = LIM[s] * np.clip(pr[s].values[k] / scale[s], -1, 1)
                trade = target - pos
                pnl -= abs(trade) * cost * max(H[k], 1) / 100   # cost in % of price
                pos = target
                nxt = H[k + 1] if k + 1 < len(g) else g[f"Hend_{s}"].values[0]
                pnl += pos * (nxt - H[k])
            pnl -= abs(pos) * cost * max(g[f"Hend_{s}"].values[0], 1) / 100
            row[s] = pnl
        res.append(row)
    return pd.DataFrame(res)
