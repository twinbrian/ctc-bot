"""Public-information feature table at a grid of times within each round (what a bot could know at t)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from load import CASE, ISLANDS, STOCKS, tables

OUTLOOKS = list(CASE["outlooks"])
PRIOR = {"chain": 0.227, "direct": 0.250, "erratic": 0.020, "eyewall": 0.145, "recurve": 0.207, "stall": 0.151}
P_OFF, P_WIRE = (0.70, 0.06), (0.40, 0.12)       # P(say k | truth k), P(say k | truth j != k)


def posterior(msgs) -> dict:
    """Bayesian posterior over steering patterns from (source, outlook) messages so far."""
    lp = {k: np.log(PRIOR[k]) for k in OUTLOOKS}
    for src, out in msgs:
        hit, miss = P_OFF if src == "official" else P_WIRE
        for k in OUTLOOKS:
            lp[k] += np.log(hit if k == out else miss)
    m = max(lp.values())
    z = sum(np.exp(v - m) for v in lp.values())
    return {k: float(np.exp(lp[k] - m) / z) for k in OUTLOOKS}


def build(times=(120, 180, 240, 300, 360, 420, 480, 520)) -> pd.DataFrame:
    T = tables()
    O = T["obs"]; A = T["official"]; Wr = T["wire"]; P = T["prices"]
    Og = dict(list(O.groupby(["season", "round"])))
    Ag = dict(list(A.groupby(["season", "round"])))
    Wg = dict(list(Wr.groupby(["season", "round"])))
    H = P.set_index(["season", "round", "t"])[STOCKS]
    port = CASE["companies"]["PORT"]["sites"][0]
    rows = []
    for key, g in Og.items():
        g = g.sort_values("t")
        a_all = Ag.get(key, A.iloc[:0]); w_all = Wg.get(key, Wr.iloc[:0])
        for t in times:
            o = g[g.t <= t].iloc[-1]
            a = a_all[a_all.t <= t].sort_values("t"); w = w_all[w_all.t <= t]
            msgs = [("official", x) for x in a.outlook] + [("wire", x) for x in w.outlook]
            post = posterior(msgs)
            last = a.iloc[-1] if len(a) else None
            row = {"season": key[0], "round": key[1], "t": t, "x": o.x, "y": o.y, "vx": o.vx, "vy": o.vy,
                   "radius": o.radius, "intensity": o.intensity, "age": o.age,
                   "d_port": np.hypot(o.x - port["x"], o.y - port["y"]) / o.radius,
                   **{f"post_{k}": post[k] for k in OUTLOOKS}}
            if last is not None:
                row.update({f"p_{i}": last[f"p_{i}"] for i in ISLANDS})
                cone = last.cone
                far = cone[-1]
                row.update({"cone_x": far["x"], "cone_y": far["y"], "cone_spread": far["spread"]})
            for s in STOCKS:
                row[f"H_{s}"] = H.at[(key[0], key[1], float(t)), s]
                row[f"Hlag_{s}"] = H.at[(key[0], key[1], float(t - 10)), s]
                row[f"Hend_{s}"] = H.at[(key[0], key[1], 542.0), s]
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    F = build()
    F.to_pickle("features_grid.pkl")
    print(F.shape)
