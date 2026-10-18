"""Physical nowcast of each storm's damage at time t, from public information only.

locked-in:  exposure accumulated along the observed track up to t
projected:  expected future exposure along the latest official cone. Position at future age tau is
            ~ N(mu_tau, (spread/sqrt 2)^2 per axis), so E[exp(-|X-s|^2/r^2)] = r^2/(r^2+2 s^2) exp(-|mu-s|^2/(r^2+2 s^2))
value:      season damage so far (previous rounds) + this storm's locked-in + projected damage, through value.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load import PHYS, STOCKS, SYMS, ISLANDS, tables
from wind import SITES, SX, SY, THR
from damage import COMP_IDX, W
from value import values
from features import posterior, OUTLOOKS
from montecarlo import Library

DISS = 420.0     # intensity decays linearly to 0 between storm age 420 and 480


def cone_path(o, cone, age_now):
    """Future ages (1 s grid) with mean position and per-axis sd, from current obs + cone points."""
    pts = [(age_now, o["x"], o["y"], 0.0)] + [(c["age"], c["x"], c["y"], c["spread"] / np.sqrt(2)) for c in cone if c["age"] > age_now]
    if len(pts) < 2:
        return None
    a = np.array(pts)
    taus = np.arange(age_now + 1, 481)
    # extrapolate beyond the last cone point with the last segment's velocity and growing spread
    mx = np.interp(taus, a[:, 0], a[:, 1]); my = np.interp(taus, a[:, 0], a[:, 2]); sd = np.interp(taus, a[:, 0], a[:, 3])
    last = a[-1]; prev = a[-2]
    beyond = taus > last[0]
    if beyond.any():
        dt = taus[beyond] - last[0]; seg = max(last[0] - prev[0], 1.0)
        mx[beyond] = last[1] + (last[1] - prev[1]) / seg * dt
        my[beyond] = last[2] + (last[2] - prev[2]) / seg * dt
        sd[beyond] = last[3] + (last[3] - prev[3]) / seg * dt
    return taus, mx, my, sd


def projected(o, cone, age_now):
    """Expected future exposure (17,) and expected future closed seconds (17,) along the cone."""
    cp = cone_path(o, cone, age_now)
    if cp is None:
        return np.zeros(17), np.zeros(17)
    taus, mx, my, sd = cp
    r = o["radius"]
    I = np.where(taus < DISS, o["intensity"], o["intensity"] * np.clip((480 - taus) / (480 - DISS), 0, 1))
    den = r * r + 2 * sd[:, None] ** 2
    d2 = (mx[:, None] - SX) ** 2 + (my[:, None] - SY) ** 2
    ek = (r * r / den) * np.exp(-d2 / den)
    X = (I[:, None] * ek).sum(0)
    # closure: P(wind > THR) ~ P(|X-s| < r sqrt(ln(I/THR))) under a 2-D normal -> Rice; approximate with
    # the mean-path indicator smoothed by the spread
    rc2 = r * r * np.log(np.clip(I / THR, 1e-9, None))[:, None]
    # P(site inside the closure radius rc) for a 2-D normal position, approximated by a Gaussian blur of the disc
    rc2 = np.clip(rc2, 0, None)
    pclosed = np.clip((rc2 / (rc2 + 2 * sd[:, None] ** 2 + 1e-9)) * np.exp(-d2 / (rc2 + 2 * sd[:, None] ** 2 + 1e-9)) * 2.0, 0, 1)
    return X, pclosed.sum(0)


def build(times=tuple(range(70, 531, 20))):
    T = tables(); lib = Library()
    st = lib.storms
    Dsite_after = lib.site_state()                                    # after each storm (reported)
    O = T["obs"]; A = T["official"]; Wr = T["wire"]; P = T["prices"]
    Og = dict(list(O.groupby(["season", "round"]))); Ag = dict(list(A.groupby(["season", "round"])))
    Wg = dict(list(Wr.groupby(["season", "round"])))
    H = {s: P.sort_values(["season", "round", "t"])[s].values.reshape(len(st), 390) for s in SYMS}
    z = np.load("storm_tensors.npz"); U, Iv = z["U"], z["I"]
    kern = np.where(U < 2, np.exp(-U.astype(np.float64) ** 2), 0.0) * Iv[:, :, None]
    cumX = np.cumsum(kern, 1) * 0.5                                   # (1800, 961, 17) by storm age (0.5 s)
    cumC = np.cumsum((Iv[:, :, None] * np.exp(-U.astype(np.float64) ** 2)) > THR, 1) * 0.5
    Cprev = {c: np.zeros(len(st)) for c in ("PORT", "GRID", "FERRY")}
    # expected damage of one future storm per site (library average, incl. the mean of the lognormal noise)
    from montecarlo import MU, SIG
    q = (1 - np.exp(-lib.v * lib.X)).mean(0) * np.exp(MU + SIG ** 2 / 2)
    cfut = {c: (lib.closed[:, COMP_IDX[c]] * W[COMP_IDX[c]]).sum(1).mean() for c in Cprev}
    rows = []
    for i, (s, r) in enumerate(zip(st.season.values, st["round"].values)):
        D0 = Dsite_after[i - 1] if r > 1 else np.zeros(17)
        C0 = {c: sum((lib.closed[i - k][COMP_IDX[c]] * W[COMP_IDX[c]]).sum() for k in range(1, r)) for c in Cprev}
        g = Og[(s, r)].sort_values("t"); a_all = Ag[(s, r)]; w_all = Wg.get((s, r), Wr.iloc[:0])
        for t in times:
            age = t - 60
            o = g[g.t <= t].iloc[-1]
            a = a_all[a_all.t <= t].sort_values("t"); w = w_all[w_all.t <= t]
            j = int(age / 0.5)
            Xs = cumX[i, j]; Cs = cumC[i, j]
            if len(a):
                Xp, Cp = projected(o, a.iloc[-1].cone, age)
            else:
                Xp, Cp = np.zeros(17), np.zeros(17)
            def val(X, Cc, k=0):
                add = (1 - D0) * (1 - np.exp(-lib.v * X))
                Ds = 1 - (1 - (D0 + add)) * (1 - q) ** k          # expected site damage after k more storms
                Dc = {c: (Ds[COMP_IDX[c]] * W[COMP_IDX[c]]).sum() for c in PHYS}
                C = {c: C0[c] + (Cc[COMP_IDX[c]] * W[COMP_IDX[c]]).sum() + k * cfut[c] for c in Cprev}
                return values(Dc, C)
            vl = val(Xs, Cs); vp = val(Xs + Xp, Cs + Cp); ve = val(Xs + Xp, Cs + Cp, 3 - r)
            post = posterior([("official", x) for x in a.outlook] + [("wire", x) for x in w.outlook])
            row = {"season": s, "round": r, "t": t, "x": o.x, "y": o.y, "vx": o.vx, "vy": o.vy, "radius": o.radius,
                   "intensity": o.intensity, **{f"post_{k}": post[k] for k in OUTLOOKS}}
            if len(a):
                row.update({f"p_{isl}": a.iloc[-1][f"p_{isl}"] for isl in ISLANDS})
            for sym in SYMS:
                row[f"H_{sym}"] = H[sym][i, t // 2]
                row[f"Hpre_{sym}"] = H[sym][i, 29]
                row[f"Hlag_{sym}"] = H[sym][i, t // 2 - 5]
                row[f"Hend_{sym}"] = H[sym][i, 271]
                row[f"Hclose_{sym}"] = H[sym][i, 389]
                row[f"Vlock_{sym}"] = float(vl[sym]); row[f"Vproj_{sym}"] = float(vp[sym]); row[f"Vexp_{sym}"] = float(ve[sym])
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import time
    t0 = time.time()
    F = build()
    F.to_pickle("nowcast_grid.pkl")
    print(F.shape, f"{time.time() - t0:.0f}s")
