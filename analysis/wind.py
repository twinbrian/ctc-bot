"""Wind at each company site, reconstructed from the storm observations.

Fitted model (exact against settlements; see notes):
    wind(site, t) = intensity(t) * exp(-(d(t) / radius(t))**2)
    hurricane conditions (site closed) while wind > THR
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load import CASE, PHYS

A, THR = 1.0, 0.35
SITES = pd.DataFrame([{"company": c, "site": s["name"], "island": s["island"], "x": s["x"], "y": s["y"],
                       "weight": s["weight"]} for c in PHYS for s in CASE["companies"][c]["sites"]])
SX, SY = SITES.x.values, SITES.y.values


def storm_arrays(g: pd.DataFrame, step: float = 1.0):
    """Interpolate one storm's observations (one row per 2 s) onto a `step`-second grid."""
    g = g.sort_values("age")
    ages = np.arange(0, g.age.max() + 1e-9, step)
    out = {c: np.interp(ages, g.age.values, g[c].values) for c in ("x", "y", "radius", "intensity")}
    return ages, out


def site_wind(arr, a: float = A) -> np.ndarray:
    """(n_ages, n_sites) wind matrix."""
    d = np.hypot(arr["x"][:, None] - SX[None, :], arr["y"][:, None] - SY[None, :])
    return arr["intensity"][:, None] * np.exp(-a * (d / arr["radius"][:, None]) ** 2)


def site_features(obs: pd.DataFrame, step: float = 1.0, a: float = A, thr: float = THR) -> pd.DataFrame:
    """Per storm x site: peak wind, seconds closed, integrated excess wind, min distance/radius."""
    rows = []
    for (season, rnd), g in obs.groupby(["season", "round"]):
        ages, arr = storm_arrays(g, step)
        w = site_wind(arr, a)
        d = np.hypot(arr["x"][:, None] - SX[None, :], arr["y"][:, None] - SY[None, :]) / arr["radius"][:, None]
        over = w > thr
        for j in range(len(SITES)):
            rows.append({"season": season, "round": rnd, "site_i": j,
                         "wmax": w[:, j].max(), "closed_s": over[:, j].sum() * step,
                         "excess": np.clip(w[:, j] - thr, 0, None).sum() * step,
                         "wint": w[:, j].sum() * step, "w2": (w[:, j] ** 2).sum() * step,
                         "umin": d[:, j].min()})
    return pd.DataFrame(rows)
