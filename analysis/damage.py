"""Site-level damage model: storm track -> damage % per company.

    hazard_s(storm) = v_s * sum_t I(t)^q * g(u_s(t)) * dt,    u = distance / radius, g = 0 for u >= 2
    site damage added = (1 - D_s) * (1 - exp(-hazard_s))      (D_s = site damage before the storm)
    company damage_pct = 100 * sum_s weight_s * site damage added
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load import PHYS
from wind import SITES, SX, SY, storm_arrays

STEP = 0.5
COMP_IDX = {c: np.where(SITES.company.values == c)[0] for c in PHYS}
W = SITES.weight.values


def kernel(u, beta, gamma):
    """Damage-rate shape vs distance/radius; zero outside 2 radii."""
    return np.where(u < 2.0, np.exp(-beta * u ** gamma), 0.0)


def storm_tensors(obs: pd.DataFrame, keys):
    """u (n_storms, n_t, 17) and I (n_storms, n_t), padded with I=0 past dissipation."""
    groups = dict(list(obs.groupby(["season", "round"])))
    n_t = int(480 / STEP) + 1
    U = np.full((len(keys), n_t, len(SITES)), 9.0, np.float32)
    I = np.zeros((len(keys), n_t), np.float32)
    for i, k in enumerate(keys):
        ages, arr = storm_arrays(groups[k], STEP)
        n = len(ages)
        U[i, :n] = np.hypot(arr["x"][:, None] - SX, arr["y"][:, None] - SY) / arr["radius"][:, None]
        I[i, :n] = arr["intensity"]
    return U, I


def exposure(U, I, beta, gamma, q=1.0):
    """(n_storms, 17) intensity-weighted kernel exposure."""
    return (I[:, :, None] ** q * kernel(U, beta, gamma)).sum(1) * STEP


def roll_season(X, v, seasons, rounds):
    """Apply hazards storm by storm within each season. Returns site damage added (n_storms, 17)."""
    add = np.zeros_like(X)
    Dsite = {}
    for i, (s, r) in enumerate(zip(seasons, rounds)):
        prev = Dsite.get(s, np.zeros(X.shape[1])) if r > 1 else np.zeros(X.shape[1])
        a = (1 - prev) * (1 - np.exp(-v * X[i]))
        add[i] = a
        Dsite[s] = prev + a
    return add


def company_damage(site_add):
    return pd.DataFrame({c: 100 * (site_add[:, idx] * W[idx]).sum(1) for c, idx in COMP_IDX.items()})
