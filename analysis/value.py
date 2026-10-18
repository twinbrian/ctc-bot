"""Fundamental value of every symbol, recovered exactly from the 600 research settlements.

Inputs (season to date):
    D[c]  cumulative damage of island business c, as a fraction of its original value
          (the sum of the per-storm damage_pct / 100 -- damage adds, it does not compound)
    C[c]  outage: sum over c's sites of (site weight x real seconds with wind > 0.35)
          (only PORT, GRID and FERRY are priced on outage)

    PORT  = 100 - 55 D - 0.02 C            insured asset value $900M
    ISLE  =  60 - 38 D                      $600M
    AQUA  =  35 - 20 D                      $250M
    GRID  =  50 - 24 D - 0.05 C             $700M
    FERRY =  25 -  5 D - 0.06 C             $120M
    L     = sum(insured asset value x D)    season insured loss, $M (== manifest season_loss)
    HAVEN =  75 - 0.06 * clip(L - 400, 0, 900)      layer $400M .. $1300M
    REIN  =  90 - 0.035 * max(L - 1300, 0)          layer above $1300M (upper limit unseen)
    RECON =  40 + 24 (1 - exp(-(900 D_PORT + 700 D_GRID + 120 D_FERRY) / 1000))
    MAT   =  28 + 14 (1 - exp(-L / 1200))
    GEN   =  32 + 18 (1 - exp(-C_GRID / 120))
    ARCH  = PORT + ISLE + 2 AQUA + GRID + 2 FERRY + HAVEN
    REBLD = 2 RECON + 2 MAT + GEN
"""

from __future__ import annotations

import numpy as np

ASSET = {"PORT": 900.0, "ISLE": 600.0, "AQUA": 250.0, "GRID": 700.0, "FERRY": 120.0}
V0 = {"PORT": 100.0, "ISLE": 60.0, "AQUA": 35.0, "GRID": 50.0, "FERRY": 25.0}
KD = {"PORT": 55.0, "ISLE": 38.0, "AQUA": 20.0, "GRID": 24.0, "FERRY": 5.0}
KC = {"PORT": 0.02, "ISLE": 0.0, "AQUA": 0.0, "GRID": 0.05, "FERRY": 0.06}
BASKETS = {"ARCH": {"PORT": 1, "ISLE": 1, "AQUA": 2, "GRID": 1, "FERRY": 2, "HAVEN": 1},
           "REBLD": {"RECON": 2, "MAT": 2, "GEN": 1}}


def insured_loss(D):
    """Season insured loss in $M from cumulative damage fractions."""
    return sum(ASSET[c] * D[c] for c in ASSET)


def haven(L):
    return 75.0 - 0.06 * np.clip(np.asarray(L, float) - 400.0, 0.0, 900.0)


def rein(L):
    return 90.0 - 0.035 * np.clip(np.asarray(L, float) - 1300.0, 0.0, None)


def values(D, C, L=None):
    """Value of all 12 symbols. D, C: dict company -> fraction / weighted closed seconds (scalars or arrays)."""
    L = insured_loss(D) if L is None else L
    v = {c: V0[c] - KD[c] * D[c] - KC[c] * C.get(c, 0.0) for c in V0}
    v["HAVEN"], v["REIN"] = haven(L), rein(L)
    v["RECON"] = 40 + 24 * (1 - np.exp(-(900 * D["PORT"] + 700 * D["GRID"] + 120 * D["FERRY"]) / 1000))
    v["MAT"] = 28 + 14 * (1 - np.exp(-np.asarray(L, float) / 1200))
    v["GEN"] = 32 + 18 * (1 - np.exp(-np.asarray(C.get("GRID", 0.0), float) / 120))
    for etf, basket in BASKETS.items():
        v[etf] = sum(q * v[s] for s, q in basket.items())
    return v
