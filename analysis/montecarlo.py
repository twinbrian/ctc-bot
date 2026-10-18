"""Monte Carlo expectation of settlement given the season state, by resampling historical storms.

State: site damage D_s (17 sites, fraction of original), outage C_c so far (weighted closed seconds),
       number of storms still to come.
Each future storm is a historical storm j: site exposure X_j and closed seconds c_j.
    site damage added = (1 - D_s) * (1 - exp(-v_s * X_js * exp(eta_c))),  eta_c ~ N(MU, SIG^2)
    (noise on the hazard, shared by a company's sites: matches the residual-vs-size profile and the L tail)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from load import PHYS, STOCKS, SYMS, tables
from wind import SITES, THR
from damage import exposure, roll_season, company_damage, COMP_IDX, W
from value import values

SIG, MU = 0.33, 0.0
COMP_OF_SITE = np.array([PHYS.index(c) for c in SITES.company.values])


class Library:
    def __init__(self):
        T = tables()
        self.storms = T["storms"].sort_values(["season", "round"]).reset_index(drop=True)
        z = np.load("storm_tensors.npz")
        U, I = z["U"], z["I"]
        p = np.load("damage_params.npy")
        self.beta, self.gamma, self.v = p[0], p[1], np.exp(p[2:])
        self.X = exposure(U, I, self.beta, self.gamma)                       # (1800, 17)
        wind = I[:, :, None] * np.exp(-U.astype(np.float64) ** 2)
        self.closed = (wind > THR).sum(1) * 0.5                                # (1800, 17) seconds
        self.n = len(self.X)

    def site_state(self):
        """Per-season site damage after each round, apportioning reported company damage to sites."""
        st = self.storms
        Y = st[[f"d_{c}" for c in PHYS]].values / 100
        out = np.zeros((self.n, 17))
        D = None
        for i, (s, r) in enumerate(zip(st.season.values, st["round"].values)):
            if r == 1:
                D = np.zeros(17)
            a = (1 - D) * (1 - np.exp(-self.v * self.X[i]))
            for k, c in enumerate(PHYS):
                idx = COMP_IDX[c]
                tot = (a[idx] * W[idx]).sum()
                if tot > 0:
                    a[idx] *= Y[i, k] / tot          # scale model split to match the reported company damage
                else:
                    a[idx] = 0
            D = np.minimum(D + a, 1.0)
            out[i] = D
        return out


def simulate(lib: Library, D0, C0, k: int, n: int = 20000, rng=None):
    """Distribution of settlement values given site damage D0 (17,), outage C0 (dict), k storms left."""
    rng = rng or np.random.default_rng()
    D = np.tile(np.asarray(D0, float), (n, 1))
    C = {c: np.full(n, C0.get(c, 0.0)) for c in ("PORT", "GRID", "FERRY")}
    for _ in range(k):
        j = rng.integers(0, lib.n, n)
        eta = np.exp(rng.normal(MU, SIG, (n, len(PHYS))))[:, COMP_OF_SITE]
        D = D + (1 - D) * (1 - np.exp(-lib.v * lib.X[j] * eta))
        for c in C:
            idx = COMP_IDX[c]
            C[c] += (lib.closed[j][:, idx] * W[idx]).sum(1)
    Dc = {c: (D[:, COMP_IDX[c]] * W[COMP_IDX[c]]).sum(1) for c in PHYS}
    return values(Dc, C)


def expect(lib, D0, C0, k, n=20000, rng=None):
    v = simulate(lib, D0, C0, k, n, rng)
    return {s: float(np.mean(v[s])) for s in SYMS}
