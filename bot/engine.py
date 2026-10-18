"""Live state and features: turns feed messages into exactly the features the models were trained on.

Everything is keyed on storm age (seconds since `storm_formed`), never on the clock, so the bot works
with the server's timing (20 s pre-storm / 180 s aftermath) and the research timing (60 s / 240 s).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import numpy as np

from load import PHYS, STOCKS, SYMS, ISLANDS
from wind import SX, SY, THR
from damage import COMP_IDX, W
from value import values
from features import posterior, OUTLOOKS
from nowcast import projected

STEP = 0.5          # integration grid (s), same as the research tensors
N_SITES = len(SX)


@dataclass
class Storm:
    round: int
    name: str
    obs: dict = field(default_factory=dict)          # age -> dict(x, y, vx, vy, radius, intensity)
    official: list = field(default_factory=list)     # advisory messages, in order
    wire: list = field(default_factory=list)
    grid_age: float = -STEP                          # last integrated grid age
    cumX: np.ndarray = field(default_factory=lambda: np.zeros(N_SITES))
    cumC: np.ndarray = field(default_factory=lambda: np.zeros(N_SITES))
    hist: dict = field(default_factory=dict)         # grid age -> (cumX, cumC) snapshots at decision ages
    hist_mids: dict = field(default_factory=dict)    # decision age -> (mids now, mids 10 s earlier)
    dissipated: bool = False
    joined_late: bool = False
    gaps: list = field(default_factory=list)

    def add_obs(self, m: dict):
        a = m["age"]
        prev = max((k for k in self.obs if k < a), default=None)
        self.obs[a] = {k: m[k] for k in ("x", "y", "vx", "vy", "radius", "intensity")}
        if prev is None:
            if a == 0:
                self._integrate_grid(np.array([0.0]), self.obs[0], self.obs[0], 0, 0)
            else:
                self.joined_late = True
            return
        p, o = self.obs[prev], self.obs[a]
        if a - prev > 10:
            self.gaps.append((prev, a))                  # missed observations: the track is interpolated
        ages = np.arange(self.grid_age + STEP, a + 1e-9, STEP)
        if len(ages):
            self._integrate_grid(ages, p, o, prev, a)

    def _integrate_grid(self, ages, p, o, a0, a1):
        f = (ages - a0) / (a1 - a0) if a1 > a0 else np.zeros_like(ages)
        lerp = {k: p[k] + (o[k] - p[k]) * f for k in ("x", "y", "radius", "intensity")}
        u = np.hypot(lerp["x"][:, None] - SX, lerp["y"][:, None] - SY) / lerp["radius"][:, None]
        u = u.astype(np.float32).astype(np.float64)            # the research tensors stored u as float32
        I = lerp["intensity"][:, None]
        self.cumX += (np.where(u < 2, np.exp(-u ** 2), 0.0) * I).sum(0) * STEP
        self.cumC += ((I * np.exp(-u ** 2)) > THR).sum(0) * STEP
        self.grid_age = float(ages[-1])

    def snapshot(self, age):
        self.hist[age] = (self.cumX.copy(), self.cumC.copy())


@dataclass
class Season:
    """Damage carried between rounds: site-level damage and weighted outage seconds."""
    D_site: np.ndarray = field(default_factory=lambda: np.zeros(N_SITES))
    C_prev: dict = field(default_factory=lambda: {c: 0.0 for c in ("PORT", "GRID", "FERRY")})
    finals: dict = field(default_factory=dict)       # round -> damage_pct dict
    approximate: bool = False                        # True if a storm's track was missed

    def apply_final(self, rnd: int, damage_pct: dict, storm: Storm | None, v: np.ndarray):
        """Add a storm's final damage, split across sites in proportion to the model's site damage."""
        if rnd in self.finals:
            return
        self.finals[rnd] = damage_pct
        X = storm.cumX if storm is not None and not storm.joined_late else None
        if X is None:
            self.approximate = True
        a = (1 - self.D_site) * (1 - np.exp(-v * X)) if X is not None else (1 - self.D_site) * W
        for c in PHYS:
            idx = COMP_IDX[c]
            tot = (a[idx] * W[idx]).sum()
            a[idx] = a[idx] * (damage_pct[c] / 100) / tot if tot > 0 else 0
        self.D_site = np.minimum(self.D_site + a, 1.0)
        if X is not None:
            for c in self.C_prev:
                self.C_prev[c] += float((storm.cumC[COMP_IDX[c]] * W[COMP_IDX[c]]).sum())

    def company_damage(self):
        return {c: float((self.D_site[COMP_IDX[c]] * W[COMP_IDX[c]]).sum()) for c in PHYS}


def value_with(season: Season, consts, X, Cc, k):
    """Value of every symbol if this storm adds exposure X / closure Cc, then k more average storms."""
    D0, v, q = season.D_site, consts["v"], consts["q"]
    add = (1 - D0) * (1 - np.exp(-v * X))
    Ds = 1 - (1 - (D0 + add)) * (1 - q) ** k
    Dc = {c: float((Ds[COMP_IDX[c]] * W[COMP_IDX[c]]).sum()) for c in PHYS}
    C = {c: season.C_prev[c] + float((Cc[COMP_IDX[c]] * W[COMP_IDX[c]]).sum()) + k * consts["cfut"][c] for c in season.C_prev}
    return {s: float(x) for s, x in values(Dc, C).items()}


def exact_values(season: Season, storm: Storm | None):
    """Settlement values once every storm's final report is in. Damage = the sum of the reported company
    damage (exact); the per-site split caps sites at 100% and must not be used here. Outage from the tracks."""
    D = {c: sum(d[c] for d in season.finals.values()) / 100 for c in PHYS}
    return {s: float(x) for s, x in values(D, dict(season.C_prev)).items()}


def features(storm: Storm, season: Season, consts, age: int, H: dict, Hpre: dict, Hlag: dict) -> dict:
    """Feature vector per stock at decision age `age`, identical to analysis/nowcast.py + bot/train.design."""
    o = storm.obs[max(k for k in storm.obs if k <= age)]
    offs = [m for m in storm.official if m["age"] <= age]
    wires = [m for m in storm.wire if m["age"] <= age]
    Xs, Cs = storm.hist[age]
    if offs:
        Xp, Cp = projected(o, offs[-1]["cone"], age)
    else:
        Xp, Cp = np.zeros(N_SITES), np.zeros(N_SITES)
    vl = value_with(season, consts, Xs, Cs, 0)
    ve = value_with(season, consts, Xs + Xp, Cs + Cp, 3 - storm.round)
    post = posterior([("official", m["outlook"]) for m in offs] + [("wire", m["outlook"]) for m in wires])
    base = {"age": age, "round": storm.round, **{k: o[k] for k in ("x", "y", "vx", "vy", "radius", "intensity")},
            **{f"post_{k}": post[k] for k in OUTLOOKS},
            **{f"p_{i}": (offs[-1]["hurricane_conditions"][i] if offs else np.nan) for i in ISLANDS}}
    for sym in STOCKS:
        base[f"gexp_{sym}"] = ve[sym] - H[sym]
    rows = {}
    for s in STOCKS:
        rows[s] = [*base.values(), Hpre[s] - H[s], vl[s] - H[s], H[s] - Hlag[s]]
    names = list(base) + ["gpre", "glock", "mom"]
    return {"names": names, "rows": rows, "vexp": ve, "vlock": vl}
