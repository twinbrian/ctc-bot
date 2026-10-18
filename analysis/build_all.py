"""Rebuild every cached artifact from the raw research data, in dependency order (~6 min).

    /opt/anaconda3/bin/python3 -I analysis/build_all.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import numpy as np
from scipy.optimize import least_squares

from load import tables, PHYS
from wind import site_features
from damage import storm_tensors, exposure, roll_season, company_damage


def step(name):
    print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)


step("1/7 parse research data -> cache.pkl")
T = tables(rebuild=True)
ST = T["storms"].sort_values(["season", "round"]).reset_index(drop=True)

step("2/7 per-site wind features (closure seconds) -> site_features.pkl")
site_features(T["obs"], step=0.5).to_pickle("site_features.pkl")

step("3/7 storm tensors (d/r per site, intensity, 0.5 s) -> storm_tensors.npz")
U, I = storm_tensors(T["obs"], list(zip(ST.season, ST["round"])))
np.savez_compressed("storm_tensors.npz", U=U, I=I)

step("4/7 fit site-level damage model -> damage_params.npy")
Y = ST[[f"d_{c}" for c in PHYS]].values
seasons, rounds = ST.season.values, ST["round"].values


def resid(p):
    pred = company_damage(roll_season(exposure(U, I, p[0], p[1]), np.exp(p[2:]), seasons, rounds)).values
    return (np.log1p(pred) - np.log1p(Y)).ravel()


fit = least_squares(resid, np.r_[1.0, 2.0, np.log(np.full(17, 0.01))], x_scale="jac", max_nfev=200)
np.save("damage_params.npy", fit.x)
print(f"    beta {fit.x[0]:.3f} gamma {fit.x[1]:.3f}")

step("5/7 physical nowcast grid -> nowcast_grid.pkl")
from nowcast import build as build_nowcast
F = build_nowcast().reset_index(drop=True)
F.to_pickle("nowcast_grid.pkl")

step("6/7 cross-validated in-storm model + backtest -> oos_pred.pkl")
from backtest import oos_predictions, simulate, SYMS
pred = oos_predictions(F)
pred.to_pickle("oos_pred.pkl")
scale = {s: pred[s].abs().quantile(0.8) for s in SYMS}
for cost in (0.0, 0.1, 0.2):
    R = simulate(F, pred, cost, scale)
    seas = R.groupby("season")[SYMS].sum().sum(1)
    print(f"    cost {cost:.1f}%: ${R[SYMS].sum(1).mean():,.0f}/storm, ${seas.mean():,.0f}/season, P(season<0) {(seas < 0).mean():.2f}")

step("7/7 figures")
import figures
figures.insurance(); figures.house_path(); figures.pattern_id()
step("done")
