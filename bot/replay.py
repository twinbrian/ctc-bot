"""Replay research seasons through the live strategy code (no exchange needed).

    /opt/anaconda3/bin/python3 -I bot/replay.py --seasons 501-600 --models bot/models_holdout100.pkl --half-spread 0.1

Quotes are the house price +/- half-spread (% of price); orders fill instantly at the quote.
Checks that the live feature code matches the research features, then reports PnL per season.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot"))

import numpy as np
import pandas as pd

from strategy import Strategy
from broker import SimBroker
from engine import STOCKS, SYMS

LIMITS = json.loads((ROOT / "case.json").read_text())["limits"]


def real_t(time: str) -> float:
    t = datetime.fromisoformat(time)
    return ((t.hour - 9) * 3600 + (t.minute - 30) * 60 + t.second) / 30.0


def replay_season(n: int, models: Path, hs: float, verbose=False, profile=None, reflatten=True, hold_rounds=(3,), per_round=None,
                  front_run=False, per_symbol=None, extra=None, tweak=None):
    folder = ROOT / "history" / f"season-{n:03d}"
    msgs = [json.loads(l) for l in (folder / "feed.jsonl").open()]
    rows = list(csv.DictReader((folder / "market.csv").open()))
    settle = json.loads((folder / "settlement.json").read_text())
    events = [((int(r["round"]) - 1) * 10000 + real_t(r["time"]), 0, i, ("q", r)) for i, r in enumerate(rows)]
    events += [((m["round"] - 1) * 10000 + real_t(m["time"]), 1, m["seq"], ("m", m)) for m in msgs]
    events.sort(key=lambda e: e[:3])
    now = {"t": 0.0}
    logs = []
    log = (lambda ev, **kw: print(ev, kw)) if verbose else (lambda ev, **kw: logs.append((ev, kw)))
    strat = Strategy(models, LIMITS, None, log, clock=lambda: now["t"], persist=False, reflatten=reflatten)
    broker = SimBroker(strat.quotes, log, LIMITS, clock=lambda: now["t"], profile=profile, extra=extra)
    strat.broker = broker
    strat.hold_rounds = set(hold_rounds)
    strat.front_run = front_run
    if tweak:
        tweak(strat)
    eq_before_final = None
    marks, last_round, round_mids = {}, 1, {}
    for clk, kind, _, (k, x) in events:
        now["t"] = clk
        if k == "q":
            r = int(x["round"])
            if r != last_round:                              # mark equity at each round's close
                marks[last_round] = broker.cash + sum(broker.pos.get(s, 0) * sum(strat.quotes[s]) / 2 for s in SYMS)
                round_mids[last_round] = {s: sum(strat.quotes[s]) / 2 for s in SYMS}
                last_round = r
            for s in SYMS:
                mid = float(x[s])
                h = profile[s][0][0] if profile else hs / 100          # touch offset from the modelled book
                strat.on_quote(s, mid * (1 - h), mid * (1 + h))
            broker.tick()
        else:
            if x["type"] == "damage_report" and x["stage"] == "final" and x["round"] == 3:
                eq_before_final = broker.cash + sum(broker.pos.get(s, 0) * sum(strat.quotes[s]) / 2 for s in SYMS)
            strat.on_news(x)
    eq = broker.cash + sum(broker.pos.get(s, 0) * settle[s] for s in SYMS)
    marks[3] = eq
    if per_symbol is not None:                               # PnL by (round, symbol): fills + marks at round ends
        round_mids[3] = settle
        pos = {x: 0 for x in SYMS}
        for r in (1, 2, 3):
            start = dict(pos)
            cash = {x: 0.0 for x in SYMS}
            for (x, q, px, reason, clk, _) in broker.fills:
                if int(clk // 10000) + 1 == r:
                    cash[x] -= q * px; pos[x] += q
            for x in SYMS:
                m0 = round_mids.get(r - 1, {}).get(x, 0.0)
                per_symbol.append({"season": n, "round": r, "symbol": x,
                                   "pnl": cash[x] + pos[x] * round_mids[r][x] - start[x] * m0})
    if per_round is not None:
        per_round.append({"season": n, "r1": marks.get(1, 0), "r2": marks.get(2, 0) - marks.get(1, 0), "r3": eq - marks.get(2, 0)})
    replay_season.last_fills = broker.fills
    return eq, eq_before_final, strat.features_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="501-600")
    ap.add_argument("--models", default=str(ROOT / "bot" / "models_holdout100.pkl"))
    ap.add_argument("--half-spread", type=float, default=0.1, help="% of price")
    ap.add_argument("--parity", action="store_true", help="compare features with analysis/nowcast_grid.pkl")
    ap.add_argument("--book", action="store_true", help="fill against the recorded testnet book depth (bot/logs/books-*)")
    args = ap.parse_args()
    a, b = map(int, args.seasons.split("-"))
    grid = pd.read_pickle(ROOT / "analysis" / "nowcast_grid.pkl") if args.parity else None
    if grid is not None:
        sys.path.insert(0, str(ROOT / "bot"))
        import os; cwd = os.getcwd(); os.chdir(ROOT / "analysis")
        from train import design
        os.chdir(cwd)
    profile = None
    if args.book:
        from book_profile import build as build_profile
        profile = build_profile()
    res, diffs = [], []
    for n in range(a, b + 1):
        eq, eqb, feats = replay_season(n, Path(args.models), args.half_spread, profile=profile)
        res.append({"season": n, "pnl": eq, "in_storm": eqb, "snipe": eq - (eqb or 0)})
        if grid is not None:
            g = grid[grid.season == n]
            for f in feats:
                row = g[(g["round"] == f["round"]) & (g.t == 60 + f["age"])]
                if len(row):
                    ref = design(row, "PORT").values[0]
                    live = np.array(f["PORT"], dtype=float)
                    diffs.append(np.nan_to_num(np.abs(live - ref), nan=0.0))
        print(f"season {n}: pnl ${eq:,.0f}  (in-storm ${eqb or 0:,.0f}, final snipe ${eq - (eqb or 0):,.0f})", flush=True)
    R = pd.DataFrame(res)
    print(f"\n{len(R)} seasons, {'recorded book depth' if profile else f'half-spread {args.half_spread}%'}: mean ${R.pnl.mean():,.0f}/season, "
          f"P(season>0) {(R.pnl > 0).mean():.2f}; in-storm ${R.in_storm.mean():,.0f}, snipe ${R.snipe.mean():,.0f}")
    if diffs:
        D = np.array(diffs)
        names = design(grid.head(1), "PORT").columns
        worst = pd.Series(D.max(0), names).sort_values(ascending=False)
        print("feature parity (live engine vs research), max |diff| over all decisions, worst 8:")
        print(worst.head(8).round(4).to_string())


if __name__ == "__main__":
    main()
