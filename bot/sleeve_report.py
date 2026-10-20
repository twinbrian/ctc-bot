"""Live report on the formation sleeve (strategy.py, FORM_SIZE): is it doing what the backtest said?

    OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3 -I bot/sleeve_report.py [--since "YYYY-MM-DD HH:MM"]

Per round where the sleeve ran:
  - sleeve PnL: its own fills from its entry to storm age 30, anything left marked at mid at age 30
  - share of the target filled before the storm formed, fill distance from mid (entry and exit), leftover at age 30
  - price impact: live mid minus the house's own price (research market.csv of the season the testnet replays) in the
    entry window, in our direction (our buying should not push the price much: backtest allowance ~0.14% at 20%)
The open is timed as storm_formed - 60 s (formation is market 10:00): in round 1 of a testnet season the clock sits at
09:30:00 for ~120 s, so the first 09:30 snapshot is not the open.

Backtest of this code on 94 recorded live rounds (scratchpad livebook_backtest.py, 55 orders/min): sleeve's own PnL
+$311/round (se 38), positive in 82%, worst -$392, 100% filled before formation, ~3% left at age 28, ~27 orders/round.
Stop rule (agreed before going live): switch it off ({"sleeve": 0} in bot/control.json) if, after 15+ live rounds, its
mean PnL is below 0, or if any single round loses more than $1,000 without an explanation that is not the sleeve's.
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot")); sys.path.insert(0, str(ROOT / "starter_kit"))
import watch as W
from history import load
from replay import real_t

H = load(ROOT / "history")
BACKTEST = {"mean": 311.0, "se": 38.0, "positive": 0.82, "worst": -392.0}
STOP_ROUNDS, ROUND_BAD = 15, -1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    args = ap.parse_args()
    since = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M")) if args.since else 0.0
    D = W.Data(); D.refresh()
    entries = [(ts, r) for ts, e, r in D.events if ts >= since and e == "FORMATION SLEEVE entry"]

    def house(date, rnd, s, sec):
        se = H.season_of(date)
        if se is None:
            return None
        rows = house.cache.get(se.number)
        if rows is None:
            rows = house.cache[se.number] = [(int(r["round"]), real_t(r["time"]), r) for r in csv.DictReader(open(ROOT / "history" / f"season-{se.number:03d}" / "market.csv"))]
        v = [float(r[s]) for rr, t, r in rows if rr == rnd and abs(t - sec) <= 2]
        return statistics.mean(v) if v else None
    house.cache = {}

    print(f"FORMATION SLEEVE: {len(entries)} rounds with the sleeve")
    rows = []
    for ts, r in entries:
        idt = next((x for x in D.ident if ts < x[0] < ts + 60 and x[2] == r.get("round")), None)
        if idt is None or time.time() < idt[0] + 40:
            continue
        tf, date, rnd = idt[0], idt[1], idt[2]
        t_open = tf - 60
        targets = r.get("targets") or {}
        fl = [x for x in D.fills if ts - 1 <= x[0] < tf + 30 and x[2] in targets]
        pre = [x for x in fl if x[0] < tf]
        filled = {s: sum(q for e, lt, x, q, p in pre if x == s) for s in targets}
        share = sum(min(abs(filled[s]), abs(t)) for s, t in targets.items()) / max(sum(abs(t) for t in targets.values()), 1)
        dist_in = [abs(p / D.mid(x, e - 0.5) - 1) for e, lt, x, q, p in pre if D.mid(x, e - 0.5)]
        dist_out = [abs(p / D.mid(x, e - 0.5) - 1) for e, lt, x, q, p in fl if e >= tf and D.mid(x, e - 0.5)]
        cash = -sum(q * p for e, lt, x, q, p in fl)
        pos = collections.Counter()
        for e, lt, x, q, p in fl:
            pos[x] += q
        pnl = cash + sum(q * D.mid(x, tf + 30) for x, q in pos.items())
        left = {x: q for x, q in pos.items() if q}
        imp = []
        for s in targets:
            for sec in range(32, 57, 8):
                h = house(date, rnd, s, sec)
                if h:
                    imp.append((D.mid(s, t_open + sec) / h - 1) * (1 if targets[s] > 0 else -1))
        rows.append((pnl, share, statistics.mean(dist_in) if dist_in else 0, statistics.mean(dist_out) if dist_out else 0,
                     statistics.mean(imp) if imp else 0, left, date, rnd))
        flag = "  <-- LOSS BEYOND THE STOP LEVEL: diagnose" if pnl < ROUND_BAD else ""
        print(f"  {time.strftime('%H:%M', time.localtime(tf))} {date} r{rnd}: sleeve PnL ${pnl:7,.0f} | filled {share:.0%} of target before "
              f"formation | fills from mid: entry {rows[-1][2]:.2%}, exit {rows[-1][3]:.2%} | live mid vs house in the "
              f"entry window, in our direction: {rows[-1][4]:+.2%} | leftover {left or 'none'}{flag}")
    if rows:
        p = [x[0] for x in rows]
        se = statistics.stdev(p) / len(p) ** 0.5 if len(p) > 1 else float("nan")
        print(f"  => live: mean ${statistics.mean(p):,.0f}/round (se {se:,.0f}), positive {sum(x > 0 for x in p) / len(p):.0%}, "
              f"worst ${min(p):,.0f}, fill share {statistics.mean(x[1] for x in rows):.0%}, live-vs-house drift in our direction "
              f"{statistics.mean(x[4] for x in rows):+.2%}")
        print(f"     backtest: mean ${BACKTEST['mean']:,.0f}/round (se {BACKTEST['se']:,.0f}), positive {BACKTEST['positive']:.0%}, "
              f"worst ${BACKTEST['worst']:,.0f}, fill share 100%")
        if len(p) >= STOP_ROUNDS and statistics.mean(p) < 0:
            print(f"  STOP RULE: {len(p)} rounds with a negative mean -> switch the sleeve off: {{\"sleeve\": 0}} in bot/control.json")
        elif len(p) < STOP_ROUNDS:
            print(f"  ({len(p)} of {STOP_ROUNDS} rounds needed before the stop rule applies)")


if __name__ == "__main__":
    main()
