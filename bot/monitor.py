"""Round monitor: flags bad results and diagnoses them automatically -- storm variance, a bot/execution problem,
or an accounting error.

    /opt/anaconda3/bin/python3 -I bot/monitor.py            # runs forever (checks every 15 s)
    /opt/anaconda3/bin/python3 -I bot/monitor.py --once     # report all completed rounds once and exit

During a round (live watch):
  - INTRA-ROUND DRAWDOWN every -$4k of the round's running PnL, and a SUDDEN DROP of -$3k within 60 s
  - BOT SILENT when the bot has not logged for 90 s (process dead, frozen, or the Mac asleep)
  - BOT EVENT for errors, rejected orders and disconnects
After every completed round:
  1. live PnL by symbol, exact: the bot's fills (reconciled with the exchange's positions) + settlement values
  2. testnet only: the same research season replayed with an honest model (never trained on it) and the
     settings that were live then = what the strategy is expected to do on that storm
  3. verdict:  OK | VARIANCE (the honest replay lost too) | LIVE-SPECIFIC (live much worse than the replay)
  4. diagnosis, automatic, for every flagged round:
       accounting  -- the monitor's PnL vs the bot's own exchange ledger (a mismatch = a monitor bug)
       storm       -- how sharply the track turned, islands hit, reported damage
       direction   -- how often the bot's position direction matched the move to round end
       execution   -- what was paid vs mid on each symbol
       leftovers   -- positions still open at the end of round 1/2 (the bot should be flat)
       bot events  -- restarts, rejects, disconnects, errors, unlogged fills
     and a one-line CAUSE.
  5. running live-minus-replay gap by symbol: a symbol persistently worse live = an execution problem
Every ALERT line goes to bot/logs/monitor.log, bot/logs/alerts.log and a macOS notification. Reports go to
bot/logs/round_reports.md / .jsonl. It only reports; it never changes the bot (tuning a strategy on a
handful of rounds is overfitting).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot"))
sys.path.insert(0, str(ROOT / "starter_kit"))

from history import load
from watch import Data, live_watch, records, _book, _feed, reconcile, exec_time

LOGS = ROOT / "bot" / "logs"
BAD = -5000.0                     # a round worse than this is investigated (same materiality as the live watch)
SY = ["PORT", "ISLE", "AQUA", "GRID", "FERRY", "RECON", "MAT", "GEN", "HAVEN", "REIN", "ARCH", "REBLD"]
HONEST = [(61, 160, ROOT / "bot" / "models_holdout_61_160.pkl"), (401, 500, ROOT / "bot" / "models_holdout_401_500.pkl"),
          (501, 600, ROOT / "bot" / "models_holdout100.pkl")]
ANOMALIES = ("ORDER REJECTED", "DISCONNECTED", "error", "SNIPE", "ANNOUNCEMENT - PAUSED", "PREDICTION REJECTED (implausible)",
             "JOINED LATE: outage history unknown -> no in-storm trading this season; final-report trade limited to outage-free symbols")
H = load(ROOT / "history")
# settings history: a round is compared with a replay that uses the settings live when it started.
# Append a line here whenever the bot's settings change.
SETTINGS = [
    ("2026-10-18 06:54", dict(BUFFER=0.001, EDGE_SHARE=0.5, allow_flip=True,  min_age=10)),
    ("2026-10-18 13:35", dict(BUFFER=0.01,  EDGE_SHARE=0.5, allow_flip=False, min_age=10)),
    ("2026-10-18 14:12", dict(BUFFER=0.01,  EDGE_SHARE=0.5, allow_flip=False, min_age=30)),
    ("2026-10-18 14:27", dict(BUFFER=0.01,  EDGE_SHARE=0.7, allow_flip=False, min_age=30)),
]


def alert(msg):
    """An ALERT: monitor.log (stdout), alerts.log and a macOS notification."""
    line = f"ALERT {time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOGS / "alerts.log", "a") as f:
        f.write(line + "\n")
    try:
        text = msg[:180].replace("\\", "").replace('"', "'")
        subprocess.run(["osascript", "-e", f'display notification "{text}" with title "Hurricane bot ALERT"'],
                       timeout=5, capture_output=True)
    except Exception:
        pass


def settings_at(started):
    cur = SETTINGS[0]
    for when, cfg in SETTINGS:
        if started >= when:
            cur = (when, cfg)
    return cur


ERRATIC = 1.8          # track surprise (radii) above this = an erratic storm (top ~20% of the live storms)


def storm_shape(obs, horizon=60):
    """Track surprise: the largest distance, in storm radii, between where the storm was and where its own motion
    pointed `horizon` s earlier. Live storms 71-82: surprise >= 1.9 in 7 rounds, and the honest replay lost in 6 of
    them (Wren 80-2 among them); below 1.1 nearly every round was positive. obs = {age: (x, y, vx, vy, radius, I)}."""
    ages = sorted(obs)
    if len(ages) < 3:
        return None
    best, at = 0.0, None
    for a in ages:
        k = bisect.bisect_left(ages, a + horizon)
        if k == len(ages) or ages[k] - (a + horizon) > 6:
            continue
        b = ages[k]; x, y, vx, vy, r, _ = obs[a]
        e = math.hypot(obs[b][0] - (x + vx * (b - a)), obs[b][1] - (y + vy * (b - a))) / r
        if e > best:
            best, at = e, a
    return {"surprise": round(best, 2), "surprise_from_age": at,
            "peak_intensity": round(max(o[5] for o in obs.values()), 2), "duration": ages[-1]}


def bot_ledger(ledger, rnd, name, t0, t1, w1, restarts):
    """The bot's own exchange-ledger PnL for the round, or None when a restart split the round."""
    if rnd in (1, 2):
        hit = [r for ts, e, r in ledger if e == "ROUND RESULT" and r["round"] == rnd and r["storm"] == name and t1 <= ts < t1 + 40]
        if not hit or any(t0 - 15 < x < t1 + 40 for x in restarts):
            return None
        return hit[0]["pnl"]
    sr = [r for ts, e, r in ledger if e == "SEASON RESULT" and w1 - 30 <= ts < w1 + 60 and r.get("pnl") is not None]
    r2 = [r for ts, e, r in ledger if e == "ROUND RESULT" and r["round"] == 2 and t0 <= ts < t0 + 40]
    if not sr or not r2 or any(t0 - 15 < x < w1 + 60 for x in restarts):
        return None
    return sr[0]["pnl"] - r2[0]["season_to_date"]


def live_rounds(since="run-20261018-065413"):
    """Completed live rounds with exact PnL by symbol and the inputs of the diagnosis, from the bot's logs."""
    runs = [f for f in sorted(glob.glob(str(LOGS / "run-*.jsonl"))) if Path(f).name >= since + ".jsonl"]
    timeline, forms, notes, restarts, decisions, ledger = [], [], collections.defaultdict(list), [], [], []
    for f in runs:
        recs = records(f)
        if not recs:
            continue
        t_start = recs[0]["ts"]; start = t_start + 5; restarts.append(t_start)
        snap = {}
        for r in recs:
            e = r["event"]
            if e == "fill":
                timeline.append((exec_time(r), "fill", (r["symbol"], r["qty"], r["price"])))
            elif e == "position change" and r["ts"] < t_start + 3 and r["frm"] == 0 and r["to"]:
                snap[r["symbol"]] = (r["to"], -r["cash_flow"] / r["to"])   # first positions read: exchange avg price
            elif r["ts"] < start:
                continue                                                  # catch-up replays recorded news
            elif e == "status": timeline.append((r["ts"], "status", r["positions"]))
            elif e == "new season": timeline.append((r["ts"], "reset", None))
            elif e == "storm_formed": forms.append(r["ts"])
            elif e == "decision": decisions.append((r["ts"], r.get("goals") or {}))
            elif e in ("ROUND RESULT", "SEASON RESULT"): ledger.append((r["ts"], e, r))
            if e in ANOMALIES: notes[e].append(r["ts"])
        timeline.append((t_start + 1.0, "startup", snap))
    ident, storms = [], collections.defaultdict(lambda: {"obs": {}, "impacts": set(), "reports": {}})
    for f in sorted(glob.glob(str(LOGS / "feed-*.jsonl"))):
        for m in records(f, _feed):
            t = m["type"]
            if t == "storm_formed":
                ident.append((m["received_at"], m["time"][:10], m["round"], m["storm"]))
                continue
            k = (m["time"][:10], m["round"], m["storm"])
            if t == "storm":
                storms[k]["obs"][m["age"]] = (m["x"], m["y"], m["vx"], m["vy"], m["radius"], m["intensity"])
            elif t == "impact":
                storms[k]["impacts"].add(m["island"])
            elif t == "damage_report":
                storms[k]["reports"][m["stage"]] = (m["damage_pct"], m.get("industry_loss_musd"))
    ident.sort(); it = [x[0] for x in ident]
    forms = sorted(set(forms))
    books = collections.defaultdict(list)
    for f in sorted(glob.glob(str(LOGS / "books-*.jsonl"))):
        for s, ts, m in records(f, _book):
            books[s].append((ts, m))
    for v in books.values():
        v.sort()
    bt = {s: [x[0] for x in v] for s, v in books.items()}
    mid = lambda s, t: books[s][max(bisect.bisect_right(bt[s], t) - 1, 0)][1] if books.get(s) else 0.0
    fills = reconcile(timeline, mid)
    synthetic = [ts for ts, *_, syn in fills if syn]
    # a season's rounds run from its reset to the next reset: pre-storm trades (between the reset and the first
    # storm) belong to round 1 of the NEW season, not to round 3 of the old one (which settles at the reset)
    resets = sorted({round(ts) for ts, kind, _ in timeline if kind == "reset"})
    out, pos = [], collections.defaultdict(int)
    for k, t0 in enumerate(forms):
        if k + 1 >= len(forms):
            break                                                  # the current round is not complete
        t1 = forms[k + 1]
        j = bisect.bisect_right(it, t0 + 3) - 1
        _, date, rnd, name = ident[j]
        se = H.season_of(date)
        if se is None:
            continue
        w0, w1 = t0, t1
        if rnd == 1:
            pos = collections.defaultdict(int)
            w0 = min([x for x in resets if (forms[k - 1] if k else 0) < x <= t0], default=t0)
        if rnd == 3:
            w1 = min([x for x in resets if t0 < x < t1], default=t1)
        start = dict(pos); cash = collections.defaultdict(float)
        pre_cash, pre_pos, exe = 0.0, collections.defaultdict(int), collections.defaultdict(float)
        for ts, s, q, px, syn in fills:
            if w0 <= ts < w1:
                cash[s] -= q * px; pos[s] += q
                if ts < t0:
                    pre_cash -= q * px; pre_pos[s] += q
                if not syn:
                    exe[s] += q * (px - mid(s, ts - 2.5))            # paid vs mid just before the fill
        endv = {s: se.settlement[s] if rnd == 3 else mid(s, t1) for s in SY}
        leftover = {s: q for s, q in pos.items() if q} if rnd != 3 else {}
        end = {s: pos[s] * endv[s] for s in SY}
        if rnd == 3:
            pos = collections.defaultdict(int)
        by_sym = {s: cash[s] + end[s] - start.get(s, 0) * mid(s, t0) for s in SY}
        pre = pre_cash + sum(q * mid(s, t0) for s, q in pre_pos.items())
        hits = collections.defaultdict(lambda: [0, 0])
        for ts, goals in decisions:
            if t0 <= ts < w1:
                for s, g in goals.items():
                    if g[0] and s in endv:
                        hits[s][1] += 1; hits[s][0] += (g[0] > 0) == (endv[s] > mid(s, ts))
        sk = storms.get((date, rnd, name))
        anom = {e: sum(w0 <= x < w1 for x in v) for e, v in notes.items()}
        anom["restart"] = sum(w0 < x < w1 for x in restarts)
        anom["unlogged fills (reconciled)"] = sum(w0 <= x < w1 for x in synthetic)
        out.append({"key": f"{se.number}-{rnd}", "season": se.number, "round": rnd, "storm": name, "date": date,
                    "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(t0)), "pnl": sum(by_sym.values()),
                    "t_end": w1, "by_symbol": by_sym, "anomalies": {k_: v for k_, v in anom.items() if v},
                    "diag": {"core_pnl": sum(by_sym.values()) - pre, "pre_storm_pnl": pre,
                             "bot_ledger": bot_ledger(ledger, rnd, name, t0, t1, w1, restarts),
                             "exec_cost": dict(exe), "leftover": leftover, "carried_in": {s_: q for s_, q in start.items() if q}, "hits": {s: tuple(v) for s, v in hits.items()},
                             "move": {s: (mid(s, t0), endv[s]) for s in SY},
                             "storm": storm_shape(sk["obs"]) if sk else None,
                             "islands": sorted(sk["impacts"]) if sk else [],
                             "report": (sk["reports"].get("final") or sk["reports"].get("preliminary")) if sk else None}})
    return out


_replay_cache = {}


def honest_replay(season, epoch):
    """Per-round, per-symbol PnL of the same research season with an honest model and the settings of `epoch`."""
    key = (season, epoch[0])
    if key in _replay_cache:
        return _replay_cache[key]
    model = next((m for a, b, m in HONEST if a <= season <= b and m.exists()), None)
    if model is None:
        return None
    import strategy as S
    from replay import replay_season
    from book_profile import build
    cfg = epoch[1]
    S.BUFFER, S.EDGE_SHARE = cfg["BUFFER"], cfg["EDGE_SHARE"]
    def tweak(st):
        st.allow_flip, st.min_age = cfg["allow_flip"], cfg["min_age"]
    extra = {"PORT": 0.002, "FERRY": 0.0012, "AQUA": 0.0021, "MAT": 0.0007, "GEN": 0.0005}   # live-calibrated costs
    rows = []
    replay_season(season, model, 0.1, profile=build(), per_symbol=rows, extra=extra, tweak=tweak)
    res = {}
    for r in rows:
        res.setdefault(r["round"], {})[r["symbol"]] = r["pnl"]
    _replay_cache[key] = res
    _path_cache[key] = _replay_paths(season, replay_season.last_fills)
    return res


_path_cache = {}


def _replay_paths(n, fills):
    """Round PnL of a replay by storm age (5 s steps): fills + positions at the house mid, minus the value at formation."""
    import csv
    from replay import real_t
    folder = ROOT / "history" / f"season-{n:03d}"
    msgs = [json.loads(l) for l in open(folder / "feed.jsonl")]
    forms = {m["round"]: (m["round"] - 1) * 10000 + real_t(m["time"]) for m in msgs if m["type"] == "storm_formed"}
    rows = sorted(((int(r["round"]) - 1) * 10000 + real_t(r["time"]), {x: float(r[x]) for x in SY})
                  for r in csv.DictReader(open(folder / "market.csv")))
    times = [x[0] for x in rows]
    fl = sorted((clk, x, q, px) for x, q, px, _, clk, _ in fills)
    def value(c):
        cash, pos = 0.0, collections.defaultdict(int)
        for clk, x, q, px in fl:
            if clk > c:
                break
            cash -= q * px; pos[x] += q
        m = rows[max(bisect.bisect_right(times, c) - 1, 0)][1]
        return cash + sum(q * m[x] for x, q in pos.items())
    out = {}
    for r, tf in forms.items():
        base = value(tf)
        out[r] = [value(tf + a) - base for a in range(0, 545, 5)]
    return out


def replay_at(rd, age):
    """Honest replay round PnL of the same storm at this storm age (testnet seasons only), or None."""
    date = rd["key"][0]
    se = H.season_of(date) if date else None
    if se is None:
        return None
    epoch = settings_at(time.strftime("%Y-%m-%d %H:%M", time.localtime(rd["t_form"])))
    if honest_replay(se.number, epoch) is None:
        return None
    path = _path_cache[(se.number, epoch[0])].get(rd["round"])
    if not path:
        return None
    return path[min(max(int(age // 5), 0), len(path) - 1)]


def verdict(live_pnl, rep_pnl):
    if rep_pnl is None:
        return "NO REPLAY (new storm): judge by anomalies and symbols"
    tol = max(2500.0, 0.5 * abs(rep_pnl))
    if live_pnl - rep_pnl < -tol:
        return "LIVE-SPECIFIC: live much worse than the honest replay of the same storm -- investigate"
    if live_pnl < BAD and rep_pnl < BAD / 2:
        return "VARIANCE: the honest replay of this storm lost too (the storm, not the bot)"
    if live_pnl < BAD:
        return "MIXED: bad round, replay lost less -- check the symbols with the largest gaps"
    return "OK"


def diagnose(r, v, rep_total, sym_gap):
    """Automatic analysis of a round: findings (list of lines) and a one-line cause."""
    d, lines, problems = r["diag"], [], []
    acct_bad = False
    if d["bot_ledger"] is not None and (d["leftover"] or d["carried_in"]):
        lines.append(f"accounting: monitor ${d['core_pnl']:,.0f} vs bot's exchange ledger ${d['bot_ledger']:,.0f} -- not comparable "
                     f"(positions crossed the round boundary; the two mark them at different moments)")
    elif d["bot_ledger"] is not None:
        diff = d["core_pnl"] - d["bot_ledger"]
        acct_bad = abs(diff) > max(750.0, 0.1 * abs(d["bot_ledger"]))
        lines.append(f"accounting: monitor ${d['core_pnl']:,.0f} vs bot's exchange ledger ${d['bot_ledger']:,.0f} "
                     f"({'MISMATCH' if acct_bad else 'agree'})" + (f"; pre-storm trades ${d['pre_storm_pnl']:,.0f}" if abs(d['pre_storm_pnl']) > 1 else ""))
    else:
        lines.append("accounting: no bot ledger for this round (a restart split it); positions reconciled with the exchange")
    sh = d["storm"]
    if sh:
        rep = d["report"]
        lines.append(f"storm: track surprise {sh['surprise']} radii from age {sh['surprise_from_age']} "
                     f"({'ERRATIC' if sh['surprise'] >= ERRATIC else 'normal'}), "
                     f"peak intensity {sh['peak_intensity']}, islands hit {d['islands'] or 'none'}"
                     + (f", damage {rep[0]}, insured loss ${rep[1]}M" if rep else ""))
    worst = sorted(r["by_symbol"].items(), key=lambda kv: kv[1])[:3]
    mv = "; ".join(f"{s} ${x:,.0f} (price {d['move'][s][0]:.2f} -> {d['move'][s][1]:.2f}, "
                   f"exec ${d['exec_cost'].get(s, 0):,.0f}"
                   + (f", target direction right {d['hits'][s][0]}/{d['hits'][s][1]} decisions" if s in d["hits"] else "") + ")"
                   for s, x in worst if x < 0)
    if mv:
        lines.append(f"worst symbols: {mv}")
    n_hit = sum(h for h, n in d["hits"].values()); n_dec = sum(n for h, n in d["hits"].values())
    if n_dec:
        lines.append(f"direction: the bot's target direction matched the move to round end in {n_hit}/{n_dec} symbol-decisions "
                     f"({100 * n_hit / n_dec:.0f}%)")
    exe = sum(d["exec_cost"].values())
    lines.append(f"execution: paid ${exe:,.0f} vs mid in total")
    if sym_gap:
        lines.append("live minus replay by symbol (worst): " + ", ".join(f"{s} ${g:,.0f}" for s, g in sorted(sym_gap.items(), key=lambda kv: kv[1])[:3]))
    if d["leftover"]:
        problems.append(f"positions left open at the end of round {r['round']} (should be flat): {d['leftover']}")
    ev = {k: n for k, n in r["anomalies"].items() if k not in ("ANNOUNCEMENT - PAUSED",)}
    if ev:
        problems.append(f"bot events {ev}")
    if exe > max(1500.0, 0.5 * abs(r["pnl"])):
        problems.append(f"execution cost ${exe:,.0f}")
    lines += [f"check: {p}" for p in problems]
    turn = (f"; erratic storm: it ended up {sh['surprise']} radii from where its motion pointed at age {sh['surprise_from_age']}"
            if sh and sh["surprise"] >= ERRATIC else "")
    hit = f"; the bot's target direction was right in only {100 * n_hit / n_dec:.0f}% of decisions" if n_dec and n_hit / n_dec < 0.5 else ""
    if acct_bad:
        cause = "ACCOUNTING: the monitor disagrees with the bot's exchange ledger -- a monitor bug, not necessarily a trading loss"
    elif v.startswith("LIVE-SPECIFIC"):
        cause = ("BOT/EXECUTION: " + "; ".join(problems)) if problems else \
                "UNEXPLAINED: live much worse than the replay with no bot anomaly -- needs a manual look"
    elif v.startswith(("VARIANCE", "MIXED")):
        cause = f"STORM (variance): the honest replay of the same storm made ${rep_total:,.0f}{turn}{hit}"
        if problems:
            cause += " | also check: " + "; ".join(problems)
    elif v.startswith("NO REPLAY"):
        cause = ("check: " + "; ".join(problems)) if problems else f"no replay; no bot anomaly{turn}{hit}"
    else:
        cause = "OK" + (" | check: " + "; ".join(problems) if problems else "")
    return lines, cause


def report(rounds, seen, st):
    md, new = [], []
    for r in rounds:
        if r["key"] in seen:
            continue
        epoch = settings_at(r["started"])
        rep = honest_replay(r["season"], epoch)
        rp = rep.get(r["round"]) if rep else None
        rep_total = sum(rp.values()) if rp else None
        v = verdict(r["pnl"], rep_total)
        sym_gap = {s: r["by_symbol"][s] - rp.get(s, 0.0) for s in SY} if rp else {}
        flagged = r["pnl"] < BAD or not v.startswith(("OK", "NO REPLAY")) or bool(r["diag"]["leftover"])
        lines, cause = diagnose(r, v, rep_total, sym_gap)
        if r["diag"]["bot_ledger"] is not None and cause.startswith("ACCOUNTING"):
            flagged = True
        worst = sorted(r["by_symbol"].items(), key=lambda kv: kv[1])[:3]
        rec = {"round": r["key"], "storm": r["storm"], "started": r["started"], "settings_since": epoch[0], "live_pnl": round(r["pnl"]),
               "honest_replay_pnl": None if rep_total is None else round(rep_total), "verdict": v, "cause": cause,
               "worst_symbols": {s: round(x) for s, x in worst}, "anomalies": r["anomalies"],
               "live_minus_replay": {s: round(g) for s, g in sym_gap.items()}, "diagnosis": lines,
               "bot_ledger": None if r["diag"]["bot_ledger"] is None else round(r["diag"]["bot_ledger"])}
        new.append(rec); seen.add(r["key"])
        line = (f"{r['started']}  season {r['season']} r{r['round']} {r['storm']:6s} live ${r['pnl']:8,.0f}  "
                f"replay {'   n/a' if rep_total is None else f'${rep_total:8,.0f}'}  {v.split(':')[0]}")
        if flagged:
            msg = "ROUND " + line + f" | CAUSE: {cause}"
            if time.time() - r["t_end"] < 1800:
                alert(msg)
            else:
                print("PAST " + msg, flush=True)                 # finished before the monitor started: no alert
            for l in lines:
                print("      " + l, flush=True)
        else:
            print(line, flush=True)
        md.append(f"### {r['started']} — season {r['season']} round {r['round']} ({r['storm']})\n"
                  f"- live PnL **${r['pnl']:,.0f}**; honest replay of the same storm: "
                  f"{'n/a' if rep_total is None else f'${rep_total:,.0f}'}\n- verdict: **{v}**\n- cause: **{cause}**\n"
                  + "".join(f"- {l}\n" for l in lines))
    if new:
        with open(LOGS / "round_reports.jsonl", "a") as f:
            for rec in new:
                f.write(json.dumps(rec) + "\n")
        with open(LOGS / "round_reports.md", "a") as f:
            f.write("\n".join(md) + "\n")
        gaps = collections.defaultdict(list)
        for l in open(LOGS / "round_reports.jsonl"):
            for s, g in (json.loads(l).get("live_minus_replay") or {}).items():
                gaps[s].append(g)
        persist = {s: sum(g) for s, g in gaps.items() if len(g) >= 5 and sum(g) < -3000}
        for s, tot in persist.items():
            step = int(-tot // 2500)
            msg = f"persistent live underperformance vs replay: {s} ${tot:,.0f} over {len(gaps[s])} rounds"
            if step > st.setdefault("gap_step", {}).get(s, 0):
                st["gap_step"][s] = step
                alert(msg)
            else:
                print("NOTE " + msg, flush=True)


def emit_alert(kind, msg, lines):
    """A live alert: ALERT line + its diagnosis (monitor.log, alerts.log, live_alerts.md) and a macOS notification."""
    alert(f"{kind} {msg}")
    for l in lines:
        print("      " + l, flush=True)
    with open(LOGS / "alerts.log", "a") as f:
        f.write("".join("      " + l + "\n" for l in lines))
    with open(LOGS / "live_alerts.md", "a") as f:
        f.write(f"### {time.strftime('%Y-%m-%d %H:%M:%S')} {kind}\n{msg}\n" + "".join(f"- {l}\n" for l in lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="report completed rounds once and exit")
    ap.add_argument("--replay", metavar="HH:MM", help="validation: run the live watch over today's logs from HH:MM, print alerts")
    args = ap.parse_args()
    if args.replay:
        start = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + args.replay, "%Y-%m-%d %H:%M"))
        out = []
        D = Data(); D.replay_hook = replay_at; D.refresh()
        st, now = {}, start
        while now <= time.time() - 10:
            live_watch(D, st, now, lambda k, m, l: out.append((now, k, m, l)))
            now += 5.0
        for t, k, m, l in out:
            print(f"[{time.strftime('%H:%M:%S', time.localtime(t))}] {k} {m}")
            for x in l:
                print("      " + x)
        print(f"{len(out)} alerts")
        return
    state = LOGS / "monitor_state.json"
    st = json.loads(state.read_text()) if state.exists() and not args.once else {}
    seen = set(st.get("seen", []))
    live = {"gap_step": st.get("gap_step", {})}
    D = Data(); D.replay_hook = replay_at
    wst = {}
    last_sig, last_err = None, 0.0
    while True:
        try:
            D.refresh()
            if len(D.forms) != last_sig:                           # a round may have completed
                report(live_rounds(), seen, live); last_sig = len(D.forms)
            if not args.once:
                live_watch(D, wst, time.time(), emit_alert)
                state.write_text(json.dumps({"seen": sorted(seen), "gap_step": live["gap_step"]}))
        except Exception as exc:
            print("monitor error:", repr(exc), flush=True)
            if time.time() - last_err > 600:
                last_err = time.time(); alert(f"MONITOR ERROR {exc!r}")
        if args.once:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
