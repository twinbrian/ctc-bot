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
# formation sleeve (live since 2026-10-19). Backtest on 94 recorded live rounds: its own PnL +$311/round (se 38),
# worst round -$392. Alert on a round below SLEEVE_BAD, and when SLEEVE_MIN_N+ live rounds average below 0 (the
# agreed signal to switch it off: {"sleeve": 0} in bot/control.json). Details: bot/sleeve_report.py
SLEEVE_BAD, SLEEVE_MIN_N = -1000.0, 15
SY = ["PORT", "ISLE", "AQUA", "GRID", "FERRY", "RECON", "MAT", "GEN", "HAVEN", "REIN", "ARCH", "REBLD"]
STORM_9 = SY[:9]                                   # the stocks the in-storm model and the formation sleeve trade
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
# (the formation sleeve, live since 2026-10-19 16:38, is not a replay setting: the replays don't run it, and the verdict
#  compares live minus the sleeve with them)


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
    timeline, forms, notes, restarts, decisions, ledger, sleeve_marks = [], [], collections.defaultdict(list), [], [], [], []
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
            elif e == "FORMATION SLEEVE entry": sleeve_marks.append(r["ts"])
            if e in ANOMALIES: notes[e].append(r["ts"])
        timeline.append((t_start + 1.0, "startup", snap))
    ident, storms, opens = [], collections.defaultdict(lambda: {"obs": {}, "impacts": set(), "reports": {}}), {}
    for f in sorted(glob.glob(str(LOGS / "feed-*.jsonl"))):
        for m in records(f, _feed):
            t = m["type"]
            if t == "storm_formed":
                ident.append((m["received_at"], m["time"][:10], m["round"], m["storm"]))
                continue
            if t == "open":
                opens.setdefault((m["time"][:10], m["round"]), m["received_at"])
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
        # a round runs from its market open to the next round's market open (a season's first round from the
        # season reset): trades made before a storm forms (the formation sleeve) belong to that storm's round
        o0 = opens.get((date, rnd))
        o0 = o0 if o0 is not None and 0 < t0 - o0 < 300 else None
        w0, w1 = (o0 or t0), t1
        if rnd == 1:
            pos = collections.defaultdict(int)
            w0 = min([x for x in resets if (forms[k - 1] if k else 0) < x <= t0], default=(o0 or t0))
        if rnd == 3:
            w1 = min([x for x in resets if t0 < x < t1], default=t1)
        else:
            j1 = bisect.bisect_right(it, t1 + 3) - 1
            o1 = opens.get((ident[j1][1], ident[j1][2])) if j1 >= 0 else None
            if o1 is not None and t0 < o1 < t1:
                w1 = o1
        start = dict(pos); cash = collections.defaultdict(float)
        pre_cash, pre_pos, exe = 0.0, collections.defaultdict(int), collections.defaultdict(float)
        # formation-to-formation PnL (the bot's own ROUND RESULT window) for the ledger cross-check
        ff_end = t1 if rnd != 3 else w1
        ff_pos = collections.defaultdict(int, start); ff_cash = 0.0
        sleeve_on = any(w0 <= x <= t0 for x in sleeve_marks)
        sl_cash, sl_pos = collections.defaultdict(float), collections.defaultdict(int)
        for ts, s, q, px, syn in fills:
            if w0 <= ts < t0:
                ff_pos[s] += q
            elif t0 <= ts < ff_end:
                ff_cash -= q * px; ff_pos[s] += q
            if sleeve_on and w0 <= ts < t0 + 30 and s in STORM_9:
                sl_cash[s] -= q * px; sl_pos[s] += q
            if w0 <= ts < w1:
                cash[s] -= q * px; pos[s] += q
                if ts < t0:
                    pre_cash -= q * px; pre_pos[s] += q
                if not syn:
                    exe[s] += q * (px - mid(s, ts - 2.5))            # paid vs mid just before the fill
        start_t0 = {s: start.get(s, 0) + pre_pos.get(s, 0) for s in SY}
        ff_pnl = ff_cash + sum(ff_pos[s] * (se.settlement[s] if rnd == 3 else mid(s, ff_end)) for s in SY) \
            - sum(start_t0[s] * mid(s, t0) for s in SY)
        sleeve_pnl = (sum(sl_cash.values()) + sum(q * mid(s, t0 + 30) for s, q in sl_pos.items())) if sleeve_on else 0.0
        endv = {s: se.settlement[s] if rnd == 3 else mid(s, w1) for s in SY}
        leftover = {s: q for s, q in pos.items() if q} if rnd != 3 else {}
        end = {s: pos[s] * endv[s] for s in SY}
        if rnd == 3:
            pos = collections.defaultdict(int)
        by_sym = {s: cash[s] + end[s] - start.get(s, 0) * mid(s, w0) for s in SY}
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
                    "diag": {"core_pnl": ff_pnl, "pre_storm_pnl": pre, "sleeve_pnl": sleeve_pnl, "sleeve_on": sleeve_on,
                             "bot_ledger": bot_ledger(ledger, rnd, name, t0, t1, w1, restarts),
                             "exec_cost": dict(exe), "leftover": leftover, "carried_in": {s_: q for s_, q in start.items() if q}, "hits": {s: tuple(v) for s, v in hits.items()},
                             "move": {s: (mid(s, t0), endv[s]) for s in SY},
                             "storm": storm_shape(sk["obs"]) if sk else None,
                             "islands": sorted(sk["impacts"]) if sk else [],
                             "report": (sk["reports"].get("final") or sk["reports"].get("preliminary")) if sk else None}})
    return out


_replay_cache = {}


LIVE_MODEL = ROOT / "bot" / "models.pkl"


_profile = {"t": 0.0, "p": None}


def book_profile():
    """The depth model from the recorded books (book_profile.build) reads every recorded book: ~35 s. Rebuilding it for
    every replay stalled the monitor loop ~100 s at each new season (2 replays), and the live check then ran on stale
    data: a false BOT SILENT at 2026-10-19 16:44. It changes slowly as books accumulate: rebuilt at most every 6 h."""
    if _profile["p"] is None or time.time() - _profile["t"] > 6 * 3600:
        from book_profile import build
        _profile["p"], _profile["t"] = build(), time.time()
    return _profile["p"]


def honest_replay(season, epoch, live_model=False):
    """Per-round, per-symbol PnL of the same research season with the settings of `epoch`, using an honest model
    (never trained on this season: what the strategy does on an unseen storm) or, with live_model=True, the live
    bot's own model (what the live bot would have done at house prices; it was trained on this season, so it is
    optimistic, but it shares the live bot's decisions)."""
    key = (season, epoch[0], live_model)
    if key in _replay_cache:
        return _replay_cache[key]
    model = LIVE_MODEL if live_model else next((m for a, b, m in HONEST if a <= season <= b and m.exists()), None)
    if model is None:
        return None
    import strategy as S
    from replay import replay_season
    cfg = epoch[1]
    S.BUFFER, S.EDGE_SHARE = cfg["BUFFER"], cfg["EDGE_SHARE"]
    def tweak(st):
        st.allow_flip, st.min_age = cfg["allow_flip"], cfg["min_age"]
    extra = {"PORT": 0.002, "FERRY": 0.0012, "AQUA": 0.0021, "MAT": 0.0007, "GEN": 0.0005}   # live-calibrated costs
    rows = []
    replay_season(season, model, 0.1, profile=book_profile(), per_symbol=rows, extra=extra, tweak=tweak)
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


def replay_at(rd, age, live_model=False):
    """Replay round PnL of the same storm at this storm age (testnet seasons only), or None."""
    date = rd["key"][0]
    se = H.season_of(date) if date else None
    if se is None:
        return None
    epoch = settings_at(time.strftime("%Y-%m-%d %H:%M", time.localtime(rd["t_form"])))
    if honest_replay(se.number, epoch, live_model) is None:
        return None
    path = _path_cache[(se.number, epoch[0], live_model)].get(rd["round"])
    if not path:
        return None
    return path[min(max(int(age // 5), 0), len(path) - 1)]


def verdict(live_pnl, rep_pnl, rep2_pnl=None):
    """rep_pnl: honest replay; rep2_pnl: replay with the live bot's own model. Live-specific only if live is much
    worse than both; variance if either lost too."""
    reps = [x for x in (rep_pnl, rep2_pnl) if x is not None]
    if not reps:
        return "NO REPLAY (new storm): judge by anomalies and symbols"
    worst = min(reps)
    tol = max(2500.0, 0.5 * abs(worst))
    if live_pnl - worst < -tol:                      # much worse than BOTH replays
        return "LIVE-SPECIFIC: live much worse than both replays of the same storm -- investigate"
    if live_pnl < BAD and worst < BAD / 2:
        return "VARIANCE: a replay of this storm lost too (the storm, not the bot)"
    if live_pnl < BAD:
        return "MIXED: bad round, replay lost less -- check the symbols with the largest gaps"
    return "OK"


def diagnose(r, v, rep_total, sym_gap, rep2_total=None):
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
        lines.append("accounting: no bot ledger line for this round (a restart split it, or it was not logged yet); "
                     "positions reconciled with the exchange")
    if d.get("sleeve_on"):
        lines.append(f"formation sleeve (positions built before the storm formed, closed by storm age 30): ${d['sleeve_pnl']:,.0f}; "
                     f"the verdict compares live minus the sleeve with the replays")
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
        reps = ", ".join(f"{n} ${x:,.0f}" for n, x in (("honest replay", rep_total), ("same-model replay", rep2_total)) if x is not None)
        cause = f"STORM (variance): replays of the same storm made {reps}{turn}{hit}"
        if problems:
            cause += " | also check: " + "; ".join(problems)
    elif v.startswith("NO REPLAY"):
        cause = ("check: " + "; ".join(problems)) if problems else f"no replay; no bot anomaly{turn}{hit}"
    else:
        cause = "OK" + (" | check: " + "; ".join(problems) if problems else "")
    return lines, cause


def sleeve_review(new):
    """After each new round with the sleeve: alert when SLEEVE_MIN_N+ live sleeve rounds average below 0."""
    if not any(rec.get("formation_sleeve_pnl") is not None for rec in new):
        return None
    sl = []
    for line in open(LOGS / "round_reports.jsonl"):
        try:
            x = json.loads(line).get("formation_sleeve_pnl")
        except ValueError:
            continue
        if x is not None:
            sl.append(x)
    if len(sl) >= SLEEVE_MIN_N and sum(sl) / len(sl) < 0:
        msg = (f"FORMATION SLEEVE: {len(sl)} live rounds average ${sum(sl) / len(sl):,.0f}/round (backtest +$311): the agreed "
               f"signal to switch it off ({{\"sleeve\": 0}} in bot/control.json) -- check bot/sleeve_report.py first")
        alert(msg)
        return msg
    return None


def report(rounds, seen, st):
    md, new = [], []
    for r in rounds:
        if r["key"] in seen:
            continue
        epoch = settings_at(r["started"])
        rep = honest_replay(r["season"], epoch)
        rp = rep.get(r["round"]) if rep else None
        rep_total = sum(rp.values()) if rp else None
        rep2 = honest_replay(r["season"], epoch, live_model=True) if rep is not None else None
        rp2 = rep2.get(r["round"]) if rep2 else None
        rep2_total = sum(rp2.values()) if rp2 else None
        sleeve = r["diag"].get("sleeve_pnl", 0.0)
        v = verdict(r["pnl"] - sleeve, rep_total, rep2_total)
        sym_gap = {s: r["by_symbol"][s] - rp.get(s, 0.0) for s in SY} if rp else {}       # includes the sleeve if on
        sleeve_bad = bool(r["diag"].get("sleeve_on")) and sleeve < SLEEVE_BAD
        flagged = r["pnl"] < BAD or not v.startswith(("OK", "NO REPLAY")) or bool(r["diag"]["leftover"]) or sleeve_bad
        lines, cause = diagnose(r, v, rep_total, sym_gap, rep2_total)
        if sleeve_bad:
            cause += (f" | FORMATION SLEEVE lost ${-sleeve:,.0f} this round (backtest worst -$392): check its fills "
                      f"(bot/sleeve_report.py) and whether the house moved against its usual formation direction")
        if r["diag"]["bot_ledger"] is not None and cause.startswith("ACCOUNTING"):
            flagged = True
        worst = sorted(r["by_symbol"].items(), key=lambda kv: kv[1])[:3]
        rec = {"round": r["key"], "storm": r["storm"], "started": r["started"], "settings_since": epoch[0], "live_pnl": round(r["pnl"]),
               "honest_replay_pnl": None if rep_total is None else round(rep_total),
               "same_model_replay_pnl": None if rep2_total is None else round(rep2_total), "verdict": v, "cause": cause,
               "formation_sleeve_pnl": round(sleeve) if r["diag"].get("sleeve_on") else None,
               "worst_symbols": {s: round(x) for s, x in worst}, "anomalies": r["anomalies"],
               "live_minus_replay": {s: round(g) for s, g in sym_gap.items()}, "diagnosis": lines,
               "bot_ledger": None if r["diag"]["bot_ledger"] is None else round(r["diag"]["bot_ledger"])}
        new.append(rec); seen.add(r["key"])
        line = (f"{r['started']}  season {r['season']} r{r['round']} {r['storm']:6s} live ${r['pnl']:8,.0f}  "
                f"replay {'   n/a' if rep_total is None else f'${rep_total:8,.0f}'}"
                f"{'' if rep2_total is None else f' (same model ${rep2_total:,.0f})'}"
                f"{f'  [formation sleeve ${sleeve:,.0f}]' if r['diag'].get('sleeve_on') else ''}  {v.split(':')[0]}")
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
        sleeve_review(new)
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
        D = Data(); D.replay_hook = replay_at; D.replay_hook2 = lambda rd, a: replay_at(rd, a, live_model=True); D.refresh()
        end = time.time() - 10                       # the logs as loaded now (the run itself takes minutes)
        st, now = {}, start
        while now <= end:
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
    D = Data(); D.replay_hook = replay_at; D.replay_hook2 = lambda rd, a: replay_at(rd, a, live_model=True)
    wst = {}
    last_sig, last_err = None, 0.0
    while True:
        try:
            D.refresh()
            # a round completed when the next storm formed; wait 20 s so the bot has logged its own ROUND RESULT
            if len(D.forms) != last_sig and (not D.forms or time.time() - D.forms[-1][0] > 20):
                report(live_rounds(), seen, live); last_sig = len(D.forms)
            if not args.once:
                D.refresh()                                   # a round report can take a while: never judge on stale data
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
