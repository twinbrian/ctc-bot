"""Live watch: the round's true running PnL, alerts on real drops, and an immediate diagnosis of every alert.

The PnL comes from one consistent source: the bot's logged fills, ordered by their EXCHANGE execution time, valued at
the recorded book mid, and reconciled with the exchange's own positions (a fill the bot never logged is booked from
the exchange's position snapshot). It is evaluated LAG seconds behind real time, so every fill executed by then has
been logged (logging lag on 2026-10-18: median 1.1 s, max 2.5 s). Checked against the exchange dashboard: -$6,182.71
vs -$6,116.82 shown (the difference is mid vs the exchange's marks), same positions.

The bot's own status PnL is NOT used: it mixes positions (polled every 1 s) with fill cash (polled every 2 s), so
during fast entries it showed phantom swings of up to -$15k (Lena 17:50:22), which raised false alerts and used up
the alert levels that the real drop three minutes later needed.
"""

from __future__ import annotations

import bisect
import collections
import glob
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "bot" / "logs"
SINCE = "run-20261018-065413"
CASE = json.loads((ROOT / "case.json").read_text())
ISLES = {k: (sum(p[0] for p in v["polygon"]) / len(v["polygon"]), sum(p[1] for p in v["polygon"]) / len(v["polygon"]))
         for k, v in CASE["islands"].items()}

LAG = 5.0                                   # evaluate this far behind real time (all fills logged by then)
SILENT = 90.0                               # no bot status line for this long = bot down, frozen or asleep
# A drop is material (about one standard deviation of a round's PnL) when any of these holds:
ROUND_LOSS = 5000.0                         # the round is down $5k
FROM_BEST = 8000.0                          # the round is $8k below its best so far
SUDDEN, SUDDEN_WIN = 6000.0, 60.0           # $6k lost within 60 s
DEEPEN, RECOVER = 4000.0, 3000.0            # one alert per incident: again only if $4k worse; a new incident needs every
                                            # rule to clear AND a $3k recovery (no repeats while hovering at a threshold)
TYPICAL_COST_BP = 46.0                      # execution cost vs mid on 2026-10-18: 46 bp of value traded (90th pct per round 87)
BOT_EVENTS = ("error", "ORDER REJECTED", "DISCONNECTED", "PREDICTION REJECTED (implausible)",
              "ROUND STOP-LOSS: flattening and sitting out the rest of the storm")
HEALTH = ("ORDER REJECTED", "DISCONNECTED", "error", "rate budget exhausted (lower-priority symbols wait)",
          "PREDICTION REJECTED (implausible)", "LEFTOVER re-flatten from current mid")


# ---- incremental log reading: only new lines are parsed, records are kept in compact form ----------------
_cache: dict = {}


def records(path, compact=None):
    st = _cache.setdefault(path, {"off": 0, "recs": []})
    with open(path, "rb") as f:
        f.seek(st["off"]); data = f.read()
    end = data.rfind(b"\n") + 1
    for line in data[:end].splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        r = compact(r) if compact else r
        if r is not None:
            st["recs"].append(r)
    st["off"] += end
    return st["recs"]


def _book(b):
    if b.get("bids") and b.get("asks"):
        return b["symbol"], b["received_at"], (b["bids"][0]["price"] + b["asks"][0]["price"]) / 2
    return None


FEED_KEEP = ("storm_formed", "storm", "impact", "damage_report", "advisory", "storm_dissipated")


def _feed(fr):
    m = fr.get("message") or {}
    if m.get("type") not in FEED_KEEP:
        return None
    keep = ("type", "time", "round", "storm", "age", "x", "y", "vx", "vy", "radius", "intensity", "island", "stage",
            "damage_pct", "industry_loss_musd", "source", "outlook", "headline")
    return {"received_at": fr["received_at"], **{k: m[k] for k in keep if k in m}}


def exec_time(r):
    """The exchange's execution time of a logged fill (local clock); the log time if it is missing."""
    at = r.get("at")
    if not at:
        return r["ts"]
    try:
        t = time.mktime(time.strptime(at[:19], "%Y-%m-%dT%H:%M:%S"))
        return t + (float("0" + at[19:26]) if at[19:20] == "." else 0.0)
    except ValueError:
        return r["ts"]


def reconcile(timeline, mid):
    """Logged fills -> fills consistent with the exchange's positions. A fill the bot never logged (made just
    before a restart, or missed by the GET /trades poll) is added as a synthetic fill: at the exchange's average
    price when a restart's position snapshot shows it, else at mid once the gap has persisted for 3 status lines.
    Without this, a restart mid-round leaves phantom positions that are charged to the next round."""
    fills, pos, pending = [], collections.defaultdict(int), {}
    def synth(ts, s, q, px):
        fills.append((ts, s, q, px, True)); pos[s] += q
    for ts, kind, x in sorted(timeline, key=lambda e: e[0]):
        if kind == "fill":
            s, q, px = x; fills.append((ts, s, q, px, False)); pos[s] += q
        elif kind == "reset":                                   # new testnet season: the exchange zeroed positions
            pos, pending = collections.defaultdict(int), {}
        elif kind == "startup":                                 # exchange positions when a bot process started
            for s in set(pos) | set(x):
                q, avg = x.get(s, (0, None))
                if q != pos[s]:
                    synth(ts, s, q - pos[s], avg if avg and q * (q - pos[s]) > 0 else mid(s, ts))
            pending = {}
        elif kind == "status":
            gaps = {s: x.get(s, 0) - pos[s] for s in set(pos) | set(x) if x.get(s, 0) != pos[s]}
            for s, d in gaps.items():
                n = pending.get(s, (d, 0))[1] + 1 if pending.get(s, (None,))[0] == d else 1
                if n >= 3:
                    synth(ts, s, d, mid(s, ts)); n = 0
                pending[s] = (d, n)
            pending = {s: v for s, v in pending.items() if s in gaps and v[1]}
    return fills


def _angle(vx, vy):
    return math.degrees(math.atan2(vy, vx))


def _adiff(a, b):
    return abs((a - b + 180) % 360 - 180)


class Data:
    """Everything the bot logged, read incrementally. Queries take a time t and a time `known` and use only what
    had been logged by `known`, so the same code runs live and in a replay of a past day."""

    def __init__(self, logs=LOGS, since=SINCE):
        self.logs, self.since = logs, since
        self.fills = []                     # (exec_t, log_t, sym, qty, px), sorted by exec_t
        self.status = []                    # (t, exchange positions, bot_pnl)
        self.forms = []                     # (t, round, storm): live storm formations
        self.resets = []                    # t: season changes (the exchange zeroes positions)
        self.decisions = []                 # (t, round, age, preds, goals)
        self.events = []                    # (t, event, record)
        self.snaps = []                     # (t, {sym: (qty, avg price)}): positions when a bot process started
        self.restarts = []
        self.book_t, self.book_m = collections.defaultdict(list), collections.defaultdict(list)
        self.ident = []                     # (t, date, round, storm) from the feed
        self.obs = collections.defaultdict(list)       # (date, round, storm) -> [(t, age, x, y, vx, vy, radius, I)]
        self.adv = collections.defaultdict(list)       # key -> [(t, age, source, outlook, headline)]
        self.impacts = collections.defaultdict(list)   # key -> [(t, island)]
        self.reports = collections.defaultdict(list)   # key -> [(t, stage, damage_pct, loss)]
        self.dissipated = {}                           # key -> t
        self.replay_hook = None                        # fn(round info, storm age) -> honest replay round PnL, or None
        self._done, self._snap = {}, {}

    def refresh(self):
        for f in sorted(glob.glob(str(self.logs / "run-*.jsonl"))):
            if Path(f).name < self.since + ".jsonl":
                continue
            recs = records(f); i = self._done.get(f, 0)
            if not recs:
                continue
            t0 = recs[0]["ts"]
            if i == 0:
                self.restarts.append(t0); self._snap[f] = {}
            for r in recs[i:]:
                e, ts = r["event"], r["ts"]
                if f in self._snap and ts >= t0 + 3:
                    self.snaps.append((t0 + 1.0, self._snap.pop(f)))
                if e == "fill":
                    self.fills.append((exec_time(r), ts, r["symbol"], r["qty"], r["price"]))
                elif e == "position change" and ts < t0 + 3 and r["frm"] == 0 and r["to"]:
                    if f in self._snap:
                        self._snap[f][r["symbol"]] = (r["to"], -r["cash_flow"] / r["to"])
                elif ts < t0 + 5:
                    continue                                   # catch-up replays recorded news
                elif e == "status":
                    self.status.append((ts, r.get("positions") or {}, r.get("bot_pnl")))
                elif e == "storm_formed":
                    self.forms.append((ts, r["round"], r["storm"]))
                elif e == "new season":
                    if not self.resets or ts - self.resets[-1] > 30:   # one season change logs two events
                        self.resets.append(ts)
                elif e == "decision":
                    self.decisions.append((ts, r["round"], r["age"], r.get("preds") or {}, r.get("goals") or {}))
                else:
                    self.events.append((ts, e, r))
            self._done[f] = len(recs)
        self.fills.sort(key=lambda x: x[0])
        self._fill_t = [x[0] for x in self.fills]
        self._status_t = [x[0] for x in self.status]
        for f in sorted(glob.glob(str(self.logs / "books-*.jsonl"))):
            recs = records(f, _book); i = self._done.get(f, 0)
            for s, t, m in recs[i:]:
                if self.book_t[s] and t < self.book_t[s][-1]:
                    k = bisect.bisect_right(self.book_t[s], t); self.book_t[s].insert(k, t); self.book_m[s].insert(k, m)
                else:
                    self.book_t[s].append(t); self.book_m[s].append(m)
            self._done[f] = len(recs)
        for f in sorted(glob.glob(str(self.logs / "feed-*.jsonl"))):
            recs = records(f, _feed); i = self._done.get(f, 0)
            for m in recs[i:]:
                t, typ = m["received_at"], m["type"]
                if typ == "storm_formed":
                    self.ident.append((t, m["time"][:10], m["round"], m["storm"])); continue
                k = (m["time"][:10], m["round"], m["storm"])
                if typ == "storm":
                    self.obs[k].append((t, m["age"], m["x"], m["y"], m["vx"], m["vy"], m["radius"], m["intensity"]))
                elif typ == "advisory":
                    self.adv[k].append((t, m.get("age"), m.get("source"), m.get("outlook"), m.get("headline", "")))
                elif typ == "impact":
                    self.impacts[k].append((t, m.get("island")))
                elif typ == "damage_report":
                    self.reports[k].append((t, m.get("stage"), m.get("damage_pct"), m.get("industry_loss_musd")))
                elif typ == "storm_dissipated":
                    self.dissipated.setdefault(k, t)
            self._done[f] = len(recs)
        self.ident.sort()

    # ---- queries ------------------------------------------------------------------------------------------
    def mid(self, s, t):
        ts = self.book_t.get(s)
        if not ts:
            return 0.0
        return self.book_m[s][max(bisect.bisect_right(ts, t) - 1, 0)]

    def season_start(self, t):
        k = bisect.bisect_right(self.resets, t)
        return self.resets[k - 1] if k else -1e18

    def season_value(self, t, known):
        """Season PnL at time t (fills + positions at mid), using what had been logged by `known`.
        Returns (value, positions, reconciled fills)."""
        r0 = self.season_start(t)
        a, b = bisect.bisect_right(self._fill_t, r0), bisect.bisect_right(self._fill_t, t)
        tl = [(e, "fill", (s, q, p)) for e, lt, s, q, p in self.fills[a:b] if lt <= known]
        a, b = bisect.bisect_right(self._status_t, r0), bisect.bisect_right(self._status_t, t)
        tl += [(ts, "status", pos) for ts, pos, _ in self.status[a:b]]
        tl += [(ts, "startup", sn) for ts, sn in self.snaps if r0 < ts <= t]
        fl = reconcile(tl, self.mid)
        cash, pos = 0.0, collections.defaultdict(int)
        for _, s, q, px, _ in fl:
            cash -= q * px; pos[s] += q
        return cash + sum(q * self.mid(s, t) for s, q in pos.items()), {s: q for s, q in pos.items() if q}, fl

    def round_at(self, t):
        """The storm round in progress at t: dict(t_form, round, storm, key) or None (between seasons)."""
        k = bisect.bisect_right([f[0] for f in self.forms], t)
        if not k:
            return None
        tf, rnd, name = self.forms[k - 1]
        if self.season_start(t) > tf:
            return None                                        # the season ended after this storm formed
        j = bisect.bisect_right([x[0] for x in self.ident], tf + 3) - 1
        date = self.ident[j][1] if j >= 0 and self.ident[j][2] == rnd and self.ident[j][3] == name else None
        return {"t_form": tf, "round": rnd, "storm": name, "key": (date, rnd, name)}

    def last_status_t(self, known):
        k = bisect.bisect_right(self._status_t, known)
        return self.status[k - 1][0] if k else None


# ---- diagnosis of a live drop ------------------------------------------------------------------------------
def _storm_lines(D, rd, t_ref, t):
    k = rd["key"]
    ob = [o for o in D.obs.get(k, []) if o[0] <= t]
    if not ob:
        return [], None
    now = ob[-1]
    ref = next((o for o in reversed(ob) if o[0] <= t_ref), ob[0])
    lines, surprise = [], None
    h60 = next((o for o in reversed(ob) if o[1] <= now[1] - 60), None)          # where its motion pointed 60 s ago
    if h60 is not None and now[1] - h60[1] <= 70 and now[7] > 0:
        dt = now[1] - h60[1]
        surprise = math.hypot(now[2] - (h60[2] + h60[4] * dt), now[3] - (h60[3] + h60[5] * dt)) / now[6]
    dist_now = {i: math.hypot(now[2] - x, now[3] - y) / now[6] for i, (x, y) in ISLES.items()}
    closest = {i: min(math.hypot(o[2] - x, o[3] - y) / o[6] for o in ob) for i, (x, y) in ISLES.items()}
    lines.append(f"storm now (age {now[1]}): heading {_angle(now[4], now[5]):.0f}° (was {_angle(ref[4], ref[5]):.0f}° at age {ref[1]}), "
                 f"speed {math.hypot(now[4], now[5]):.1f}, intensity {now[7]:.2f}"
                 + (f"; it is {surprise:.1f} radii from where its motion pointed 60 s ago" if surprise is not None else ""))
    lines.append("distance to islands now / closest so far (radii; damage needs < 2): "
                 + ", ".join(f"{i} {dist_now[i]:.1f}/{closest[i]:.1f}" for i in sorted(ISLES, key=lambda i: closest[i])))
    off = [a for a in D.adv.get(k, []) if a[2] == "official" and a[0] <= t]
    off_ref = [a for a in off if a[0] <= t_ref]
    if off:
        if off_ref and off_ref[-1] is not off[-1]:
            lines.append(f"official advisory then: \"{off_ref[-1][4][:110]}\"")
        lines.append(f"official advisory now: \"{off[-1][4][:110]}\"")
    imp = [i for ti, i in D.impacts.get(k, []) if ti <= t]
    if imp:
        lines.append(f"hurricane conditions reported on: {', '.join(sorted(set(imp)))}")
    return lines, surprise


def diagnose_live(D, t, known, rd, t_ref):
    """Why the round's PnL fell between t_ref and t. Returns (cause, lines)."""
    _, pos_ref, _ = D.season_value(t_ref, known)
    v_now, pos_now, fl = D.season_value(t, known)
    by = {}
    for s in set(pos_ref) | set(pos_now) | {x[1] for x in fl if t_ref < x[0] <= t}:
        held = pos_ref.get(s, 0) * (D.mid(s, t) - D.mid(s, t_ref))
        tr = [(ft, q, px) for ft, x, q, px, _ in fl if x == s and t_ref < ft <= t]
        exe = sum(q * (D.mid(s, ft - 0.5) - px) for ft, q, px in tr)
        after = sum(q * (D.mid(s, t) - D.mid(s, ft - 0.5)) for ft, q, px in tr)
        by[s] = {"held": held, "exec": exe, "after": after, "total": held + exe + after,
                 "p0": D.mid(s, t_ref), "p1": D.mid(s, t), "pos0": pos_ref.get(s, 0), "pos1": pos_now.get(s, 0)}
    loss = sum(x["total"] for x in by.values())
    execution = sum(x["exec"] for x in by.values())
    traded = sum(abs(q) * px for ft, x, q, px, _ in fl if t_ref < ft <= t)
    cost_bp = -execution / traded * 1e4 if traded else 0.0
    k = rd["key"]
    dec = [d for d in D.decisions if rd["t_form"] <= d[0] <= t and d[1] == rd["round"]]
    diss_t = D.dissipated.get(k)
    after_storm = diss_t is not None and diss_t <= t
    reps = [r[0] for r in D.reports.get(k, []) if r[0] <= t]
    # positions the bot is holding against its own plan
    flagged = {}
    for s, x in by.items():
        p = x["pos1"]
        if not p:
            continue
        if after_storm:
            if rd["round"] != 3 and reps and t - max(reps) > 60:
                flagged[s] = f"still {p:+d} more than 60 s after the damage report (rounds 1-2 should be flat)"
            continue                                            # round 3 holds to settlement by design
        recent = [d for d in dec if d[0] >= t - 45]
        if len(recent) >= 3 and recent[-1][0] - recent[0][0] >= 40 and all(
                (d[4].get(s) or [0])[0] == 0 or ((d[4].get(s) or [0])[0] > 0) != (p > 0) for d in recent):
            flagged[s] = (f"{p:+d}, but its last {len(recent)} decisions (ages {recent[0][2]}-{recent[-1][2]}) all want "
                          f"{(recent[-1][4].get(s) or [0])[0]:+d}: its orders are not getting filled")
    flagged_pnl = sum(by[s]["total"] for s in flagged)
    market = sum(x["held"] + x["after"] for s, x in by.items() if s not in flagged)
    # bot health: anything wrong near the drop (from 30 s before the window) counts; earlier events are context
    w0 = t_ref - 30
    near = collections.Counter(e for ts, e, _ in D.events if w0 <= ts <= t and e in ("ORDER REJECTED", "DISCONNECTED", "error"))
    # a restart or a silent bot can only hurt while positions are open: count them only then
    rs = sum(1 for x in D.restarts if w0 < x <= t and D.season_value(x, known)[1])
    sts = [s_ for s_ in D.status if w0 <= s_[0] <= t]
    gap = max((b_[0] - a_[0] for a_, b_ in zip(sts, sts[1:]) if a_[1]), default=0.0)
    syn = sum(1 for x in fl if x[4] and w0 <= x[0] <= t)
    serious = dict(near)
    if rs: serious["restart"] = rs
    if gap > 25: serious[f"bot silent for {gap:.0f} s"] = 1
    if syn: serious["fills the bot never logged"] = syn
    ctx = collections.Counter(e for ts, e, _ in D.events if rd["t_form"] <= ts <= t and e in HEALTH)
    ctx_rs = sum(rd["t_form"] < x <= t for x in D.restarts)
    # the honest replay of the same storm over the same storm ages (testnet: the seasons are research seasons)
    rep_d = None
    if D.replay_hook is not None:
        r1, r0 = D.replay_hook(rd, t - rd["t_form"]), D.replay_hook(rd, max(t_ref - rd["t_form"], 0.0))
        if r1 is not None and r0 is not None:
            rep_d = r1 - r0
    lines = []
    worst = sorted(by.items(), key=lambda kv: kv[1]["total"])
    n_against = sum(1 for x in by.values() if x["total"] < -50)
    age0, age1 = t_ref - rd["t_form"], t - rd["t_form"]
    lines.append(f"window age {age0:.0f}-{age1:.0f} s: ${loss:,.0f} = price moves on held positions ${market:,.0f} + execution ${execution:,.0f} "
                 f"({cost_bp:.0f} bp of ${traded:,.0f} traded; typical {TYPICAL_COST_BP:.0f} bp)"
                 + (f" + positions against the bot's plan ${flagged_pnl:,.0f}" if flagged else "") + f"; {n_against} symbols lost money")
    lines.append("by symbol: " + "; ".join(
        f"{s} ${x['total']:,.0f} (pos {x['pos0']:+d}->{x['pos1']:+d}, price {x['p0']:.2f}->{x['p1']:.2f} {x['p1'] / x['p0'] - 1:+.1%})"
        for s, x in worst[:5] if x["total"] < -50))
    if dec:
        first = next((d for d in reversed(dec) if d[0] <= t_ref), dec[0])
        lines.append("model predictions (predicted move to storm end, $/share), start of window -> now: " + ", ".join(
            f"{s} {first[3].get(s, 0):+.2f}->{dec[-1][3].get(s, 0):+.2f}" for s, x in worst[:4] if s in dec[-1][3]))
    st_lines, surprise = _storm_lines(D, rd, t_ref, t)
    lines += st_lines
    if rep_d is not None:
        lines.append(f"honest replay of this same storm over the same ages {age0:.0f}-{age1:.0f}: ${rep_d:,.0f} (live ${loss:,.0f})")
    if ctx_rs:
        ctx["restart"] = ctx_rs
    lines.append("bot health this round: " + (", ".join(f"{h} {n}" for h, n in ctx.items()) if ctx else "no errors, rejects, disconnects or restarts")
                 + (f" | NEAR THE DROP: {serious}" if serious else ""))
    for s, why in flagged.items():
        lines.append(f"CHECK {s}: {why} (${by[s]['total']:,.0f} in the window)")
    if loss >= 0:
        return "no loss in the window", lines
    if serious:
        cause = f"BOT: {', '.join(serious)} near the drop -- check whether it caused the loss"
    elif flagged and flagged_pnl <= 0.25 * loss:
        cause = f"BOT: {', '.join(flagged)} held against the bot's own plan lost ${-flagged_pnl:,.0f} of the ${-loss:,.0f}"
    elif execution <= 0.4 * loss and cost_bp > 2 * TYPICAL_COST_BP:
        cause = f"EXECUTION: trading costs ${-execution:,.0f} of the ${-loss:,.0f} ({cost_bp:.0f} bp, typical {TYPICAL_COST_BP:.0f})"
    elif rep_d is not None and rep_d <= 0.5 * loss:
        # no bot problem and normal costs: the honest replay of the same storm, over the same ages, is the arbiter
        storm = f"; the storm is {surprise:.1f} radii off where its motion pointed 60 s ago" if surprise is not None and surprise >= 1.0 else ""
        cause = (f"STORM (variance): the honest replay of this storm lost ${-rep_d:,.0f} over the same ages (live ${-loss:,.0f}); "
                 f"{market / loss:.0%} of the loss is price moves on the model's positions ({n_against} symbols), "
                 f"execution {cost_bp:.0f} bp (typical {TYPICAL_COST_BP:.0f}){storm}; no bot problem found")
    elif market <= 0.5 * loss:
        what = f"{market / loss:.0%} of the loss is price moves on the positions the model chose ({n_against} symbols moved against it together)"
        if rep_d is not None:
            cause = (f"MARKET, BUT NOT IN THE REPLAY: {what}; the honest replay over the same ages: ${rep_d:,.0f} -- "
                     f"live positions or timing differ from the strategy's; check")
        else:
            cause = f"MARKET/STORM: {what}; no bot problem found (no replay for this storm)"
    else:
        cause = "UNCLEAR: no single explanation for most of the loss -- needs a manual look"
    return cause, lines


# ---- the watch -------------------------------------------------------------------------------------------
def live_watch(D, st, now, emit):
    """One check at time `now`. emit(kind, message, lines) is called for every alert."""
    last = D.last_status_t(now)
    if last is not None and now - last > SILENT:
        if not st.get("silent"):
            emit("BOT SILENT", f"no bot status line for {now - last:.0f} s (process down, frozen, or the Mac asleep?)", [])
            st["silent"] = True
        return
    st["silent"] = False
    t_ev = st.get("ev_t", now - 15)
    for ts, e, r in D.events:
        if t_ev < ts <= now and e in BOT_EVENTS and now - st.setdefault("ev_last", {}).get(e, 0) > 300:
            st["ev_last"][e] = now
            emit("BOT EVENT", f"{e}: " + json.dumps({k: v for k, v in r.items() if k not in ("ts", "event")})[:300], [])
    st["ev_t"] = now
    t = now - LAG
    rd = D.round_at(t)
    if rd is None:
        return
    if st.get("key") != rd["t_form"]:
        st.update(key=rd["t_form"], base=D.season_value(rd["t_form"], now)[0], peak=0.0, peak_t=rd["t_form"],
                  hist=collections.deque(), alert_v=None, armed=True, path=[])
    v = D.season_value(t, now)[0] - st["base"]
    st["path"].append((t, v))
    h = st["hist"]; h.append((t, v))
    while h and h[0][0] < t - SUDDEN_WIN:
        h.popleft()
    if v > st["peak"]:
        st["peak"], st["peak_t"] = v, t
    trig = []
    if v <= -ROUND_LOSS:
        trig.append(f"round PnL ${v:,.0f}")
    if st["peak"] - v >= FROM_BEST:
        trig.append(f"${st['peak'] - v:,.0f} below the round's best (${st['peak']:,.0f})")
    hi_t, hi = max(h, key=lambda x: x[1])
    if hi - v >= SUDDEN:
        trig.append(f"-${hi - v:,.0f} within {t - hi_t:.0f} s")
    last = st.get("alert_v")
    if not trig and (last is None or v >= last + RECOVER):
        st["armed"] = True                                 # the drop is over (rules clear AND $3k recovered): next one is new
    new_incident = st.get("armed", True)
    if trig and (new_incident or v <= last - DEEPEN):
        what = "" if new_incident else f"worse by ${last - v:,.0f} since the last alert; "
        st["alert_v"], st["armed"] = v, False
        cause, lines = diagnose_live(D, t, now, rd, st["peak_t"])
        emit("DROP", f"{rd['storm']} round {rd['round']} at age ~{t - rd['t_form']:.0f} s: {what}{'; '.join(trig)} | CAUSE: {cause}", lines)


def replay_day(start, end, emit, step=15.0):
    """Run the watch over past logs with a simulated clock (validation)."""
    D = Data(); D.refresh()
    st, now = {}, start
    paths = {}
    while now <= end:
        live_watch(D, st, now, emit)
        if st.get("key"):
            paths[st["key"]] = st["path"]
        now += step
    return D, paths
