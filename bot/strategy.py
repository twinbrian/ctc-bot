"""Event-driven strategy: decides target positions from feed events; a broker turns targets into orders.

Trades (see analysis/FINDINGS.md):
  1. In-storm:   at storm ages 10, 30, ..., 470 the models predict the house's move to the end of the storm
                 (`pred`, $/share). Goal = full position limit in the direction of `pred`, but never paying
                 more than mid + pred/2 per share (the other half pays for the exit) - so the book depth
                 decides how much actually fills. A position against the prediction is closed paying up to
                 the whole predicted move. Flatten when the storm dissipates, accepting up to EXIT_SLIP.
  2. Final snipe (round 3): the final damage report fixes every settlement exactly; take every resting
                 order on the wrong side of value -/+ margin, then hold to settlement.
"""

from __future__ import annotations

import json
import pickle
from datetime import date
from collections import deque
from pathlib import Path

import numpy as np

from engine import Storm, Season, features, exact_values, SYMS, STOCKS

EDGE_SHARE = 0.7                        # pay at most this fraction of the predicted move to enter (tested 0.3-1.0 on two
                                        # held-out sets: 0.5 -> $8,293/$8,619, 0.7 -> $8,800/$9,249, plateau to 0.85)
BUFFER = 0.01                           # enter only if half the predicted move clears 1% of price (tested out-of-sample
                                        # with live-calibrated costs: 0.1% -> $5,537/season, 1% -> $6,930, 1% + no flip -> $7,085)
EXIT_MARGIN = 0.0015                    # after the storm, close within 0.15% of the model's fair value
REPORT_EXIT_MARGIN = 0.003              # after a damage report the house quote is fair again: close within 0.3% of mid
LEFTOVER_RETRY = 20.0                   # rounds 1-2: after a report, retry a stuck exit from the current mid this often (s)
STORM_SYMS = [s for s in ("PORT", "ISLE", "AQUA", "GRID", "FERRY", "RECON", "MAT", "GEN", "HAVEN")]  # REIN: edge < cost
SNIPE_MARGIN = {"PORT": 0.15, "GRID": 0.12, "FERRY": 0.12, "GEN": 0.12, "ARCH": 0.40, "REBLD": 0.25}
DEFAULT_MARGIN = 0.05
OUTAGE_FREE = {"ISLE", "AQUA", "HAVEN", "REIN", "RECON", "MAT"}   # values need only the reported damage %
# formation front-run: the house reprices ~3-8 s after `storm_formed`, in this direction in >= 94% of
# 1,800 research storms (PORT 87% and REIN 73% are left out). Pay at most FORM_CAP from mid.
FORM_DIR = {"ISLE": 1, "AQUA": 1, "GRID": 1, "FERRY": 1, "HAVEN": 1, "RECON": -1, "MAT": -1, "GEN": -1}
FORM_CAP = 0.004
STATE_FILE = Path(__file__).resolve().parent / "state.json"
# --- formation sleeve (live since 2026-10-19) -------------------------------------------------------------------
# The house reprices island stocks/HAVEN up and rebuilders down within seconds of storm_formed (77% of the move by
# storm age 4, peak by age 28; research seasons 161-600). Build FORM_SIZE x the limit before the storm forms (30-58 s
# after market open), hold through formation, exit at storm ages 10-28, every order within FORM_BAND of the current
# mid; any leftover is handed to the in-storm model at age 30. Directions: bot/formation_dirs.json (seasons 161-600,
# never replayed on the testnet). The open is timed from the first pre-storm snapshot whose market clock is running
# (~30 s in): in round 1 of a testnet season the clock sits at 09:30:00 for ~120 s, so a 09:30 snapshot is no open.
# Backtest (this code on the recorded live books, 94 rounds, 55 orders/min): +$302/round vs without (se 46), better
# in 78% of rounds; the sleeve's own trades +$311/round, worst -$392; ~+$258/round after our own price impact.
# Full size was +$305/round after impact with -$4.7k rounds -> 20%. Off: bot/control.json {"sleeve": 0}.
FORM_SIZE = 0.2                            # fraction of each limit (set by live.py; the Strategy default is off)
FORM_BAND = 0.005
FORM_ENTRY = (30.0, 58.0)                  # seconds after market open (formation is at 60)
FORM_EXIT = (10.0, 28.0)                   # storm age
FORM_DIRS_FILE = Path(__file__).resolve().parent / "formation_dirs.json"
BASKET_OF = {"ARCH": ("PORT", "ISLE", "AQUA", "GRID", "FERRY", "HAVEN"), "REBLD": ("RECON", "MAT", "GEN")}
# implausible-prediction bound: 1.5x the largest out-of-sample prediction seen in research (% of price)
PRED_MAX = {"PORT": 0.35, "ISLE": 0.20, "AQUA": 0.24, "GRID": 0.15, "FERRY": 0.31, "RECON": 0.16, "MAT": 0.105,
            "GEN": 0.093, "HAVEN": 0.60}
HOUSE_FRACS = (0.1, 0.07, 0.3, 0.24)       # house market-maker quote sizes as fractions of the position limit
SNIPE_EDGE = 0.02                           # take orders more than 2% through the house mid ...
NEWS_COOLDOWN = 15.0                        # ... but never within 15 s of news (the house may simply be lagging),
SNIPE_STABLE, SNIPE_STABLE_SEC = 0.003, 10.0  # ... only if the house mid moved < 0.3% over the last 10 s (no repricing),
SNIPE_PERSIST = 2.0                         # ... only if the same mispriced order is still there after 2 s,
SNIPE_MAX_FRAC = 0.25                       # ... and at most 25% of the position limit per snipe
FOLLOW_WAIT, FOLLOW_MIN, FOLLOW_HOLD = 6.0, 0.0075, 50.0   # announcement follower: watch 6 s, need >=0.75%, hold ~50 s
STUB_WIDTH, STUB_QUIET = 0.04, 20.0         # stubs: 4% from the house mid, only after 20 s without news


class Strategy:
    def __init__(self, models_path: Path, limits: dict, broker, log, clock, persist: bool = True, reflatten: bool = True):
        blob = pickle.loads(Path(models_path).read_bytes())
        self.models, self.consts = blob["models"], blob["consts"]
        self.limits, self.broker, self.log, self.clock = limits, broker, log, clock
        self.persist, self.reflatten = persist, reflatten
        # tested on 100 held-out seasons with storm-time book depth (bot/compare_variants.py):
        #   hold after the storm in round 3 only: $5,729/season (best); flatten every round $5,440; hold every round $3,013
        #   formation front-run: -$280/season (the thin book eats the jump) -> off
        self.hold_rounds: set = {3}                         # rounds in which positions are held after the storm
        self.front_run = False                              # trade the formation jump before the house reprices
        self.final_values = None
        self.season_date = None
        self.allow_flip = False                             # flipping a full position through the thin book costs more than it earns
        self.paused: set = set()                            # symbols not traded (announcement or manual)
        self.halted = False                                 # manual brake (bot/control.json): no automatic trading at all
        self.flattening = False                             # manual {"flatten": true}: close everything, then stay halted
        self.form_frac = 0.0                                # formation sleeve size, fraction of the limits (0 = off)
        self.round_open = None                              # (clock at market open, round, (date, round))
        self.sleeve = None                                  # formation sleeve state for the current round
        try:
            self.form_dirs = json.loads(FORM_DIRS_FILE.read_text())["rounds"]
        except (OSError, ValueError, KeyError):
            self.form_dirs = {}
        # sniper OFF: replaying every recorded live book through it (bot/test_sniper_replay.py) lost money --
        # its fair-value estimate (house quote ladder) can be wrong and stuck, so 'mispriced' orders were mostly not.
        self.sniper_on, self.follow_on, self.stubs_on = False, True, False
        # first storm age at which the in-storm model may trade. At age 10 the model has no skill (out-of-sample R^2 ~ 0)
        # and the house still reprices 16-28 s after formation. Waiting for the first official advisory (age 30):
        #   seasons 501-600: $7,122 -> $8,264/season, P(season>0) 0.79 -> 0.86
        #   seasons 401-500 (fresh model): $7,656 -> $8,603/season, P(season>0) 0.82 -> 0.89
        self.min_age = 30
        self.stop_loss = None                               # $ loss within a storm that triggers flatten + sit out
        self.stopped_round, self.round_start_equity = False, 0.0
        self.max_age = 470                                  # last decision age
        self.last_news_t = -1e9
        self.report_t = None                                # last damage report of the current storm (clock)
        self.last_flatten_t = -1e9
        self.snipes: dict = {}                              # symbol -> (time, base position, house mid, decide count)
        self.hm_hist: dict = {}                             # symbol -> deque of (time, house mid)
        self.sighted: dict = {}                             # (symbol, side) -> (first time seen, price)
        self.follow: dict = {}                              # symbol -> announcement-follower state
        self.n_decisions = 0
        self.storm_syms = list(STORM_SYMS)
        self.sym_buffer: dict = {}                          # symbol -> entry buffer overriding BUFFER (default: none)
        self.fair: dict = {}                                # symbol -> model's estimate of the house price at storm end
        self.replaying = False                              # True while catching up from the recorded feed
        self.season = Season()
        self.storm: Storm | None = None
        self.quotes: dict = {}                              # sym -> (bid, ask)
        self.books: dict = {}                               # sym -> (bids, asks), levels [{price, quantity}]
        self.mid_hist: deque = deque(maxlen=4000)           # (clock, {sym: mid})
        self.Hpre: dict = {}
        self.pending: list = []                             # decision ages waiting for the next observation
        self.seen: set = set()
        self.in_storm_syms: set = set()
        self.decision_ages = set(self.consts["decision_ages"])
        self.features_out = []                              # kept for the replay parity check

    # ---- market data -----------------------------------------------------------------------------
    def on_quote(self, sym, bid, ask):
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return
        self.quotes[sym] = (float(bid), float(ask))
        if len(self.quotes) == len(SYMS):
            self.mid_hist.append((self.clock(), {s: (b + a) / 2 for s, (b, a) in self.quotes.items()}))

    def mids_at(self, seconds_ago=0.0):
        target = self.clock() - seconds_ago
        best = None
        for t, m in reversed(self.mid_hist):
            best = m
            if t <= target:
                break
        return best

    # ---- feed --------------------------------------------------------------------------------------
    def on_news(self, m: dict):
        seq = m.get("seq")
        if seq is not None:                                  # seq restarts every season: key on (date, seq)
            key = (str(m.get("time", ""))[:10], seq)
            if key in self.seen:
                return
            self.seen.add(key)
        typ = m.get("type")
        self._check_season(m)
        if typ in ("storm_formed", "storm_dissipated", "damage_report", "advisory", "announcement", "impact", "news"):
            self.last_news_t = self.clock()
        if typ == "snapshot" and m.get("phase") in ("pre_storm", "closed"):
            self.last_news_t = self.clock()                 # round opens/closes reprice the whole book
        if typ == "storm_formed":
            if m["round"] == 1 and self.season.finals:
                self.new_season("round-1 storm formed")
            self.storm = Storm(m["round"], m["storm"])
            self.Hpre = self.mids_at(2.0) or {}
            self.fair = {}
            self.stopped_round = False
            self.report_t = None
            if hasattr(self.broker, "equity"):
                self.round_start_equity = self.broker.equity()
            self.pending = []
            if self.front_run and not self.replaying and self.season is not None and not self.season.approximate:
                goals = {}
                for x, d in FORM_DIR.items():
                    if x in self.quotes:
                        mid = sum(self.quotes[x]) / 2
                        goals[x] = (d * self.limits[x], mid * (1 + d * FORM_CAP), 1.0)
                self.in_storm_syms |= set(goals)
                self.broker.set_targets(goals, reason="formation front-run", urgent=True)
            self.log("storm_formed", round=m["round"], storm=m["storm"], Hpre={k: round(v, 3) for k, v in (self.Hpre or {}).items()})
        elif typ == "storm" and self._current(m):
            self.storm.add_obs(m)
            a = m["age"]
            for p in [p for p in self.pending if p < a]:
                self.pending.remove(p)
                self.decide(p)
            if a in self.decision_ages and not self.storm.joined_late:
                self.storm.snapshot(a)
                self.storm.hist_mids[a] = (self.mids_at(0.0), self.mids_at(10.0))
                self.pending.append(a)
        elif typ == "advisory" and self._current(m):
            (self.storm.official if m.get("source") == "official" else self.storm.wire).append(m)
        elif typ == "storm_dissipated" and self._current(m):
            self.storm.dissipated = True
            self.pending = []
            if self.storm.round in self.hold_rounds:
                self.broker.set_targets({x: (self.broker.position(x), None, 0.0) for x in STOCKS}, reason="hold after storm")
            else:
                self.flatten_in_storm("storm dissipated")
        elif typ == "damage_report":
            self.log("damage_report", round=m["round"], stage=m["stage"], damage=m["damage_pct"], loss=m["industry_loss_musd"])
            if self.storm is not None and self.storm.round == m["round"]:
                self.report_t = self.clock()
            self.flatten_leftovers()
            if m["stage"] == "final":
                st = self.storm if (self.storm and self.storm.round == m["round"]) else None
                self.season.apply_final(m["round"], m["damage_pct"], st, self.consts["v"])
                self.save_state()
                if m["round"] == 3:
                    self.final_snipe()
        elif typ == "snapshot":
            self._note_open(m)
            self.on_snapshot(m)
        elif typ == "announcement":
            # the company's economics changed: the valuation formulas may now be wrong for these symbols.
            # Stop trading them (positions are held) until a human resumes them via bot/control.json.
            syms = [x for x in (m.get("symbols") or []) if x in self.limits] or list(self.limits)
            self.paused |= set(syms)
            if self.sleeve:                                  # the sleeve stops working them: they are paused/followed now
                for x in syms:
                    self.sleeve["targets"].pop(x, None)
            for x in ("ARCH", "REBLD"):                      # ETFs contain the paused stocks
                if any(y in BASKET_OF.get(x, ()) for y in syms):
                    self.paused.add(x)
            self.broker.set_targets({x: (self.broker.position(x), None, 0.0) for x in self.paused if x in self.quotes},
                                    reason="announcement pause")
            now = self.clock()
            for x in syms:
                if x in STOCKS and self.follow_on and not self.replaying and not self.halted:
                    self.follow[x] = {"t0": now, "h0": self.top_mid(x), "state": "watch", "base": self.broker.position(x)}
            self.log("ANNOUNCEMENT - PAUSED", headline=m.get("headline"), symbols=syms, paused=sorted(self.paused))

    # ---- other participants: house fair value, mispriced orders, announcements, stubs ------------------------
    def top_mid(self, s):
        return sum(self.quotes[s]) / 2 if s in self.quotes else None

    def house_mid(self, s):
        """Fair value as quoted by the house market makers: their quote sizes are fixed fractions of the limit, so
        other teams' orders (any other size) don't distort it. Falls back to the top-of-book mid."""
        bids, asks = self.books.get(s, ([], []))
        mids = []
        for f in HOUSE_FRACS:
            q = round(f * self.limits[s])
            bp = [l["price"] for l in bids if l["quantity"] == q]; ap = [l["price"] for l in asks if l["quantity"] == q]
            if bp and ap:
                mids.append((max(bp) + min(ap)) / 2)
        if mids:
            return float(np.median(mids))
        return sum(self.quotes[s]) / 2 if s in self.quotes else None

    def on_book(self, s):
        """Sniper: take an order priced more than SNIPE_EDGE through the house mid (a broken bot's order), then
        unwind at fair value. Only when the house is NOT repricing: no news/round/season change for 15 s, house mid
        stable for 10 s, and the order has sat there for 2 s. Repricing (news, season rollover) looks identical to a
        mispriced order for a few seconds -- the 2026-10-18 season-80 open fired 5 false snipes before these rules."""
        now = self.clock()
        hm = self.house_mid(s)
        if hm:
            h = self.hm_hist.setdefault(s, deque(maxlen=200)); h.append((now, hm))
        if (not self.sniper_on or self.halted or self.replaying or s in self.paused or s in self.follow
                or now - self.last_news_t < NEWS_COOLDOWN or self.final_values is not None or s in self.snipes):
            return
        bids, asks = self.books.get(s, ([], []))
        if not hm or not bids or not asks:
            return
        recent = [m for t, m in self.hm_hist[s] if now - t <= SNIPE_STABLE_SEC]
        span = (self.hm_hist[s][-1][0] - self.hm_hist[s][0][0]) if self.hm_hist[s] else 0
        if span < SNIPE_STABLE_SEC or (max(recent) - min(recent)) / hm > SNIPE_STABLE:
            self.sighted.pop((s, "buy"), None); self.sighted.pop((s, "sell"), None)
            return
        pos, lim = self.broker.position(s), self.limits[s]
        cap_q = max(1, int(lim * SNIPE_MAX_FRAC))
        cheap = [l for l in asks if l["price"] <= hm * (1 - SNIPE_EDGE)]
        rich = [l for l in bids if l["price"] >= hm * (1 + SNIPE_EDGE)]
        side = "buy" if cheap and pos < lim else ("sell" if rich and pos > -lim else None)
        for sd in ("buy", "sell"):
            if sd != side:
                self.sighted.pop((s, sd), None)
        if side is None:
            return
        best = cheap[0]["price"] if side == "buy" else rich[0]["price"]
        seen = self.sighted.get((s, side))
        if seen is None or abs(seen[1] - best) > 1e-9:
            self.sighted[(s, side)] = (now, best); return       # first sighting: wait to see that it persists
        if now - seen[0] < SNIPE_PERSIST:
            return
        if side == "buy":
            q = min(sum(l["quantity"] for l in cheap), lim - pos, cap_q); goal = (pos + q, hm * (1 - SNIPE_EDGE), 1e9)
        else:
            q = min(sum(l["quantity"] for l in rich), lim + pos, cap_q); goal = (pos - q, hm * (1 + SNIPE_EDGE), 1e9)
        self.sighted.pop((s, side), None)
        self.snipes[s] = (now, pos, hm, self.n_decisions)
        self.broker.set_targets({s: goal}, reason="snipe mispriced order", urgent=True)
        self.log("SNIPE", symbol=s, house_mid=round(hm, 3), qty=q, side=side, cap=round(goal[1], 3))

    def on_stub_fill(self, s, side, qty, price):
        """A stub was hit: someone traded far through fair value. Unwind it at fair value."""
        hm = self.top_mid(s) or price
        pos = self.broker.position(s)
        tgt = pos - qty if side == "buy" else pos + qty
        cap = hm * (1 - 0.003) if side == "buy" else hm * (1 + 0.003)
        self.broker.set_targets({s: (tgt, cap, 1e9)}, reason="unwind stub fill", urgent=True)
        self.log("STUB FILLED", symbol=s, side=side, qty=qty, price=price, house_mid=round(hm, 3))

    def _sleeve_syms(self):
        """Symbols the formation sleeve is still working (excluded from the leftover/after-storm exits). A paused
        symbol (announcement or manual pause) is never traded by the sleeve."""
        sl = self.sleeve
        return {s for s in sl["targets"] if s not in self.paused} if sl and sl["state"] in ("entry", "hold", "exit") else set()

    def _note_open(self, m):
        """Time the round's open from the first pre-storm snapshot whose market clock is running (past 09:30:00;
        real seconds = market seconds / 30). A 09:30:00 snapshot is skipped: the clock may be paused there."""
        if self.replaying or m.get("phase") != "pre_storm":
            return
        t = str(m.get("time", ""))
        try:
            el = ((int(t[11:13]) - 9) * 3600 + (int(t[14:16]) - 30) * 60 + int(t[17:19])) / 30.0
        except ValueError:
            return
        key = (t[:10], m.get("round"))
        if not 0 < el < 60 or (self.round_open and self.round_open[2] == key):
            return
        self.round_open = (self.clock() - el, m.get("round"), key)
        self.sleeve = {"state": "wait", "round": m.get("round"), "last": -1e9, "targets": {}} if self.form_frac > 0 else None
        self.log("market open", round=m.get("round"), seconds_since_open=round(el, 1), sleeve=self.form_frac)

    def _sleeve_tick(self, now):
        sl = self.sleeve
        if not sl or self.round_open is None or self.season.approximate or self.replaying:
            return
        t_open, rnd, _ = self.round_open
        el = now - t_open
        dirs = self.form_dirs.get(str(rnd), {})
        st = self.storm
        formed = st is not None and st.round == rnd and bool(st.obs) and not st.joined_late
        age = max(st.obs) if formed else None
        if sl["state"] == "wait":
            if formed or el > FORM_ENTRY[1]:
                sl["state"] = "done"; return                    # opened too late to enter before the storm
            if el >= FORM_ENTRY[0]:
                sl["state"] = "entry"
                sl["targets"] = {s: (1 if d["move"] > 0 else -1) * int(self.limits[s] * self.form_frac)
                                 for s, d in dirs.items() if s in self.limits and s not in self.paused
                                 and abs(d["move"]) / d["price"] >= 2 * FORM_BAND and int(self.limits[s] * self.form_frac)}
                self.log("FORMATION SLEEVE entry", round=rnd, targets=sl["targets"])
        if sl["state"] == "entry":
            if formed or el > FORM_ENTRY[1]:
                sl["state"] = "hold"
                self.broker.set_targets({s: (self.broker.position(s), None, 0.0) for s in self._sleeve_syms()}, reason="formation sleeve hold")
                self.log("FORMATION SLEEVE holding through formation", positions={s: self.broker.position(s) for s in sl["targets"]})
            elif now - sl["last"] >= 4.0:
                sl["last"] = now
                goals = {}
                for s, tgt in sl["targets"].items():
                    if s in self.quotes and s not in self.paused:
                        mid = sum(self.quotes[s]) / 2; d = 1 if tgt > 0 else -1
                        goals[s] = (tgt, mid * (1 + d * FORM_BAND), abs(dirs[s]["move"]) * abs(tgt), FORM_BAND)
                if goals:
                    self.broker.set_targets(goals, reason="formation sleeve entry")
        if sl["state"] == "hold" and age is not None and age >= FORM_EXIT[0]:
            sl["state"] = "exit"; sl["last"] = -1e9
        if sl["state"] == "exit":
            if age is not None and age >= FORM_EXIT[1]:
                syms = self._sleeve_syms()
                sl["state"] = "done"
                left = {s: self.broker.position(s) for s in syms if self.broker.position(s)}
                self.broker.set_targets({s: (self.broker.position(s), None, 0.0) for s in syms}, reason="formation sleeve done")
                self.log("FORMATION SLEEVE done: leftover handed to the in-storm model", leftover=left)
            elif now - sl["last"] >= 3.0:
                sl["last"] = now
                goals = {}
                for s in self._sleeve_syms():
                    p = self.broker.position(s)
                    if p and s in self.quotes:
                        mid = sum(self.quotes[s]) / 2; d = 1 if p > 0 else -1
                        goals[s] = (0, mid * (1 - d * FORM_BAND), abs(p) * abs(dirs[s]["move"]), FORM_BAND)
                if goals:
                    self.broker.set_targets(goals, reason="formation sleeve exit")

    def tick(self):
        """Called every second by live.py: snipe unwinds, the announcement follower, stub quotes."""
        if self.replaying:
            return
        now = self.clock()
        # manual flatten: keep closing from the current mid until flat (a fixed cap can go stale, like HAVEN in 74 r1)
        if self.flattening:
            if now - self.last_flatten_t >= LEFTOVER_RETRY:
                if any(self.broker.position(x) for x in self.limits):
                    self.manual_flatten(retry=True)
                else:
                    self.flattening = False
                    self.log("manual flatten complete: flat, still halted")
            return
        if self.halted:
            self.follow.clear(); self.snipes.clear()            # the brake stops every automatic trade; positions are held
            self.broker.sync_stubs({})
            return
        self._sleeve_tick(now)
        # 0. rounds 1-2: be flat before the next storm forms (its formation reprices everything). An exit whose cap
        #    went stale -- the price drifted past it, live season 74 r1 carried HAVEN -142 into the next storm and
        #    lost ~$1.1k on the jump -- is retried from the current mid every LEFTOVER_RETRY s once a report is out.
        st = self.storm
        if (self.reflatten and st is not None and st.dissipated and st.round not in self.hold_rounds
                and self.report_t is not None and now - self.last_flatten_t >= LEFTOVER_RETRY):
            left = {x: self.broker.position(x) for x in self.limits
                    if self.broker.position(x) and x not in self.follow and x not in self._sleeve_syms()}
            if left:
                self.log("LEFTOVER re-flatten from current mid", positions=left)
                self.flatten_leftovers()
        # 1. unwind snipes at fair value after 3 s, unless the in-storm model has decided since
        for s, (t, base, hm, nd) in list(self.snipes.items()):
            if now - t < 3:
                continue
            self.snipes.pop(s)
            if (self.n_decisions == nd or s not in self.storm_syms) and s in self.quotes:
                cur = self.broker.position(s)
                if cur != base:
                    cap = hm * (1 + 0.003) if base > cur else hm * (1 - 0.003)
                    self.broker.set_targets({s: (base, cap, 1e9)}, reason="unwind snipe")
        # 2. announcement follower: the house takes ~45 s to adjust; follow the direction of its first 6 s
        for s, f in list(self.follow.items()):
            hm = self.top_mid(s)
            if hm is None or f["h0"] is None:
                self.follow.pop(s); continue
            if f["state"] == "watch" and now - f["t0"] >= FOLLOW_WAIT:
                move = (hm - f["h0"]) / f["h0"]
                if abs(move) < FOLLOW_MIN:
                    self.follow.pop(s); self.log("announcement follower: no clear house move", symbol=s, move=round(move, 4)); continue
                d = 1 if move > 0 else -1
                tgt = d * max(self.limits[s] // 2, d * f["base"]) if d * f["base"] > 0 else d * (self.limits[s] // 2)
                cap = hm * (1 + d * abs(move))                 # pay at most one more 'first move' beyond now
                f.update(state="in", d=d, h_in=hm, move=abs(move))
                self.broker.set_targets({s: (tgt, cap, 1e9)}, reason="announcement follower", urgent=True)
                self.log("ANNOUNCEMENT FOLLOW", symbol=s, direction=d, first_move=round(move, 4), target=tgt, cap=round(cap, 3))
            elif f["state"] == "in":
                reversed_ = f["d"] * (hm - f["h_in"]) / f["h_in"] < -f["move"] / 2
                if now - f["t0"] >= FOLLOW_HOLD or reversed_:
                    half = (self.quotes[s][1] - self.quotes[s][0]) / 2 if s in self.quotes else 0.0
                    cur = self.broker.position(s)
                    cap = hm - half - 0.0015 * hm if cur > f["base"] else hm + half + 0.0015 * hm
                    self.broker.set_targets({s: (f["base"], cap, 1e9)}, reason="announcement follower exit")
                    self.follow.pop(s)
                    self.log("announcement follower exit", symbol=s, reversed=reversed_)
        # 3. stub quotes far from fair value, only in quiet periods
        want = {}
        quiet = now - self.last_news_t >= STUB_QUIET
        st = self.storm
        if st is not None and not st.dissipated and st.obs:
            age = max(st.obs)
            if 25 <= age % 60 <= 40 or age < 60:              # advisory windows and the formation repricing
                quiet = False
        if self.stubs_on and not self.halted and quiet:
            for s in self.limits:
                hm = self.top_mid(s)
                if hm and s not in self.paused and s not in self.follow:
                    q = max(1, self.limits[s] // 4)
                    want[(s, "buy")] = (hm * (1 - STUB_WIDTH), q)
                    want[(s, "sell")] = (hm * (1 + STUB_WIDTH), q)
        self.broker.sync_stubs(want)

    def new_season(self, why: str):
        """A new season (the testnet replays one after another): positions are reset by the exchange."""
        self.last_news_t = self.clock()                     # the whole book reprices at a season change
        self.snipes, self.sighted = {}, {}
        self.season = Season()
        self.storm, self.in_storm_syms, self.final_values, self.fair = None, set(), None, {}
        self.season_date = None
        self.broker.clear()
        self.log("new season", why=why)

    def _check_season(self, m):
        """Each season's rounds are consecutive market days; a jump of more than a few days = a new season."""
        d = str(m.get("time", ""))[:10]
        if len(d) != 10:
            return
        day = date.fromisoformat(d)
        if self.season_date is not None and abs((day - self.season_date).days) > 5:
            self.new_season(f"market date jumped {self.season_date} -> {day}")
        if self.season_date is None:
            self.season_date = day

    def _current(self, m):
        return self.storm is not None and m.get("round") == self.storm.round and m.get("storm") == self.storm.name

    def save_state(self):
        """Persist the season so a restart within the same season recovers it exactly."""
        if self.season.approximate or not self.persist:
            return
        STATE_FILE.write_text(json.dumps({"finals": {str(r): d for r, d in self.season.finals.items()},
                                          "D_site": self.season.D_site.tolist(), "C_prev": self.season.C_prev}))

    def on_snapshot(self, s: dict):
        """Late join: restore the saved season if it matches the snapshot's final reports; otherwise rebuild the
        damage from the snapshot and mark the season approximate (outage history unknown)."""
        if s.get("phase") in ("storm", "aftermath") and (self.storm is None or self.storm.round != s.get("round")):
            # restarted mid-round: adopt the storm (no in-storm decisions) so held positions still get flattened
            self.storm = Storm(s["round"], s["storm"], joined_late=True)
            self.storm.dissipated = s.get("phase") == "aftermath"
            self.in_storm_syms |= {x for x in STOCKS if self.broker.position(x) != 0}
            self.log("adopted in-progress storm after restart", round=s["round"], storm=s["storm"], phase=s.get("phase"))
        st = self.storm
        if st is not None and not st.joined_late and st.round == s.get("round") and st.name == s.get("storm") and st.obs:
            # after a reconnect: fill missed observations of the current storm from the snapshot's track
            last_a = max(st.obs); last = st.obs[last_a]
            o = s.get("obs") or {}
            pts = [p for p in (s.get("track") or []) if p[0] > last_a]
            if pts and (pts[-1][0] - last_a) > 2:
                end_age, end_i = o.get("age", pts[-1][0]), o.get("intensity", last["intensity"])
                for a, x, y, r in pts:
                    f = (a - last_a) / max(end_age - last_a, 1e-9)
                    st.add_obs({"age": a, "x": x, "y": y, "vx": 0.0, "vy": 0.0, "radius": r,
                                "intensity": last["intensity"] + f * (end_i - last["intensity"])})
                if o.get("age", -1) > max(st.obs):
                    st.add_obs({**o})
                self.log("filled observation gap from snapshot", round=st.round, frm=last_a, to=max(st.obs))
        snap = {rep["round"]: rep["damage_pct"] for rep in (s.get("damage") or []) if rep.get("stage") == "final"}
        if not snap or all(r in self.season.finals for r in snap):
            return
        missing = [r for r in snap if r not in self.season.finals]
        if st is not None and not st.joined_late and missing == [st.round] and all(r in self.season.finals for r in snap if r != st.round):
            self.season.apply_final(st.round, snap[st.round], st, self.consts["v"])   # missed only this storm's final
            self.save_state()
            self.log("applied missed final report from snapshot", round=st.round)
            return
        try:
            saved = json.loads(STATE_FILE.read_text())
        except (OSError, ValueError):
            saved = None
        if saved and {int(r): d for r, d in saved["finals"].items()} == snap:
            self.season.finals = dict(snap)
            self.season.D_site = np.array(saved["D_site"])
            self.season.C_prev = saved["C_prev"]
            self.log("restored season state from bot/state.json", rounds=sorted(snap))
            return
        for r in sorted(snap):
            if r not in self.season.finals:
                self.season.apply_final(r, snap[r], None, self.consts["v"])
        self.log("JOINED LATE: outage history unknown -> no in-storm trading this season; "
                 "final-report trade limited to outage-free symbols", rounds=sorted(snap))
        obs = s.get("obs")
        if s.get("phase") == "storm" and self.storm is None and obs:
            self.log("joined mid-storm: no in-storm trading until the next storm")

    # ---- decisions ---------------------------------------------------------------------------------
    def decide(self, age):
        st = self.storm
        if self.replaying:
            return
        if self.season.approximate or age < self.min_age or age > self.max_age or self.stopped_round:
            return
        if self.stop_loss is not None and hasattr(self.broker, "equity"):
            if self.broker.equity() - self.round_start_equity < -self.stop_loss:
                self.stopped_round = True
                self.log("ROUND STOP-LOSS: flattening and sitting out the rest of the storm", age=age)
                self.flatten_in_storm("round stop-loss")
                return
        mids, lag = st.hist_mids.get(age, (None, None))
        if not mids or not lag or not self.Hpre or age not in st.hist:
            self.log("skip decision (no quotes yet)", age=age)
            return
        f = features(st, self.season, self.consts, age, mids, self.Hpre, lag)
        assert f["names"] == self.consts["feature_names"], "feature order differs from training"
        self.features_out.append({"round": st.round, "age": age, **{s: f["rows"][s] for s in STOCKS}})
        goals, preds = {}, {}
        for s in self.storm_syms:
            pred = float(self.models[s].predict(np.array([f["rows"][s]], dtype=float))[0])
            preds[s] = round(pred, 3)
            if s not in self.quotes or s in self.paused or self.halted:
                continue
            if abs(pred) > PRED_MAX[s] * sum(self.quotes[s]) / 2:   # beyond anything seen in research: a feature bug
                self.log("PREDICTION REJECTED (implausible)", symbol=s, pred=pred)
                continue
            bid, ask = self.quotes[s]
            mid = (bid + ask) / 2
            self.fair[s] = mids[s] + pred                       # where the house price should be at storm end
            cur, sgn = self.broker.position(s), int(np.sign(pred))
            buf = self.sym_buffer.get(s, BUFFER)
            if cur != 0 and np.sign(cur) != sgn and abs(pred) * EDGE_SHARE > buf * mid and self.allow_flip:
                # against the prediction and the prediction is big enough to enter: flip to the full limit
                cap = mid + sgn * (abs(pred) * EDGE_SHARE - buf * mid)
                goals[s] = (sgn * self.limits[s], cap, abs(pred) * (self.limits[s] + abs(cur)))
            elif cur != 0 and np.sign(cur) != sgn:
                # against a small prediction: close, paying up to the whole predicted move
                goals[s] = (0, mid + sgn * max(abs(pred) - buf * mid, 0.0), abs(pred) * abs(cur))
            elif abs(pred) * EDGE_SHARE > buf * mid:
                # add toward the full limit, paying at most half the predicted move
                cap = mid + sgn * (abs(pred) * EDGE_SHARE - buf * mid)
                goals[s] = (sgn * self.limits[s], cap, abs(pred) * self.limits[s])
            else:
                goals[s] = (cur, mid, 0.0)                  # small prediction: hold (and cancel any older goal)
        self.in_storm_syms |= set(goals)
        self.n_decisions += 1
        self.log("decision", round=st.round, age=age, preds=preds,
                 goals={s: (t, round(c, 3)) for s, (t, c, _) in goals.items()})
        self.broker.set_targets(goals, reason=f"in-storm r{st.round} age {age}")

    def flatten_in_storm(self, reason, after_report=False):
        """Close in-storm positions, but never worse than fair value -/+ a small margin: the live book can
        run several dollars away from the house's fair value during a storm and snap back right after it."""
        if self.replaying:
            return
        if self.halted:
            self.log("halted: after-storm exit skipped (positions held)", reason=reason)
            return
        self.last_flatten_t = self.clock()
        goals = {}
        for s in self.in_storm_syms - self._sleeve_syms():
            if s in self.quotes:
                bid, ask = self.quotes[s]
                mid, half, cur = (bid + ask) / 2, (ask - bid) / 2, self.broker.position(s)
                if cur == 0:
                    continue
                fair, m = (mid, REPORT_EXIT_MARGIN) if after_report else (self.fair.get(s, mid), EXIT_MARGIN)
                slack = fair * m + half                          # cross a normal spread, never chase a runaway book
                goals[s] = (0, fair + slack if cur < 0 else fair - slack, abs(cur) * mid)
        self.in_storm_syms = set()
        if goals:
            self.broker.set_targets(goals, reason=reason, urgent=True)

    def manual_flatten(self, retry=False):
        """{"flatten": true}: close every position within 1% of the current mid, then stay halted. tick() retries
        from the current mid every LEFTOVER_RETRY s until flat."""
        self.halted, self.flattening = True, True
        self.last_flatten_t = self.clock()
        goals = {}
        for x in self.limits:
            q = self.broker.position(x)
            if q and x in self.quotes:
                m_ = sum(self.quotes[x]) / 2
                goals[x] = (0, m_ * 1.01 if q < 0 else m_ * 0.99, abs(q) * m_)
            else:
                goals[x] = (q, None, 0.0)                    # HOLD: stops any entry still being worked in this symbol
        self.broker.set_targets(goals, reason="manual flatten (retry)" if retry else "manual flatten", urgent=True)

    def flatten_leftovers(self):
        """Positions the post-storm flatten could not close (cap too tight): retry from current mids."""
        if not self.reflatten or self.storm is None or not self.storm.dissipated or self.storm.round in self.hold_rounds:
            return
        self.in_storm_syms |= {s for s in self.limits if self.broker.position(s) != 0 and s not in self.follow
                               and s not in self._sleeve_syms()}
        self.flatten_in_storm("re-flatten after report", after_report=True)

    def final_snipe(self):
        if self.replaying:
            return
        vals = exact_values(self.season, self.storm)
        self.final_values = vals                              # settlement marks for the season result
        goals = {}
        for s in SYMS:
            if s not in self.quotes or (self.season.approximate and s not in OUTAGE_FREE) or s in self.paused or self.halted:
                continue
            bid, ask = self.quotes[s]
            margin = SNIPE_MARGIN.get(s, DEFAULT_MARGIN)
            if bid > vals[s] + margin:
                goals[s] = (-self.limits[s], vals[s] + margin, (bid - vals[s]) * self.limits[s])
            elif ask < vals[s] - margin:
                goals[s] = (self.limits[s], vals[s] - margin, (vals[s] - ask) * self.limits[s])
            else:
                goals[s] = (self.broker.position(s), vals[s], 0.0)   # no edge: hold to settlement
        self.log("final snipe", values={k: round(v, 3) for k, v in vals.items()},
                 goals={s: (t, round(c, 3)) for s, (t, c, _) in goals.items()}, approximate=self.season.approximate)
        self.broker.set_targets(goals, reason="round-3 final report", urgent=True)
