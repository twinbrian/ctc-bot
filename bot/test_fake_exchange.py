"""End-to-end checks of strategy + LiveBroker against an in-memory fake exchange (no network).

    /opt/anaconda3/bin/python3 -I bot/test_fake_exchange.py
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot"))

import broker as B
import strategy as S

B.RETRY_GAP, B.CANCEL_AFTER = 0.4, 0.3
LIMITS = json.loads((ROOT / "case.json").read_text())["limits"]
SYM = "GRID"                                     # limit 400 -> house sizes 40, 28, 120, 96, 1200


class FakeExchange:
    """House ladder around `mid` for one symbol; our orders match it, remainders rest with an id."""

    def __init__(self, mid=40.0):
        self.mid, self.lock = mid, threading.Lock()
        self.pos, self.avg, self.resting, self.trades = 0, 0.0, {}, []
        self.ids = itertools.count(1)
        self.extra_asks, self.extra_bids = [], []    # other teams' orders

    def book(self):
        m, L = self.mid, LIMITS[SYM]
        bids = [{"price": round(m - d, 2), "quantity": round(f * L)} for d, f in ((0.05, .1), (0.08, .07), (0.15, .3), (0.25, .24))]
        asks = [{"price": round(m + d, 2), "quantity": round(f * L)} for d, f in ((0.05, .1), (0.08, .07), (0.15, .3), (0.25, .24))]
        bids = sorted(bids + self.extra_bids, key=lambda l: -l["price"]); asks = sorted(asks + self.extra_asks, key=lambda l: l["price"])
        return bids, asks

    def _fill(self, side, qty, price):
        sg = 1 if side == "buy" else -1
        if self.pos == 0 or (self.pos > 0) == (sg > 0):
            self.avg = (self.avg * abs(self.pos) + price * qty) / (abs(self.pos) + qty)
        self.pos += sg * qty
        self.trades.append({"trade_id": f"t{len(self.trades)}", "symbol": SYM, "quantity": qty, "price": price,
                            "side": side, "executed_at": "now", "kind": "trade"})

    def call(self, method, path, body=None, api=True):
        with self.lock:
            if method == "GET" and path == "/positions":
                return 200, {"positions": [{"symbol": SYM, "quantity": self.pos, "average_price": self.avg}] if self.pos else []}
            if method == "GET" and path == "/orders/open":
                return 200, {"orders": [dict(o, order_id=k) for k, o in self.resting.items()]}
            if method == "GET" and path == "/trades":
                return 200, {"trades": list(self.trades[-100:])}
            if method == "DELETE" and path.startswith("/orders/"):
                self.resting.pop(path.rsplit("/", 1)[1], None); return 200, {}
            if method == "POST" and path == "/orders":
                if abs(body["price"] * 100 - round(body["price"] * 100)) > 1e-6:
                    return 400, '{"detail":"Price must be a multiple of 0.01"}'
                buy, left = body["side"] == "buy", body["quantity"]
                bids, asks = self.book()
                for lv in (asks if buy else bids):
                    if (buy and lv["price"] > body["price"]) or (not buy and lv["price"] < body["price"]) or left == 0:
                        break
                    t = min(left, lv["quantity"]); left -= t; self._fill(body["side"], t, lv["price"])
                    if lv in self.extra_asks or lv in self.extra_bids:          # other teams' orders are consumed
                        lv["quantity"] -= t
                        if lv["quantity"] == 0:
                            (self.extra_asks if buy else self.extra_bids).remove(lv)
                oid = str(next(self.ids))
                if left:
                    self.resting[oid] = {"symbol": SYM, "side": body["side"], "price": body["price"], "quantity": left, "filled_quantity": 0}
                return 200, {"order_id": oid, "status": "filled" if not left else "pending"}
            return 404, None

    def hit_resting(self, side):
        """A broken bot sweeps through one of our resting orders on `side`."""
        with self.lock:
            for k, o in list(self.resting.items()):
                if o["side"] == side:
                    self._fill(side, o["quantity"], o["price"]); self.resting.pop(k); return o
        return None


def setup(fx):
    events = []
    log = lambda ev, **kw: events.append((ev, kw))
    st = S.Strategy(ROOT / "bot" / "models.pkl", LIMITS, None, log, clock=time.monotonic, persist=False)
    b = B.LiveBroker([fx], st.quotes, st.books, log, LIMITS)
    b.on_stub_fill = st.on_stub_fill
    st.broker = b

    def refresh():
        bids, asks = fx.book()
        st.books[SYM] = (bids, asks); st.quotes[SYM] = (bids[0]["price"], asks[0]["price"])
        b.last_frame = time.monotonic()
    refresh()
    return st, b, events, refresh


def wait(sec, refresh):
    t = time.time()
    while time.time() - t < sec:
        refresh(); time.sleep(0.1)


def main():
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, info)

    # 1. HOLD never trades even if the position differs from the target
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos = 50; fx.avg = 40.0; wait(1.2, refresh)
    b.set_targets({SYM: (0, None, 0.0)}, reason="hold"); wait(1.5, refresh)
    check("1 HOLD goal sends no orders", not [e for e, _ in ev if e == "order"], f"pos {fx.pos}")

    # 2. sniper takes an order 3% under the house mid, then unwinds at fair value
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.last_news_t = -1e9; st.sniper_on = True; S.SNIPE_STABLE_SEC = 0.0; S.SNIPE_PERSIST = 0.0
    fx.extra_asks = [{"price": round(40.0 * 0.97, 2), "quantity": 33}]
    refresh(); st.on_book(SYM); wait(1.5, refresh)
    refresh(); st.on_book(SYM); wait(1.5, refresh)      # second sighting (persistence rule)
    check("2a sniper (mechanics only; OFF in production) bought the mispriced order", fx.pos > 0, f"pos {fx.pos}")
    wait(3.5, lambda: (refresh(), st.tick())); wait(2.0, refresh)
    check("2b snipe unwound at fair value", fx.pos == 0, f"pos {fx.pos}, trades {[(t['side'], t['quantity'], t['price']) for t in fx.trades]}")

    # 3. no sniping within the news cooldown
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.last_news_t = time.monotonic(); st.sniper_on = True
    fx.extra_asks = [{"price": round(40.0 * 0.97, 2), "quantity": 33}]
    refresh(); st.on_book(SYM); wait(1.0, refresh)
    check("3 no snipe right after news", fx.pos == 0, f"pos {fx.pos}")

    # 4. announcement follower (OFF by default): house moves +2% in the first seconds -> follow (half limit), exit
    S.FOLLOW_WAIT, S.FOLLOW_HOLD = 1.0, 3.0
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.on_news({"type": "announcement", "round": 1, "storm": "X", "time": "2025-08-04T11:00:00", "headline": "test",
                "symbols": [SYM], "seq": 1})
    fx.mid = 40.8; wait(2.5, lambda: (refresh(), st.tick()))
    check("4 follower off by default: the announcement freezes the symbol and nothing is traded",
          SYM in st.paused and not st.follow_on and fx.pos == 0 and not fx.trades, f"pos {fx.pos}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.follow_on = True                                        # the follower's mechanics, when switched on
    st.on_news({"type": "announcement", "round": 1, "storm": "X", "time": "2025-08-04T11:00:00", "headline": "test",
                "symbols": [SYM], "seq": 1})
    check("4a announcement pauses the symbol", SYM in st.paused)
    fx.mid = 40.8; wait(1.3, lambda: (refresh(), st.tick())); wait(1.0, refresh)
    check("4b follower bought half the limit", fx.pos == LIMITS[SYM] // 2, f"pos {fx.pos}")
    fx.mid = 41.2; wait(3.0, lambda: (refresh(), st.tick())); wait(2.0, refresh)
    check("4c follower exited", fx.pos == 0, f"pos {fx.pos}")

    # 5/6. stubs: placed 4% away, survive; a broken bot hits one -> unwound at fair; news cancels stubs
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.stubs_on, st.last_news_t = True, -1e9
    wait(1.5, lambda: (refresh(), st.tick()))
    rest = list(fx.resting.values())
    check("5a two stubs resting at +/-4%", len(rest) == 2 and all(abs(o["price"] / 40.0 - 1) > 0.035 for o in rest), str(rest))
    hit = fx.hit_resting("buy"); wait(3.0, lambda: (refresh(), st.tick()))
    check("5b stub fill unwound", hit and fx.pos == 0, f"pos {fx.pos}, trades {[(t['side'], t['quantity'], t['price']) for t in fx.trades]}")
    st.last_news_t = time.monotonic(); wait(1.5, lambda: (refresh(), st.tick()))
    check("6 news cancels stubs", not fx.resting, str(fx.resting))

    # 7. dead feed: nothing is sent
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    b.last_frame = time.monotonic() - 60
    b.set_targets({SYM: (40, 41.0, 1.0)}, reason="test", urgent=True); time.sleep(1.0)
    check("7 no orders on a dead feed", fx.pos == 0 and not [e for e, _ in ev if e == "order"])

    # 8. rounds 1-2: an exit whose cap went stale is retried from the current mid after a report (be flat before
    #    the next storm); round 3 positions are held to settlement
    S.LEFTOVER_RETRY = 1.0
    from engine import Storm
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    st.storm = Storm(1, "X"); st.storm.dissipated = True
    b.set_targets({SYM: (0, 39.9, 1e9)}, reason="stale exit cap", urgent=True)    # below the ask: never fills
    wait(1.5, refresh)
    check("8a stale exit cap leaves the position open", fx.pos == -50, f"pos {fx.pos}")
    st.report_t = time.monotonic()
    wait(3.0, lambda: (refresh(), st.tick())); wait(1.5, refresh)
    check("8b leftover re-flattened from the current mid", fx.pos == 0,
          f"pos {fx.pos}, trades {[(t['side'], t['quantity'], t['price']) for t in fx.trades]}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    st.storm = Storm(3, "X"); st.storm.dissipated = True; st.report_t = time.monotonic()
    wait(2.5, lambda: (refresh(), st.tick()))
    check("8c round 3: position held to settlement", fx.pos == -50, f"pos {fx.pos}")

    # 9. the brake ({"halt": true}) stops every automatic trade; {"flatten": true} closes everything, retrying
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    st.halted = True
    st.storm = Storm(1, "X"); st.storm.dissipated = True; st.in_storm_syms = {SYM}
    st.flatten_in_storm("storm dissipated"); wait(1.0, refresh)
    check("9a halted: no after-storm exit", fx.pos == -50 and not [e for e, _ in ev if e == "order"], f"pos {fx.pos}")
    st.report_t = time.monotonic(); st.last_flatten_t = -1e9
    wait(2.5, lambda: (refresh(), st.tick()))
    check("9b halted: no leftover re-flatten", fx.pos == -50 and not [e for e, _ in ev if e == "order"], f"pos {fx.pos}")
    S.FOLLOW_WAIT = 1.0; st.follow_on = True                   # even with the follower on, the brake stops it
    st.on_news({"type": "announcement", "round": 1, "storm": "X", "time": "2025-08-04T11:00:00", "headline": "test",
                "symbols": [SYM], "seq": 2})
    fx.mid = 40.8; wait(2.5, lambda: (refresh(), st.tick()))
    check("9c halted: announcement pauses but the follower does not trade", SYM in st.paused and fx.pos == -50, f"pos {fx.pos}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    fx.mid = 41.5; st.quotes[SYM] = (39.95, 40.05)                 # the bot still sees the old price: a stale 1% cap
    st.manual_flatten(); wait(0.6, refresh)
    check("9d flatten: the first close hits a stale cap and does not fill", fx.pos == -50 and st.halted, f"pos {fx.pos}")
    wait(3.0, lambda: (refresh(), st.tick())); wait(1.0, refresh)
    check("9e flatten: retried from the current mid until flat, still halted", fx.pos == 0 and st.halted and not st.flattening,
          f"pos {fx.pos}, trades {[(t['side'], t['quantity'], t['price']) for t in fx.trades]}")

    # 10. per-order band (the formation sleeve's orders): whatever the cap, no order further than band x mid from mid
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    b.set_targets({SYM: (300, 41.0, 1.0, 0.002)}, reason="band test", urgent=True); wait(1.5, refresh)
    prices = sorted({t["price"] for t in fx.trades})
    check("10a band 0.2%: only levels within 0.2% of mid (40.05, 40.08) fill", fx.trades and max(prices) <= 40.08, f"prices {prices}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    b.set_targets({SYM: (300, 41.0, 1.0)}, reason="no band", urgent=True); wait(1.0, refresh)
    prices = sorted({t["price"] for t in fx.trades})
    check("10b without a band the deep levels fill (as today)", max(prices) >= 40.15, f"prices {prices}")

    # 11. formation sleeve: enters after market open within 0.5% of mid, holds through formation, exits at ages 10-28
    FE = S.FORM_ENTRY; S.FORM_ENTRY = (20.0, 58.0)
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.form_frac = 0.2
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 1, "storm": "X", "time": "2025-08-04T09:40:00", "damage": []})
    wait(2.5, lambda: (refresh(), st.tick()))
    tgt = int(LIMITS[SYM] * 0.2) * (1 if st.form_dirs["1"][SYM]["move"] > 0 else -1)
    check("11a sleeve entered the target before the storm", fx.pos == tgt and all(abs(t["price"] / 40 - 1) <= 0.005 for t in fx.trades),
          f"pos {fx.pos}, target {tgt}, prices {sorted({t['price'] for t in fx.trades})}")
    obs = lambda a, q: {"type": "storm", "round": 1, "storm": "X", "time": "2025-08-04T10:00:00", "age": a, "x": 480.0, "y": 740.0 - 2 * a,
                        "vx": 0.0, "vy": -1.8, "radius": 80.0, "intensity": 1.0, "seq": q}
    st.on_news({"type": "storm_formed", "round": 1, "storm": "X", "time": "2025-08-04T10:00:00", "seq": 1, "headline": "formed"})
    st.on_news(obs(0, 2)); wait(1.0, lambda: (refresh(), st.tick()))
    check("11b sleeve holds through formation", fx.pos == tgt and st.sleeve["state"] == "hold", f"pos {fx.pos}, state {st.sleeve['state']}")
    st.on_news(obs(12, 3)); wait(2.5, lambda: (refresh(), st.tick()))
    check("11c sleeve exits at storm age 10-28 within 0.5% of mid", fx.pos == 0 and all(abs(t["price"] / 40 - 1) <= 0.005 for t in fx.trades),
          f"pos {fx.pos}, trades {[(t['side'], t['quantity'], t['price']) for t in fx.trades]}")
    st.on_news(obs(30, 4)); wait(0.5, lambda: (refresh(), st.tick()))
    check("11d sleeve done at age 28+", st.sleeve["state"] == "done", st.sleeve["state"])
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 1, "storm": "X", "time": "2025-08-04T09:40:00", "damage": []})
    wait(1.5, lambda: (refresh(), st.tick()))
    check("12 sleeve off (the Strategy default; live.py sets it on): no sleeve, no orders", st.sleeve is None and fx.pos == 0 and not fx.trades)
    # round 1 of a testnet season: the clock sits at 09:30:00 for ~120 s, so a 09:30 snapshot must not start the timer
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.form_frac = 0.2; S.FORM_ENTRY = FE
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 1, "storm": "X", "time": "2025-08-04T09:30:00", "damage": []})
    wait(1.5, lambda: (refresh(), st.tick()))
    check("11e a 09:30:00 snapshot (clock maybe paused) does not time the open", st.sleeve is None and fx.pos == 0 and not fx.trades,
          f"sleeve {st.sleeve}, pos {fx.pos}")
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 1, "storm": "X", "time": "2025-08-04T09:44:30", "damage": []})
    wait(3.5, lambda: (refresh(), st.tick()))
    check(f"11f the first running-clock snapshot times the open (29 s) and the sleeve enters from {FE[0]:.0f} s",
          st.round_open is not None and st.sleeve["state"] == "entry" and fx.pos == tgt, f"sleeve {st.sleeve}, pos {fx.pos}")
    # an announcement on a symbol the sleeve holds: the sleeve hands it over (paused; only the follower may trade it)
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.form_frac = 0.2
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 1, "storm": "X", "time": "2025-08-04T09:45:00", "damage": []})
    wait(2.5, lambda: (refresh(), st.tick()))
    held = fx.pos
    st.on_news({"type": "announcement", "symbols": [SYM], "headline": "test announcement", "time": "2025-08-04T09:50:00"})
    st.on_news({"type": "storm_formed", "round": 1, "storm": "X", "time": "2025-08-04T10:00:00", "seq": 1, "headline": "formed"})
    st.on_news(obs(0, 2)); st.on_news(obs(12, 3)); wait(3.5, lambda: (refresh(), st.tick()))
    sleeve_exits = [kw for e, kw in ev if e == "order" and "sleeve exit" in str(kw.get("reason"))]
    check("11g announcement: the sleeve drops the symbol; the announcement rule (not the sleeve) closes it to flat",
          held == tgt and fx.pos == 0 and SYM not in st.sleeve["targets"] and SYM in st.paused and not sleeve_exits,
          f"held {held}, pos {fx.pos}, sleeve exit orders {len(sleeve_exits)}")
    # a manual pause ({"pause": [...]} in control.json) is respected by the sleeve's exit too
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.form_frac = 0.2
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 1, "storm": "X", "time": "2025-08-04T09:45:00", "damage": []})
    wait(2.5, lambda: (refresh(), st.tick()))
    held = fx.pos; st.paused.add(SYM)
    st.on_news({"type": "storm_formed", "round": 1, "storm": "X", "time": "2025-08-04T10:00:00", "seq": 1, "headline": "formed"})
    st.on_news(obs(0, 2)); st.on_news(obs(12, 3)); wait(3.5, lambda: (refresh(), st.tick()))
    check("11h manual pause: the sleeve's exit leaves the paused symbol alone", held == tgt and fx.pos == held, f"held {held}, pos {fx.pos}")

    # 13. announcement: get flat in the frozen stock (within 1% of mid, retried from the current mid), then hold
    ann = lambda q: {"type": "announcement", "round": 1, "storm": "X", "time": "2025-08-04T11:00:00", "headline": "test",
                     "symbols": [SYM], "seq": q}
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    st.on_news(ann(3)); wait(2.5, lambda: (refresh(), st.tick()))
    check("13a announcement: the frozen stock is closed to flat within 1% of mid",
          fx.pos == 0 and SYM in st.paused and fx.trades and all(abs(t["price"] / 40 - 1) <= 0.01 for t in fx.trades),
          f"pos {fx.pos}, trades {[(t['side'], t['quantity'], t['price']) for t in fx.trades]}")
    wait(1.5, lambda: (refresh(), st.tick()))
    n = len(fx.trades); fx.pos, fx.avg = 30, 40.0                 # then a human buys 30 by hand
    wait(3.0, lambda: (refresh(), st.tick()))
    check("13b once flat it holds: a position opened by hand afterwards is left alone",
          fx.pos == 30 and len(fx.trades) == n and SYM not in st.closing, f"pos {fx.pos}, closing {st.closing}")
    keep = S.LEFTOVER_RETRY; S.LEFTOVER_RETRY = 2.0
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    fx.mid = 41.5; st.quotes[SYM] = (39.95, 40.05)               # the bot still sees the old price: a stale 1% cap
    st.on_news(ann(4)); st.tick(); wait(0.6, refresh)
    check("13c the first close from a stale price does not fill", fx.pos == -50, f"pos {fx.pos}")
    wait(3.0, lambda: (refresh(), st.tick())); wait(1.0, refresh)
    check("13d retried from the current mid until flat", fx.pos == 0, f"pos {fx.pos}")
    S.LEFTOVER_RETRY = keep

    # 14. valuation shift for an announced company (automatic on resume, or {"fair"} by hand); ETFs follow
    import numpy as np
    import engine as E
    z = np.zeros(E.N_SITES)
    sea = E.Season(); v0 = E.value_with(sea, st.consts, z, z, 3)
    sea.finals = {1: {c: 10.0 for c in E.PHYS}}; x0 = E.exact_values(sea, None)
    sea.value_offset = {"GEN": -3.0}
    v1 = E.value_with(sea, st.consts, z, z, 3); x1 = E.exact_values(sea, None)
    same = lambda a_, b_: all(abs(a_[k] - b_[k]) < 1e-9 for k in a_ if k not in ("GEN", "REBLD"))
    check("14a GEN shifted -$3: GEN and REBLD (1 GEN per share) -3 in both the model's values and the settlement "
          "values, every other symbol unchanged", abs(v1["GEN"] - v0["GEN"] + 3) < 1e-9 and abs(v1["REBLD"] - v0["REBLD"] + 3) < 1e-9
          and abs(x1["GEN"] - x0["GEN"] + 3) < 1e-9 and abs(x1["REBLD"] - x0["REBLD"] + 3) < 1e-9 and same(v0, v1) and same(x0, x1))
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.set_fair(SYM, -1.0)
    check("14b {\"fair\": ...} by hand: the season's valuation of the stock moves by it", st.season.value_offset == {SYM: -1.0},
          f"offsets {st.season.value_offset}")
    st.set_fair(SYM, 0); cleared = not st.season.value_offset
    st.set_fair(SYM, -2.0); st.new_season("test")
    check("14c 0 clears it, and a new season starts with none", cleared and not st.season.value_offset, f"offsets {st.season.value_offset}")

    # 15. resume after an announcement recalibrates: our value's gap to the price goes back to its level before the news
    gap = lambda st_, k: E.value_with(st_.season, st_.consts, z, z, k)[SYM]          # our formula between storms
    all_quotes = lambda st_: [st_.on_quote(x, 49.95, 50.05) for x in S.SYMS if x != SYM] + [st_.on_quote(SYM, 39.95, 40.05)]
    formed = {"type": "storm_formed", "round": 1, "storm": "X", "time": "2025-08-04T10:00:00", "seq": 1, "headline": "formed"}
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)                 # 15a: between storms -> at once
    st.on_news(ann(6)); fx.mid = 39.0; refresh()                         # the house reprices GRID by -$1
    st.resume([SYM, "ARCH"])
    check("15a between storms: resume recalibrates at once (-$1, the house's move) and unfreezes the stock and its ETF",
          abs(st.season.value_offset.get(SYM, 0) + 1.0) < 1e-6 and SYM not in st.paused and "ARCH" not in st.paused,
          f"offsets {st.season.value_offset}, paused {sorted(st.paused)}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)                 # 15b-c: during a storm -> next round start
    all_quotes(st); st.on_news(formed); st.on_news(obs(0, 2))
    v0 = gap(st, 3)
    st.on_news(ann(6)); fx.mid = 39.0; refresh(); st.on_quote(SYM, 38.95, 39.05)
    st.resume([SYM, "ARCH"])
    check("15b during a storm: the resume is scheduled; the stock and its ETF stay frozen",
          SYM in st.paused and "ARCH" in st.paused and st.resume_pending == {SYM, "ARCH"}, f"pending {st.resume_pending}")
    st.storm.dissipated = True; st.season.finals[1] = {c: 0.0 for c in E.PHYS}    # the storm ended, final report in
    v1 = gap(st, 2)
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 2, "storm": "Y", "time": "2025-08-04T09:45:00", "damage": []})
    want = (v0 - 40.0) - (v1 - 39.0)                                     # = house move - formula move
    check("15c at the next round start it recalibrates (house move minus our formula's move) and resumes both",
          abs(st.season.value_offset.get(SYM, 0) - want) < 1e-6 and SYM not in st.paused and "ARCH" not in st.paused
          and not st.resume_pending, f"offset {st.season.value_offset.get(SYM)}, want {want:.4f}, paused {sorted(st.paused)}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)                 # 15d: a manual fair wins, resume at once
    all_quotes(st); st.on_news(formed); st.on_news(obs(0, 2)); st.on_news(ann(6))
    st.set_fair(SYM, -2.0); st.resume([SYM])
    check("15d with a manual {\"fair\"}, resume is immediate even mid-storm and keeps that number",
          SYM not in st.paused and st.season.value_offset == {SYM: -2.0}, f"paused {sorted(st.paused)}, offsets {st.season.value_offset}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)                 # 15e: no pre-news measurement (a restart)
    st.on_news(formed); st.on_news(obs(0, 2)); st.on_news(ann(6))       # no quotes for the other stocks: no round gaps
    st.storm.dissipated = True; st.season.finals[1] = {c: 0.0 for c in E.PHYS}
    st.resume([SYM])
    check("15e no measurement from before the news (e.g. a restart): our value is set to the market price",
          abs(gap(st, 2) - st.top_mid(SYM)) < 1e-6 and SYM in st.season.value_offset and SYM not in st.paused,   # value incl. the shift
          f"offset {st.season.value_offset}")

    # 16. the commands as typed into bot/control.json on the day (live.apply_control: the bot's own handler)
    import live as LV
    clog = []
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    LV.apply_control(st, {"pause": "GRID"}, lambda e, **kw: clog.append(e))
    check("16a {\"pause\": \"GRID\"} (a single name, not a list) pauses GRID", "GRID" in st.paused, f"paused {sorted(st.paused)}")
    LV.apply_control(st, {"resume": "GRID"}, lambda e, **kw: clog.append(e))
    check("16b {\"resume\": \"GRID\"} resumes it", "GRID" not in st.paused, f"paused {sorted(st.paused)}")
    st.on_news(ann(7)); fx.mid = 39.0; refresh()
    LV.apply_control(st, {"resume": ["GRID", "ARCH"]}, lambda e, **kw: clog.append(e))
    check("16c after an announcement, {\"resume\": [\"GRID\", \"ARCH\"]} between storms recalibrates (-$1) and resumes both",
          abs(st.season.value_offset.get("GRID", 0) + 1.0) < 1e-6 and not ({"GRID", "ARCH"} & st.paused), f"offsets {st.season.value_offset}")
    try:
        LV.apply_control(st, {"fair": {"GRID": "minus three"}}, lambda e, **kw: clog.append(e)); raised = False
    except ValueError:
        raised = True
    check("16d a malformed command raises (the control loop catches it, logs 'control command error' and keeps going)", raised)
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    all_quotes(st); st.on_news(formed); st.on_news(obs(0, 2)); st.on_news(ann(8))
    LV.apply_control(st, {"fair": {"GRID": -3}, "resume": ["GRID"]}, lambda e, **kw: clog.append(e))
    check("16e {\"fair\": {\"GRID\": -3}, \"resume\": [\"GRID\"]} mid-storm: fair first, then an immediate resume",
          "GRID" not in st.paused and st.season.value_offset == {"GRID": -3.0}, f"paused {sorted(st.paused)}, offsets {st.season.value_offset}")
    LV.apply_control(st, {"halt": True}, lambda e, **kw: clog.append(e)); h1 = st.halted
    LV.apply_control(st, {"halt": False}, lambda e, **kw: clog.append(e))
    check("16f the brake: {\"halt\": true} / {\"halt\": false}", h1 and not st.halted and clog.count("CONTROL") == 6, f"log {clog}")

    # 17. a second announcement on the same stock after the first was resumed: a fresh cycle, the shifts add up
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.on_news(ann(9)); fx.mid = 39.0; refresh(); LV.apply_control(st, {"resume": ["GRID", "ARCH"]}, lambda e, **kw: None)
    first = st.season.value_offset.get("GRID", 0)
    st.on_news(ann(10)); refaced = "GRID" in st.paused and "ARCH" in st.paused
    fx.mid = 37.0; refresh(); LV.apply_control(st, {"resume": ["GRID", "ARCH"]}, lambda e, **kw: None)
    check("17 a second announcement later: frozen again, then resume adds its -$2 to the first -$1 (total -$3)",
          abs(first + 1.0) < 1e-6 and refaced and abs(st.season.value_offset.get("GRID", 0) + 3.0) < 1e-6
          and not ({"GRID", "ARCH"} & st.paused), f"first {first}, total {st.season.value_offset}")

    # 18. a later announcement closes only what IT froze; a missed announcement in a snapshot; the "announce" command
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.on_news(ann(11)); wait(1.5, lambda: (refresh(), st.tick()))        # GRID frozen (flat)
    fx.pos, fx.avg = 30, 40.0; wait(1.2, refresh)                          # a human now trades GRID by hand
    st.on_news({"type": "announcement", "round": 1, "storm": "X", "time": "2025-08-04T11:05:00", "headline": "PORT news",
                "symbols": ["PORT"], "seq": 12})
    wait(2.5, lambda: (refresh(), st.tick()))
    check("18a a second announcement (PORT) does not close the GRID position a human is trading from the first",
          fx.pos == 30 and "PORT" in st.paused and "GRID" in st.paused, f"pos {fx.pos}, paused {sorted(st.paused)}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    snap = lambda hl: {"type": "snapshot", "phase": "storm", "round": 1, "storm": "X", "time": "2025-08-04T11:10:00",
                       "damage": [], "headlines": hl}
    st.on_news(snap([{"type": "announcement", "round": 1, "time": "2025-08-04T11:06:00", "headline": "GRID news",
                      "symbols": ["GRID"], "seq": 20}]))
    wait(2.5, lambda: (refresh(), st.tick()))
    check("18b an announcement we missed, seen in a snapshot with its symbols: handled as usual (frozen, flat)",
          SYM in st.paused and fx.pos == 0, f"pos {fx.pos}, paused {sorted(st.paused)}")
    st.on_news(snap([{"type": "announcement", "round": 1, "time": "2025-08-04T11:06:00", "headline": "GRID news",
                      "symbols": ["GRID"], "seq": 20}]))
    n_ann = sum(1 for e, _ in ev if e == "ANNOUNCEMENT - PAUSED")
    check("18c the same headline in the next snapshot is not handled twice", n_ann == 1, f"handled {n_ann} times")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.on_news(snap([{"type": "announcement", "round": 1, "time": "2025-08-04T11:06:00", "headline": "A government action",
                      "seq": 21}]))
    missed = [kw for e, kw in ev if e == "ANNOUNCEMENT MISSED - SEEN ONLY IN A SNAPSHOT"]
    check("18d a missed announcement without symbols: nothing frozen, the headline is reported (ALERT)",
          not st.paused and len(missed) == 1 and missed[0]["headline"] == "A government action", f"paused {sorted(st.paused)}")
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    LV.apply_control(st, {"announce": "GRID"}, lambda e, **kw: None); wait(2.5, lambda: (refresh(), st.tick()))
    check("18e {\"announce\": \"GRID\"} runs the usual handling: frozen with its ETF, flat, gap remembered",
          SYM in st.paused and "ARCH" in st.paused and fx.pos == 0 and SYM in st.gap_before, f"pos {fx.pos}, paused {sorted(st.paused)}")
    try:
        LV.apply_control(st, {"announce": ["GRDI"]}, lambda e, **kw: None); typo_raised = False
    except ValueError:
        typo_raised = True
    check("18f a typo in announce is rejected (it must not freeze everything)", typo_raised and st.paused == {SYM, "ARCH"},
          f"paused {sorted(st.paused)}")

    # 19. full review fixes: restart fallback, brake, late manual announce, resume all, the ctl helper, monitor
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)                   # 19a restart: the announcement is replayed
    st.replaying = True; st.on_news(ann(13)); st.replaying = False
    marked = st.gap_before.get(SYM) == (None, st.season_no) and SYM in st.paused
    fx.mid = 39.0; refresh(); LV.apply_control(st, {"resume": [SYM, "ARCH"]}, lambda e, **kw: None)
    check("19a after a restart (announcement replayed): resume still recalibrates, to the market price",
          marked and abs(gap(st, 3) - st.top_mid(SYM)) < 1e-6 and not ({SYM, "ARCH"} & st.paused), f"offsets {st.season.value_offset}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)                   # 19b the brake is on at the announcement
    fx.pos, fx.avg = -50, 40.0; wait(1.2, refresh)
    st.halted = True; st.on_news(ann(14)); wait(2.5, lambda: (refresh(), st.tick()))
    held = fx.pos
    st.halted = False; wait(2.5, lambda: (refresh(), st.tick()))
    check("19b with the brake on, nothing is closed; once released, the announced stock is closed to flat",
          held == -50 and fx.pos == 0, f"while halted {held}, after {fx.pos}")
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)                   # 19c a late manual announce, between storms
    all_quotes(st); st.on_news(formed); st.on_news(obs(0, 2)); start_gap = st.round_gaps.get(SYM)
    st.storm.dissipated = True; st.season.finals[1] = {c: 0.0 for c in E.PHYS}
    fx.mid = 38.0; refresh(); st.on_quote(SYM, 37.95, 38.05)              # the house already moved: we missed it
    LV.apply_control(st, {"announce": [SYM]}, lambda e, **kw: None)
    check("19c a manual announce (late) remembers the round-start gap, from before the news, not the gap now",
          start_gap is not None and st.gap_before.get(SYM, (None,))[0] == start_gap, f"{st.gap_before.get(SYM)} vs {start_gap}")
    LV.apply_control(st, {"resume": "all"}, lambda e, **kw: None)
    check("19d {\"resume\": \"all\"} resumes everything frozen", not st.paused, f"paused {sorted(st.paused)}")
    import ctl
    built = [ctl.build(a) for a in (["resume", "gen", "rebld"], ["resume", "ALL"], ["announce", "GEN"], ["fair", "GEN", "-3"],
                                     ["halt"], ["go"])]
    bad = 0
    for a in (["resume"], ["fair", "GEN"], ["bogus"]):
        try:
            ctl.build(a)
        except ValueError:
            bad += 1
    import tempfile, threading as _th
    tmpd = Path(tempfile.mkdtemp()); cpath = tmpd / "control.json"
    taken = []
    def bot_reads():                                                      # plays the bot: read and delete once
        for _ in range(100):
            if cpath.exists():
                taken.append(json.loads(cpath.read_text())); cpath.unlink(); return
            time.sleep(0.05)
    _th.Thread(target=bot_reads).start(); sent_ok = ctl.send({"resume": ["GEN"]}, cpath, 2, 5)
    check("19e ctl.py: builds the commands, rejects bad ones, and writes atomically for the bot to pick up",
          built == [{"resume": ["GEN", "REBLD"]}, {"resume": "all"}, {"announce": ["GEN"]}, {"fair": {"GEN": -3.0}},
                    {"halt": True}, {"halt": False}] and bad == 3 and sent_ok and taken == [{"resume": ["GEN"]}], f"{built} {bad} {taken}")
    import watch as WT
    class _D: pass
    d_ = _D(); d_.events = [(10, "ANNOUNCEMENT - PAUSED", {"paused": ["GEN", "REBLD"]}), (20, "RESUMED", {"symbol": "REBLD"}),
                            (30, "CONTROL", {"paused": ["GEN", "PORT"]}), (40, "RESUMED", {"symbol": "GEN"})]
    check("19f monitor: knows which stocks are frozen (so manual trades in them are not blamed on the bot)",
          WT.frozen_at(d_, 15) == {"GEN", "REBLD"} and WT.frozen_at(d_, 25) == {"GEN"} and WT.frozen_at(d_, 35) == {"GEN", "PORT"}
          and WT.frozen_at(d_, 45) == {"PORT"})

    # 20. reading bot/control.json: a malformed file is consumed and reported; the next command goes through
    tmpd = Path(tempfile.mkdtemp()); cf = tmpd / "control.json"; elog = []
    cf.write_text('{"resume": ["GEN"'); r1 = LV.read_control(cf, lambda e, **kw: elog.append(e)); gone1 = not cf.exists()
    cf.write_text('["halt"]'); r2 = LV.read_control(cf, lambda e, **kw: elog.append(e)); gone2 = not cf.exists()
    cf.write_text('{"halt": true}'); r3 = LV.read_control(cf, lambda e, **kw: elog.append(e))
    r4 = LV.read_control(cf, lambda e, **kw: elog.append(e))
    check("20 control file: broken JSON and a non-object are consumed and reported, the next command is read, no file = none",
          r1 is None and gone1 and r2 is None and gone2 and r3 == {"halt": True} and r4 is None
          and elog == ["control file error", "control file error"], f"{r1} {r2} {r3} {r4} {elog}")

    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
