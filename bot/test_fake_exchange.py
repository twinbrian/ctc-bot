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

    # 4. announcement follower: house moves +2% in the first seconds -> follow (half limit), then exit
    S.FOLLOW_WAIT, S.FOLLOW_HOLD = 1.0, 3.0
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
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
    S.FOLLOW_WAIT = 1.0
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
    check("11g announcement: the sleeve drops the symbol and does not trade it any more (paused, follower's call)",
          held == tgt and fx.pos == held and SYM not in st.sleeve["targets"] and SYM in st.paused, f"held {held}, pos {fx.pos}")
    # a manual pause ({"pause": [...]} in control.json) is respected by the sleeve's exit too
    fx = FakeExchange(); st, b, ev, refresh = setup(fx)
    st.form_frac = 0.2
    st.on_news({"type": "snapshot", "phase": "pre_storm", "round": 1, "storm": "X", "time": "2025-08-04T09:45:00", "damage": []})
    wait(2.5, lambda: (refresh(), st.tick()))
    held = fx.pos; st.paused.add(SYM)
    st.on_news({"type": "storm_formed", "round": 1, "storm": "X", "time": "2025-08-04T10:00:00", "seq": 1, "headline": "formed"})
    st.on_news(obs(0, 2)); st.on_news(obs(12, 3)); wait(3.5, lambda: (refresh(), st.tick()))
    check("11h manual pause: the sleeve's exit leaves the paused symbol alone", held == tgt and fx.pos == held, f"held {held}, pos {fx.pos}")

    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
