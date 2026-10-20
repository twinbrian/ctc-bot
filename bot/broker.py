"""Brokers turn goals (target position + price cap per symbol) into orders.

The testnet book is thin (often 20-60 shares at the touch, ~50-500 within 1%), so a goal is worked in
pieces: every RETRY_GAP seconds, send a limit order at the cap for the quantity resting at or better than
the cap; anything left resting after CANCEL_AFTER seconds is cancelled. Positions are re-read from
GET /positions every second.

SimBroker   same logic against a modelled book (replay and --dry-run)
LiveBroker  REST orders from a worker thread; one token bucket per API key (60 actions/min per key)
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request

HEADERS = {"User-Agent": "hurricane-bot/1.0", "Content-Type": "application/json"}
CANCEL_AFTER = 2.0
RETRY_GAP = 3.0
ACTIONS_PER_MIN = 55
TICK = 0.01                                  # the exchange rejects prices that are not multiples of 0.01
REJECT_BACKOFF = 10.0
DEAD_FEED = 10.0                             # no websocket frame for this long = connection dead: don't trade
STUB_SPARE_TOKENS = 4                        # stubs only use order-rate budget left over by real trading


def to_tick(price: float, buy: bool) -> float:
    """Round a price cap to the tick in the safe direction (buy caps down, sell caps up)."""
    n = price / TICK
    return round((math.floor(n + 1e-9) if buy else math.ceil(n - 1e-9)) * TICK, 2)


def fillable(levels, side_buy: bool, cap: float) -> int:
    """Shares resting at or better than `cap` on the side we would trade against."""
    tot = 0
    for lv in levels:
        p, q = lv["price"], lv["quantity"]
        if (side_buy and p <= cap + 1e-9) or (not side_buy and p >= cap - 1e-9):
            tot += q
    return int(tot)


class Goals:
    """Shared goal book: symbol -> (target, cap, priority, reason, band). A goal is (target, cap, priority) or
    (target, cap, priority, band): with a band, no order is priced further than band x the CURRENT mid from mid,
    whatever the cap (the formation sleeve uses it: take only what is offered near mid, come back next refresh)."""

    def __init__(self, limits):
        self.limits, self.g, self.lock = limits, {}, threading.Lock()

    def set(self, goals: dict, reason: str):
        with self.lock:
            for s, g in goals.items():
                tgt, cap, prio = g[:3]
                band = float(g[3]) if len(g) > 3 and g[3] is not None else None
                tgt = max(-self.limits[s], min(self.limits[s], int(tgt)))
                self.g[s] = (tgt, None if cap is None else float(cap), float(prio), reason, band)

    def clear(self):
        with self.lock:
            self.g = {}

    def items(self):
        with self.lock:
            return sorted(self.g.items(), key=lambda kv: -kv[1][2])


class SimBroker:
    """Fills against a modelled book: `profile[s]` = [(offset as fraction of mid, qty), ...] per side,
    replenished every `refresh` seconds. With profile=None it fills everything at the touch."""

    def __init__(self, quotes: dict, log, limits: dict, clock, profile=None, refresh=4.0, books=None, extra=None):
        self.quotes, self.log, self.limits, self.clock, self.books = quotes, log, limits, clock, books
        self.extra = extra or {}                 # replay only: extra cost per share (fraction of price), calibrated to live
        self.profile, self.refresh = profile, refresh
        self.pos, self.cash, self.fills = {}, 0.0, []
        self.goals = Goals(limits)
        self.last_try, self.used = {}, {}           # symbol -> time; (symbol, side) -> (refresh epoch, qty used)

    def position(self, s):
        return self.pos.get(s, 0)

    def clear(self):
        self.goals.clear()

    def equity(self):
        return self.cash + sum(q * sum(self.quotes.get(s, (0, 0))) / 2 for s, q in self.pos.items())

    def sync_stubs(self, desired: dict):
        pass                                               # resting orders are not simulated

    def set_targets(self, goals: dict, reason: str = "", urgent: bool = False):
        self.goals.set(goals, reason)
        self.tick(force=urgent)

    def _levels(self, s, buy):
        bid, ask = self.quotes[s]
        mid = (bid + ask) / 2
        if self.books is not None and self.books.get(s):              # dry-run: fill against the real book
            return self.books[s][1] if buy else self.books[s][0]
        if self.profile is None:
            return [{"price": ask if buy else bid, "quantity": 10 ** 9}]
        e = self.extra.get(s, 0.0)
        return [{"price": mid * (1 + o + e) if buy else mid * (1 - o - e), "quantity": q} for o, q in self.profile[s]]

    def tick(self, force=False):
        now = self.clock()
        for s, (tgt, cap, _, reason, band) in self.goals.items():
            if cap is None:
                continue                                   # HOLD: do not trade
            d = tgt - self.pos.get(s, 0)
            if d == 0 or s not in self.quotes or (not force and now - self.last_try.get(s, -1e9) < RETRY_GAP):
                continue
            self.last_try[s] = now
            buy = d > 0
            if band is not None:
                mid = (self.quotes[s][0] + self.quotes[s][1]) / 2
                cap = min(cap, mid * (1 + band)) if buy else max(cap, mid * (1 - band))
            epoch = int(now // self.refresh)
            ep, taken = self.used.get((s, buy), (epoch, 0))
            taken = taken if ep == epoch else 0          # liquidity we already took since the last refresh
            skip, want, got, cost = taken, abs(d), 0, 0.0
            for lv in sorted(self._levels(s, buy), key=lambda l: l["price"] if buy else -l["price"]):
                if (buy and lv["price"] > cap) or (not buy and lv["price"] < cap):
                    break
                avail = lv["quantity"] - min(skip, lv["quantity"])
                skip -= min(skip, lv["quantity"])
                take = min(avail, want - got)
                got += take; cost += take * lv["price"]
                if got >= want:
                    break
            if got == 0:
                continue
            self.used[(s, buy)] = (epoch, taken + got)
            q = got if buy else -got
            self.pos[s] = self.pos.get(s, 0) + q
            self.cash -= cost if buy else -cost
            self.fills.append((s, q, cost / got, reason, now, (self.quotes[s][0] + self.quotes[s][1]) / 2))
            self.log("fill", symbol=s, qty=q, price=round(cost / got, 4), reason=reason, sim=True)


class RateLimiter:
    def __init__(self, per_min=ACTIONS_PER_MIN, burst=10):
        self.rate, self.cap, self.tokens, self.t = per_min / 60.0, burst, burst, time.monotonic()
        self.lock = threading.Lock()

    def take(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.cap, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class Rest:
    def __init__(self, base: str, key: str):
        self.base, self.key = base.rstrip("/"), key

    def call(self, method: str, path: str, body=None, api=True):
        url = self.base + ("/api/v1" if api else "") + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={**HEADERS, "X-API-Key": self.key})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                txt = r.read().decode()
                return r.status, (json.loads(txt) if txt else None)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()[:500]
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return 0, repr(e)


def parse_positions(js) -> dict:
    """GET /positions -> {symbol: quantity}. Accepts a list of {symbol, quantity|position|...} or a dict."""
    items = js.get("positions", js) if isinstance(js, dict) else js
    out = {}
    keys = ("quantity", "position", "qty", "net_position", "size", "net_quantity")
    if isinstance(items, dict):
        for s, v in items.items():
            q = v if isinstance(v, (int, float)) else next((v[k] for k in keys if k in v), 0)
            out[s] = int(q)
    elif isinstance(items, list):
        for v in items:
            s = v.get("symbol") or v.get("ticker")
            if s:
                out[s] = int(next((v[k] for k in keys if k in v), 0))
    return out


class LiveBroker:
    def __init__(self, rests: list, quotes: dict, books: dict, log, limits: dict):
        self.rests, self.quotes, self.books, self.log, self.limits = rests, quotes, books, log, limits
        self.buckets = [RateLimiter() for _ in rests]
        self.lock = threading.Lock()
        self.fill_ids: set = set()
        self.fill_cash = 0.0                    # exact: from the exchange's fills
        self.fills_ok = False                   # until GET /trades has answered, fall back to the estimated ledger
        self.seen_positions = False             # first positions read = opening cost basis (average price)
        self.pos: dict = {}
        self.avg: dict = {}
        self.cash = 0.0
        self.goals = Goals(limits)
        self.working: dict = {}                 # symbol -> (time sent, key index, order_id)
        self.last_frame = time.monotonic()       # last websocket frame of any kind (set by live.py)
        self.stubs: dict = {}                    # (symbol, side) -> (order_id, price, qty, key index)
        self.stub_want: dict = {}                # (symbol, side) -> (price, qty) wanted by the strategy
        self.stub_lock = threading.Lock()
        self.on_stub_fill = None                 # callback set by the strategy
        self.last_try: dict = {}
        self.positions_raw = None
        self.logged_raw = False
        self.last_budget_log = 0.0
        self.wake = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._poller, daemon=True).start()

    def position(self, s):
        return self.pos.get(s, 0)

    def clear(self):
        self.goals.clear()

    def set_targets(self, goals: dict, reason: str = "", urgent: bool = False):
        self.goals.set(goals, reason)
        if urgent:
            for s in goals:
                self.last_try.pop(s, None)
        self.wake.set()

    def _token(self):
        for i, b in enumerate(self.buckets):
            if b.take():
                return i
        return None

    def _loop(self):
        while True:
            self.wake.wait(0.5); self.wake.clear()
            try:
                self._step()
            except Exception as exc:                       # a worker thread must never die silently
                self.log("error", where="broker loop", error=repr(exc))
                time.sleep(1.0)

    def _step(self):
        now = time.monotonic()
        if now - self.last_frame > DEAD_FEED:
            return                                         # feed is dead: our view of the book is stale
        for s, (tgt, cap, _, reason, band) in self.goals.items():
            if s in self.working:
                t0, ki, oid = self.working[s]
                if now - t0 < CANCEL_AFTER:
                    continue
                st, js = self.rests[ki].call("GET", "/orders/open")
                left = [o for o in (js or {}).get("orders", []) if o.get("order_id") == oid] if st == 200 and oid else []
                if left:
                    k = self._token()
                    if k is None:
                        continue
                    self.rests[k].call("DELETE", f"/orders/{oid}")
                    self.log("cancel remainder", symbol=s, remaining=sum(o["quantity"] - o.get("filled_quantity", 0) for o in left))
                self.working.pop(s)
                self._refresh_positions()
            if cap is None:
                continue                                   # HOLD: never trade toward a stale target
            d = tgt - self.pos.get(s, 0)
            if d == 0 or s not in self.quotes or now - self.last_try.get(s, -1e9) < RETRY_GAP:
                continue
            buy = d > 0
            if band is not None:                           # never price further than band x mid from the current mid
                mid = (self.quotes[s][0] + self.quotes[s][1]) / 2
                cap = min(cap, mid * (1 + band)) if buy else max(cap, mid * (1 - band))
            price = to_tick(cap, buy)
            bids, asks = self.books.get(s, ([], []))
            q = min(abs(d), fillable(asks if buy else bids, buy, price)) if (bids or asks) else 0
            if q <= 0:
                continue
            k = self._token()
            if k is None:
                if now - self.last_budget_log > 10:
                    self.log("rate budget exhausted (lower-priority symbols wait)", symbol=s)
                    self.last_budget_log = now
                return
            order = {"symbol": s, "side": "buy" if buy else "sell", "order_type": "limit",
                     "quantity": int(q), "price": price}
            st, js = self.rests[k].call("POST", "/orders", order)
            if st != 200:
                self.last_try[s] = now + REJECT_BACKOFF - RETRY_GAP
                self.log("ORDER REJECTED", order=order, status=st, response=js, reason=reason)
                continue
            self.last_try[s] = now
            self.working[s] = (now, k, (js or {}).get("order_id") if isinstance(js, dict) else None)
            self.log("order", order=order, status=st, response=js, reason=reason, key=k)
        self._step_stubs(now)

    # ---- stub orders: wide resting quotes that only fill if someone trades far through fair value ----------
    def sync_stubs(self, desired: dict):
        """desired: {(symbol, side): (price, qty)}; anything not listed is cancelled."""
        with self.stub_lock:
            self.stub_want = dict(desired)
        self.wake.set()

    def _spare_token(self):
        for i, b in enumerate(self.buckets):
            with b.lock:
                b.tokens = min(b.cap, b.tokens + (time.monotonic() - b.t) * b.rate); b.t = time.monotonic()
                if b.tokens >= STUB_SPARE_TOKENS:
                    b.tokens -= 1
                    return i
        return None

    def _step_stubs(self, now):
        with self.stub_lock:
            want = dict(self.stub_want)
        live = {}
        if self.stubs:
            st, js = self.rests[0].call("GET", "/orders/open")
            live = {o["order_id"]: o for o in (js or {}).get("orders", [])} if st == 200 else None
        for key, (oid, px, qty, ki) in list(self.stubs.items()):
            gone = live is not None and oid not in live
            w = want.get(key)
            stale = w is None or abs(w[0] - px) / px > 0.01
            if gone:                                       # filled (or cancelled by the exchange for limit room)
                self.stubs.pop(key)
                self.log("STUB GONE (filled or cancelled)", symbol=key[0], side=key[1], price=px, qty=qty)
            elif stale:
                if not self.buckets[ki].take():               # cancel with the key that placed it
                    continue
                self.rests[ki].call("DELETE", f"/orders/{oid}")
                self.stubs.pop(key)
        for key, (px, qty) in want.items():
            if key in self.stubs or qty <= 0:
                continue
            k = self._spare_token()
            if k is None:
                return
            sym, side = key
            order = {"symbol": sym, "side": side, "order_type": "limit", "quantity": int(qty), "price": to_tick(px, side == "buy")}
            st, js = self.rests[k].call("POST", "/orders", order)
            if st == 200 and isinstance(js, dict) and js.get("order_id"):
                self.stubs[key] = (js["order_id"], order["price"], int(qty), k)
                self.log("stub placed", order=order)
            else:
                self.log("stub rejected", order=order, status=st, response=js)

    def _poller(self):
        n = 0
        while True:
            try:
                self._refresh_positions()
                if n % 2 == 0:
                    self._poll_fills()
            except Exception as exc:
                self.log("error", where="positions poller", error=repr(exc))
            n += 1
            time.sleep(1.0)

    def _poll_fills(self):
        """GET /trades returns recent fills (exact prices); poll often and accumulate by trade_id."""
        st, js = self.rests[0].call("GET", "/trades")
        if st != 200 or not isinstance(js, dict):
            return
        first = not self.fills_ok
        self.fills_ok = True
        for t in js.get("trades", []):
            if t.get("kind", "trade") != "trade" or t["trade_id"] in self.fill_ids:
                continue
            self.fill_ids.add(t["trade_id"])
            if first:
                continue                     # baseline: fills from before this process are in the opening cost basis
            sg = 1 if t["side"] == "buy" else -1
            with self.lock:
                self.fill_cash -= sg * t["quantity"] * t["price"]
            self.log("fill", symbol=t["symbol"], qty=sg * t["quantity"], price=t["price"], at=t["executed_at"])
            stub = self.stubs.get((t["symbol"], t["side"]))
            if stub and abs(stub[1] - t["price"]) < 0.005 and self.on_stub_fill:
                self.on_stub_fill(t["symbol"], t["side"], t["quantity"], t["price"])

    def _refresh_positions(self):
        st, js = self.rests[0].call("GET", "/positions")
        if st != 200:
            self.log("positions error", status=st, body=js); return
        self.positions_raw = js
        new = {v["symbol"]: (int(v.get("quantity", 0)), float(v.get("average_price") or 0))
               for v in (js or {}).get("positions", []) if v.get("symbol")}
        with self.lock:
            for s in set(new) | {k for k, q in self.pos.items() if q}:
                oq, oavg = self.pos.get(s, 0), self.avg.get(s, 0.0)
                nq, navg = new.get(s, (0, 0.0))
                dq = nq - oq
                if dq:
                    if not self.seen_positions:                               # positions held before this process started
                        self.fill_cash -= navg * nq
                    if oq == 0 or (nq * oq > 0 and abs(nq) > abs(oq)):        # opened or added: exact cost
                        cost = navg * nq - oavg * oq
                    else:                                                     # reduced/closed/flipped: touch
                        bid, ask = self.quotes.get(s, (navg or oavg, navg or oavg))
                        cost = dq * (ask if dq > 0 else bid)
                    self.cash -= cost
                    self.log("position change", symbol=s, frm=oq, to=nq, cash_flow=round(-cost, 2))
                self.pos[s], self.avg[s] = nq, navg
            self.seen_positions = True

    def reset_ledger(self):
        """New competition (testnet season): the exchange zeroed positions without trades."""
        with self.lock:
            self.cash, self.pos, self.avg, self.fill_cash = 0.0, {}, {}, 0.0
        self.goals.clear()
        self.working.clear()

    def equity(self):
        return self.pnl()

    def pnl(self, marks: dict | None = None):
        """Season PnL: exact cash from the exchange's fills + positions at `marks` (default: mid)."""
        with self.lock:
            m = marks or {}
            return (self.fill_cash if self.fills_ok else self.cash) + sum(q * m.get(s, sum(self.quotes.get(s, (0, 0))) / 2) for s, q in self.pos.items())
