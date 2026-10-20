"""Run the bot against the exchange.

    export HURRICANE_BOT_KEY=...                      # from your captain's Team page
    PY=/opt/anaconda3/bin/python3
    $PY -I bot/live.py --smoke-test                   # check the API: 1-share order far from the market, then cancel
    $PY -I bot/live.py --dry-run                      # real feed and quotes, simulated fills, no orders sent
    $PY -I bot/live.py                                # live trading

Logs go to bot/logs/: run-*.jsonl (decisions, orders, fills, positions) and feed-*.jsonl (raw news frames,
in the record_feed.py format, so `python starter_kit/check_recording.py bot/logs/feed-*.jsonl history/` works).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot"))

import websockets

from broker import LiveBroker, Rest, SimBroker
from strategy import FORM_SIZE, Strategy

DEFAULT_URL = "https://cornelltradingcompetition.org"


def make_logger(path: Path, echo: set):
    f = path.open("a")

    def log(event, **kw):
        rec = {"ts": time.time(), "event": event, **kw}
        f.write(json.dumps(rec, default=str) + "\n"); f.flush()
        if event in echo or event.isupper():
            print(time.strftime("%H:%M:%S"), event, {k: v for k, v in kw.items() if k not in ("preds",)}, flush=True)
    return log


def catch_up(strat, logs: Path, log):
    """Rebuild state (storms, outages, season damage) from the bot's own recorded feed, without trading."""
    msgs = []
    for f in sorted(logs.glob("feed-*.jsonl")):
        for line in f.open():
            try:
                fr = json.loads(line)
            except ValueError:
                continue
            if isinstance(fr.get("message"), dict):
                msgs.append((fr.get("received_at", 0.0), fr["message"]))
    msgs.sort(key=lambda x: x[0])
    strat.replaying = True
    formed_at = None
    for ts, m in msgs:
        strat.on_news(m)
        if m.get("type") == "storm_formed":
            formed_at = ts
    strat.replaying, strat.pending = False, []
    st = strat.storm
    if st is not None and not st.dissipated and formed_at:
        # pre-storm prices (a model input) from the recorded order books just before the storm formed
        last = {}
        for f in sorted(logs.glob("books-*.jsonl")):
            for line in f.open():
                try:
                    b = json.loads(line)
                except ValueError:
                    continue
                if b.get("received_at", 0) <= formed_at - 2 and b.get("bids") and b.get("asks"):
                    last[b["symbol"]] = (b["bids"][0]["price"] + b["asks"][0]["price"]) / 2
        if len(last) == len(strat.limits):
            strat.Hpre = last
        log("pre-storm prices recovered" if strat.Hpre else "pre-storm prices NOT recovered", n=len(last))
    log("CAUGHT UP from recorded feed", messages=len(msgs), finals=sorted(strat.season.finals),
        approximate=strat.season.approximate, storm=(st.round, st.name, max(st.obs, default=None)) if st else None)


def smoke_test(rest: Rest):
    show = lambda label, r: print(f"\n== {label}: HTTP {r[0]}\n{json.dumps(r[1], indent=1)[:1500] if not isinstance(r[1], str) else r[1]}")
    show("competition time", rest.call("GET", "/competitions/time"))
    show("positions", rest.call("GET", "/positions"))
    ob = rest.call("GET", "/orderbook/PORT"); show("order book PORT", ob)
    bids = (ob[1] or {}).get("bids") if isinstance(ob[1], dict) else None
    ref = None
    if bids:
        b0 = bids[0]
        ref = b0.get("price") if isinstance(b0, dict) else b0[0]
    price = round(float(ref) * 0.5, 2) if ref else 1.00
    r = rest.call("POST", "/orders", {"symbol": "PORT", "side": "buy", "order_type": "limit", "quantity": 1, "price": price})
    show(f"place 1-share PORT buy at {price} (far below the market, will rest)", r)
    show("open orders", rest.call("GET", "/orders/open"))
    show("cancel all PORT orders", rest.call("DELETE", "/orders/all?symbol=PORT"))
    show("open orders after cancel", rest.call("GET", "/orders/open"))


async def run(args):
    key = args.key or os.environ.get("HURRICANE_BOT_KEY")
    if not key:
        sys.exit("set HURRICANE_BOT_KEY (or pass --key)")
    rest = Rest(args.url, key)
    if args.smoke_test:
        return smoke_test(rest)
    st, case = rest.call("GET", "/cases/hurricane.json", api=False)
    limits = case["limits"]
    print("server timing:", case["timing"])
    logs = ROOT / "bot" / "logs"; logs.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = make_logger(logs / f"run-{stamp}.jsonl", {"storm_formed", "damage_report", "final snipe", "order", "fill",
                                                    "cancel remainder", "decision", "positions error", "error"})
    feed_file = (logs / f"feed-{stamp}.jsonl").open("a")
    strat = Strategy(ROOT / "bot" / "models.pkl", limits, None, log, clock=time.monotonic)
    strat.form_frac = FORM_SIZE                             # formation sleeve (strategy.py); off: control.json {"sleeve": 0}
    log("formation sleeve", size=strat.form_frac)
    print(f"formation sleeve: {strat.form_frac:.0%} of limits")
    keys = [k for k in [key] + os.environ.get("HURRICANE_EXTRA_KEYS", "").split(",") if k]
    strat.broker = (SimBroker(strat.quotes, log, limits, clock=time.monotonic, books=strat.books) if args.dry_run
                    else LiveBroker([Rest(args.url, k) for k in keys], strat.quotes, strat.books, log, limits))
    print(f"order-action budget: {len(keys)} key(s) x {55}/min")
    catch_up(strat, logs, log)
    print("mode:", "DRY RUN (no orders sent)" if args.dry_run else "LIVE TRADING")
    ws_url = args.url.replace("https://", "wss://").replace("http://", "ws://") + "/ws/v1/market-data"
    sub = {"action": "subscribe", "symbols": list(limits), "channels": ["news", "quotes", "orderbook", "trades"]}
    book_file = (logs / f"books-{stamp}.jsonl").open("a")
    trade_file = (logs / f"trades-{stamp}.jsonl").open("a")
    first = set()

    comp = {"id": None, "season_pnl": None, "round_key": None, "round_start": 0.0}

    async def status():
        while True:
            await asyncio.sleep(10)
            b = strat.broker
            live_b = not isinstance(b, SimBroker)
            st_, t_ = await asyncio.to_thread(rest.call, "GET", "/competitions/time")
            cid = (t_ or {}).get("competition_id") if st_ == 200 else None
            # season result: exact fills + positions held into settlement at the exact settlement values
            if live_b and strat.final_values and (any(b.pos.values()) or comp["season_pnl"] is None):
                comp["season_pnl"] = round(b.pnl(marks=strat.final_values), 2)
            # round result: ledger change between storms (open positions at mid)
            key = (cid, strat.storm.round, strat.storm.name) if strat.storm else None
            if live_b and key and key != comp["round_key"]:
                now_pnl = b.pnl()
                if comp["round_key"] is not None and comp["round_key"][0] == cid:
                    log("ROUND RESULT", round=comp["round_key"][1], storm=comp["round_key"][2],
                        pnl=round(now_pnl - comp["round_start"], 2), season_to_date=round(now_pnl, 2))
                comp["round_key"], comp["round_start"] = key, now_pnl
            if cid and cid != comp["id"]:
                if comp["id"] is not None:
                    log("SEASON RESULT", competition_id=comp["id"], pnl=comp["season_pnl"],
                        note="exact fills + settlement at exact values" if comp["season_pnl"] is not None else "no final report seen")
                    log("NEW COMPETITION (testnet season): ledger reset", competition_id=cid)
                    if live_b:
                        b.reset_ledger()
                    strat.new_season("competition id changed")
                    strat.final_values, comp["season_pnl"], comp["round_key"], comp["round_start"] = None, None, None, 0.0
                comp["id"] = cid
            pos = {s: q for s, q in getattr(b, "pos", {}).items() if q}
            mids = {s: (bb + aa) / 2 for s, (bb, aa) in strat.quotes.items()}
            line = {"positions": pos}
            if isinstance(b, SimBroker):
                line["paper_pnl"] = round(b.cash + sum(q * mids.get(s, 0) for s, q in b.pos.items()), 2)
            else:
                line["bot_pnl"] = round(b.pnl(), 2)
                line["exchange_unrealized"] = round(sum(v.get("unrealized_pnl", 0) for v in (b.positions_raw or {}).get("positions", [])), 2)
            print(time.strftime("%H:%M:%S"), "STATUS", line, flush=True)
            log("status", **line)

    control = ROOT / "bot" / "control.json"

    async def control_loop():
        """Manual control on competition day without stopping the bot (its state is kept). Write bot/control.json:
           {"halt": true}                 the brake: no automatic trading at all, positions held (also no after-storm exit)
           {"halt": false}                resume (also ends a manual flatten)
           {"pause": ["GRID"]}            stop trading these symbols   {"resume": ["GRID"]} trade them again
           {"close": ["ARCH"]}            close these positions within 1% of mid
           {"flatten": true}              close every position within 1% of mid (retried from the current mid every 20 s
                                          until flat), then stay halted
           {"sniper": false} / {"follow": false} / {"stubs": true}   switch the extra strategies
           {"sleeve": 0} / {"sleeve": 0.2}  formation sleeve size (fraction of limits). 0 = no new sleeve entries from
                                          now on; a sleeve position already held is still closed by its exit if the
                                          storm has formed, otherwise it is treated like any leftover position"""
        while True:
            await asyncio.sleep(2)
            try:
                if not control.exists():
                    continue
                cmd = json.loads(control.read_text())
                control.unlink()
            except (OSError, ValueError) as exc:
                log("control file error", error=repr(exc)); continue
            if "halt" in cmd:
                strat.halted, strat.flattening = bool(cmd["halt"]), False
                if strat.halted:
                    strat.broker.set_targets({x: (strat.broker.position(x), None, 0.0) for x in strat.quotes}, reason="manual halt")
            if "sleeve" in cmd:
                strat.form_frac = max(0.0, min(1.0, float(cmd["sleeve"])))
                if strat.form_frac == 0.0 and strat.sleeve and strat.sleeve["state"] in ("wait", "entry"):
                    strat.sleeve["state"] = "done"              # stop entering now; held sleeve positions go to the model
                    strat.broker.set_targets({x: (strat.broker.position(x), None, 0.0) for x in strat.sleeve["targets"]},
                                             reason="formation sleeve stopped")
            for flag, attr in (("sniper", "sniper_on"), ("follow", "follow_on"), ("stubs", "stubs_on")):
                if flag in cmd:
                    setattr(strat, attr, bool(cmd[flag]))
            strat.paused |= set(cmd.get("pause", []))
            strat.paused -= set(cmd.get("resume", []))
            if cmd.get("close"):                                    # close specific symbols within 1% of mid
                goals = {}
                for x in cmd["close"]:
                    q = strat.broker.position(x)
                    if q and x in strat.quotes:
                        m_ = sum(strat.quotes[x]) / 2
                        goals[x] = (0, m_ * 1.01 if q < 0 else m_ * 0.99, abs(q) * m_)
                strat.broker.set_targets(goals, reason="manual close", urgent=True)
            if cmd.get("flatten"):
                strat.manual_flatten()
            log("CONTROL", cmd=cmd, halted=strat.halted, paused=sorted(strat.paused))

    async def tick_loop():
        while True:
            await asyncio.sleep(1)
            try:
                if not args.dry_run:
                    strat.tick()
            except Exception as exc:
                log("error", where="tick", error=repr(exc))

    if not args.dry_run:
        strat.broker.on_stub_fill = strat.on_stub_fill
    asyncio.create_task(status())
    asyncio.create_task(control_loop())
    asyncio.create_task(tick_loop())
    while True:
        try:
            async with websockets.connect(ws_url, additional_headers={"X-API-Key": key, "User-Agent": "hurricane-bot/1.0"}) as ws:
                await ws.send(json.dumps(sub))
                print("connected and subscribed", flush=True)
                while True:
                    # the server sends heartbeats/snapshots every 30 s even between rounds: 40 s of silence = dead link
                    frame = await asyncio.wait_for(ws.recv(), timeout=40)
                    data = json.loads(frame)
                    typ = data.get("type")
                    if typ not in first:
                        first.add(typ); log("first frame of type (check format)", frame=data)
                    if hasattr(strat.broker, "last_frame"):
                        strat.broker.last_frame = time.monotonic()
                    try:
                        if typ == "news" and isinstance(data.get("message"), dict):
                            feed_file.write(json.dumps({**data, "received_at": time.time()}) + "\n"); feed_file.flush()
                            strat.on_news(data["message"])
                        elif typ == "quote":
                            strat.on_quote(data.get("symbol"), data.get("bid"), data.get("ask"))
                            if args.dry_run:
                                strat.broker.tick()
                        elif typ == "orderbook":
                            strat.books[data["symbol"]] = (data.get("bids") or [], data.get("asks") or [])
                            if not args.dry_run:
                                strat.on_book(data["symbol"])
                            book_file.write(json.dumps({**data, "received_at": time.time()}) + "\n")
                        elif typ == "trade":
                            trade_file.write(json.dumps({**data, "received_at": time.time()}) + "\n")
                    except Exception as exc:                       # never let one bad message kill the bot
                        log("error", where=typ, error=repr(exc))
        except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as exc:
            print(f"disconnected ({exc!r}); reconnecting in 2 s", flush=True)
            log("DISCONNECTED", error=repr(exc))
            await asyncio.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--key", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
