"""Run the real sniper code over every recorded live order book + news message, in time order.
Each snipe is scored at the mid 30 s later minus half the spread (the cost to get out)."""
import bisect, collections, glob, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot"))
import strategy as S

LIMITS = json.loads((ROOT / "case.json").read_text())["limits"]
now = {"t": 0.0}

class DummyBroker:
    def position(self, s): return 0
    def set_targets(self, goals, reason="", urgent=False): pass
    def clear(self): pass
    def sync_stubs(self, d): pass

def run(old_rules=False):
    events = []
    st = S.Strategy(ROOT / "bot" / "models.pkl", LIMITS, DummyBroker(), lambda e, **kw: events.append((now["t"], e, kw)),
                    clock=lambda: now["t"], persist=False)
    if old_rules:                                     # emulate the rules before today's fix
        S.SNIPE_STABLE, S.SNIPE_PERSIST, S.SNIPE_MAX_FRAC = 1e9, 0.0, 1.0
    stream = []
    for f in sorted(glob.glob(str(ROOT / "bot/logs/feed-*.jsonl"))):
        for line in open(f):
            fr = json.loads(line)
            if isinstance(fr.get("message"), dict): stream.append((fr["received_at"], 0, fr["message"]))
    for f in sorted(glob.glob(str(ROOT / "bot/logs/books-*.jsonl"))):
        for line in open(f):
            b = json.loads(line); stream.append((b["received_at"], 1, b))
    stream.sort(key=lambda x: (x[0], x[1]))
    mids = collections.defaultdict(list)
    for t, kind, x in stream:
        if kind == 1 and x.get("bids") and x.get("asks"):
            mids[x["symbol"]].append((t, (x["bids"][0]["price"] + x["asks"][0]["price"]) / 2, (x["asks"][0]["price"] - x["bids"][0]["price"]) / 2))
    last_t = None
    for t, kind, x in stream:
        if last_t is not None and t - last_t > 100:      # recording gap / restart: the bot would start fresh
            st.hm_hist.clear(); st.sighted.clear(); st.last_news_t = t
        last_t = t; now["t"] = t
        if kind == 0:
            st.on_news(x)
        elif x.get("bids") and x.get("asks"):
            st.books[x["symbol"]] = (x["bids"], x["asks"])
            st.quotes[x["symbol"]] = (x["bids"][0]["price"], x["asks"][0]["price"])
            st.on_book(x["symbol"]); st.snipes.clear()
    res = []
    for t, e, kw in events:
        if e != "SNIPE": continue
        arr = mids[kw["symbol"]]; i = bisect.bisect_right([a[0] for a in arr], t + 30) - 1
        _, m30, half = arr[i]
        j = bisect.bisect_right([a[0] for a in arr], t) - 1
        px = arr[j][1] + (arr[j][2] if kw["side"] == "buy" else -arr[j][2])   # filled at the touch
        px = min(px, kw["cap"]) if kw["side"] == "buy" else max(px, kw["cap"])
        pnl = (m30 - half - px) * kw["qty"] if kw["side"] == "buy" else (px - m30 - half) * kw["qty"]
        res.append((time.strftime("%H:%M:%S", time.localtime(t)), kw["symbol"], kw["side"], kw["qty"], round(kw["house_mid"], 2), round(pnl)))
    return res

if __name__ == "__main__":
    for label, old in (("OLD rules (before the fix)", True), ("NEW rules", False)):
        import importlib; importlib.reload(S)
        r = run(old_rules=old)
        print(f"{label}: {len(r)} snipes, total ${sum(x[-1] for x in r):,.0f}, losing {sum(x[-1] < 0 for x in r)}")
        for x in r[:12]: print("   ", x)
