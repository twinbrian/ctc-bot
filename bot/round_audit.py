"""Reconstruct every live round from the bot's own logs: exact PnL per round, plus what was going on.

    /opt/anaconda3/bin/python3 -I bot/round_audit.py            -> bot/logs/live_rounds.csv
"""
import bisect, collections, glob, json, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "starter_kit"))
from history import load
H = load(ROOT / "history")
SY = ["PORT","ISLE","AQUA","GRID","FERRY","RECON","MAT","GEN","HAVEN","REIN","ARCH","REBLD"]

def build(since="run-20261018-065413"):
    runs = [f for f in sorted(glob.glob(str(ROOT / "bot/logs/run-*.jsonl"))) if Path(f).name >= since + ".jsonl"]
    fills, forms, notes, restarts = [], [], collections.defaultdict(list), []
    for f in runs:
        recs = [json.loads(l) for l in open(f)]; start = recs[0]["ts"] + 5
        restarts.append(recs[0]["ts"])
        for r in recs:
            if r["ts"] < start: continue
            e = r["event"]
            if e == "fill": fills.append((r["ts"], r["symbol"], r["qty"], r["price"]))
            elif e == "storm_formed": forms.append(r["ts"])
            elif e in ("SNIPE", "ORDER REJECTED", "DISCONNECTED", "error", "ANNOUNCEMENT - PAUSED", "PREDICTION REJECTED (implausible)",
                       "JOINED LATE: outage history unknown -> no in-storm trading this season; final-report trade limited to outage-free symbols"):
                notes[e].append(r["ts"])
    # storm identity (research season, round) from the recorded feed
    ident = []
    for f in sorted(glob.glob(str(ROOT / "bot/logs/feed-*.jsonl"))):
        for line in open(f):
            fr = json.loads(line); m = fr.get("message") or {}
            if m.get("type") == "storm_formed": ident.append((fr["received_at"], m["time"][:10], m["round"], m["storm"]))
    ident.sort(); it = [x[0] for x in ident]
    forms = sorted(set(forms))
    books = collections.defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "bot/logs/books-*.jsonl"))):
        for line in open(f):
            b = json.loads(line)
            if b.get("bids") and b.get("asks"): books[b["symbol"]].append((b["received_at"], (b["bids"][0]["price"] + b["asks"][0]["price"]) / 2))
    bt = {s: [x[0] for x in v] for s, v in books.items()}
    def mid(s, t):
        i = bisect.bisect_right(bt[s], t) - 1; return books[s][i][1] if i >= 0 else 0.0
    rows, pos = [], collections.defaultdict(int)
    for k, t0 in enumerate(forms):
        j = bisect.bisect_right(it, t0 + 3) - 1
        _, date, rnd, name = ident[j]
        season = H.season_of(date).number
        t1 = forms[k + 1] if k + 1 < len(forms) else time.time()
        if rnd == 1:
            pos = collections.defaultdict(int)                       # the exchange resets positions each season
        start = dict(pos); cash = 0.0
        for ts, s, q, px in fills:
            if t0 <= ts < t1:
                cash -= q * px; pos[s] += q
        done = k + 1 < len(forms)
        if rnd == 3 and done:
            marks = H.seasons[season - 1].settlement; end_val = sum(pos[s] * marks[s] for s in SY); pos = collections.defaultdict(int)
        else:
            end_val = sum(q * mid(s, t1) for s, q in pos.items())
        start_val = sum(q * mid(s, t0) for s, q in start.items())
        pnl = cash + end_val - start_val
        note = {e: sum(t0 <= x < t1 for x in v) for e, v in notes.items()}
        rows.append({"start": time.strftime("%H:%M", time.localtime(t0)), "season": season, "round": rnd, "storm": name,
                     "pnl": round(pnl), "complete": done, "restarts": sum(t0 < x < t1 for x in restarts),
                     **{k_[:12]: v for k_, v in note.items() if v}})
    return pd.DataFrame(rows).fillna(0)

if __name__ == "__main__":
    D = build(); D.to_csv(ROOT / "bot/logs/live_rounds.csv", index=False)
    pd.set_option("display.width", 220)
    print(D.to_string(index=False))
    C = D[D.complete]
    print(f"\ncomplete rounds: {len(C)}, total ${C.pnl.sum():,.0f}; rounds < -$3k: {(C.pnl < -3000).sum()}")
