"""Build a per-symbol order-book depth profile from recorded testnet books (bot/logs/books-*.jsonl).

profile[s] = [(offset from mid as a fraction, qty), ...]  -- median cumulative depth on one side
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = np.array([0.0005, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05])


def storm_windows():
    """(start, end) wall-clock windows of storms seen in the run logs (storm_formed + 480 s)."""
    out = []
    for r in sorted(glob.glob(str(ROOT / "bot" / "logs" / "run-*.jsonl"))):
        for line in open(r):
            e = json.loads(line)
            if e["event"] == "storm_formed":
                out.append((e["ts"], e["ts"] + 480))
    return out


def build(paths=None, storm_only=True):
    """Depth profile; by default only from books recorded while a storm was live (wider, thinner)."""
    paths = paths or sorted(glob.glob(str(ROOT / "bot" / "logs" / "books-*.jsonl")))
    wins = storm_windows() if storm_only else []
    cum = {}
    for p in paths:
        for line in open(p):
            b = json.loads(line)
            if storm_only and wins and not any(a <= b.get("received_at", 0) <= z for a, z in wins):
                continue
            bids, asks = b.get("bids") or [], b.get("asks") or []
            if not bids or not asks:
                continue
            mid = (bids[0]["price"] + asks[0]["price"]) / 2
            for side, lv in (("ask", asks), ("bid", bids)):
                off = np.array([abs(l["price"] / mid - 1) for l in lv]); q = np.array([l["quantity"] for l in lv])
                cum.setdefault(b["symbol"], []).append([q[off <= g].sum() for g in GRID])
    prof = {}
    for s, rows in cum.items():
        med = np.median(np.array(rows), axis=0)
        lv, prev = [], 0.0
        for g, c in zip(GRID, med):
            if c > prev:
                lv.append((float(g), float(c - prev))); prev = c
        prof[s] = lv
    return prof


if __name__ == "__main__":
    for s, lv in build().items():
        print(s, [(round(o * 100, 2), int(q)) for o, q in lv])
