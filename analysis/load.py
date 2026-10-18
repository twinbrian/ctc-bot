"""Parse all research seasons into tidy pandas tables and cache them.

    from load import tables
    T = tables()            # dict of DataFrames; built once, then read from analysis/cache.pkl

Tables (one row per ...):
  storms      storm (season, round): true pattern, erratic flip, final damage %, insured loss
  obs         storm observation (every 2 s of storm age): x, y, vx, vy, radius, intensity
  official    official advisory: outlook, P(hurricane conditions) per island, cone
  wire        wire rumor: outlook
  impacts     first hurricane conditions on an island
  reports     damage report (preliminary + final)
  prices      house price sample (every 2 real s), with t = real seconds since the open
  settle      season settlement values
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HIST = ROOT / "history"
CACHE = Path(__file__).resolve().parent / "cache.pkl"
CASE = json.loads((ROOT / "case.json").read_text())

ISLANDS = list(CASE["islands"])
PHYS = ["PORT", "ISLE", "AQUA", "GRID", "FERRY"]
REB = ["RECON", "MAT", "GEN"]
INS = ["HAVEN", "REIN"]
STOCKS = PHYS + REB + INS
SYMS = STOCKS + ["ARCH", "REBLD"]


def _real_seconds(time: str) -> float:
    """Real seconds since the 09:30 open (market clock runs 30x real time)."""
    t = datetime.fromisoformat(time)
    return ((t.hour - 9) * 3600 + (t.minute - 30) * 60 + t.second) / 30.0


def build() -> dict[str, pd.DataFrame]:
    manifest = json.loads((HIST / "manifest.json").read_text())
    storms, obs, official, wire, impacts, reports, prices, settle = [], [], [], [], [], [], [], []
    for s in manifest["seasons"]:
        n, folder = s["season"], HIST / s["dir"]
        for st in s["storms"]:
            storms.append({"season": n, "round": st["round"], "storm": st["storm"], "pattern": st["pattern"],
                           "flip": st["turned_erratic_at"], "loss": st["industry_loss_musd"],
                           "season_loss": s["season_loss_musd"], "open": st["open"],
                           **{f"d_{k}": v for k, v in st["damage_pct"].items()}})
        with (folder / "feed.jsonl").open() as f:
            for line in f:
                m = json.loads(line)
                base = {"season": n, "round": m["round"], "seq": m["seq"], "t": _real_seconds(m["time"])}
                typ = m["type"]
                if typ == "storm":
                    obs.append({**base, **{k: m[k] for k in ("age", "x", "y", "vx", "vy", "radius", "intensity")}})
                elif typ == "advisory" and m["source"] == "official":
                    official.append({**base, "age": m["age"], "number": m["number"], "outlook": m["outlook"],
                                     **{f"p_{k}": v for k, v in m["hurricane_conditions"].items()},
                                     "cone": m["cone"]})
                elif typ == "advisory":
                    wire.append({**base, "age": m["age"], "outlook": m["outlook"]})
                elif typ == "impact":
                    impacts.append({**base, "age": m["age"], "island": m["island"]})
                elif typ == "damage_report":
                    reports.append({**base, "stage": m["stage"], "loss": m["industry_loss_musd"],
                                    **{f"d_{k}": v for k, v in m["damage_pct"].items()}})
                elif typ in ("storm_formed", "storm_dissipated", "track"):
                    pass
                else:
                    print("unknown message type", typ)
        mk = pd.read_csv(folder / "market.csv")
        mk.insert(0, "season", n)
        mk["t"] = mk["time"].map(_real_seconds)
        prices.append(mk)
        settle.append({"season": n, **json.loads((folder / "settlement.json").read_text())})
    return {
        "storms": pd.DataFrame(storms), "obs": pd.DataFrame(obs), "official": pd.DataFrame(official),
        "wire": pd.DataFrame(wire), "impacts": pd.DataFrame(impacts), "reports": pd.DataFrame(reports),
        "prices": pd.concat(prices, ignore_index=True), "settle": pd.DataFrame(settle),
    }


def tables(rebuild: bool = False) -> dict[str, pd.DataFrame]:
    if CACHE.exists() and not rebuild:
        return pickle.loads(CACHE.read_bytes())
    T = build()
    CACHE.write_bytes(pickle.dumps(T))
    return T


if __name__ == "__main__":
    T = tables(rebuild=True)
    for k, v in T.items():
        print(f"{k:9s} {v.shape}")
