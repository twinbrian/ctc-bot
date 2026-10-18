"""Replay held-out seasons under different after-storm policies; report PnL per round."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay import replay_season
from book_profile import build

variants = {"flatten all rounds": ((), False), "hold after storm in round 3": ((3,), False),
            "hold after storm in all rounds": ((1, 2, 3), False), "flatten + front-run": ((), True),
            "hold r3 + front-run": ((3,), True)}
name = sys.argv[1]; a, b = map(int, sys.argv[2].split("-"))
prof = build(); m = Path(__file__).resolve().parent / "models_holdout100.pkl"
rows = []
for n in range(a, b + 1):
    replay_season(n, m, 0.1, profile=prof, hold_rounds=variants[name][0], per_round=rows, front_run=variants[name][1])
R = pd.DataFrame(rows); R["total"] = R[["r1", "r2", "r3"]].sum(1)
print(f"{name:32s} seasons {a}-{b}: mean/season ${R.total.mean():,.0f}  P>0 {(R.total > 0).mean():.2f}  "
      f"by round r1 ${R.r1.mean():,.0f}  r2 ${R.r2.mean():,.0f}  r3 ${R.r3.mean():,.0f}")
