"""Out-of-sample replay (held-out seasons 501-600) with execution costs calibrated to the live testnet.
Prints PnL by symbol x round for one policy variant."""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategy as S
from replay import replay_season
from book_profile import build

# extra cost per share (fraction of price) so replay costs match what the live bot paid vs fair value
EXTRA = {"PORT": 0.002, "FERRY": 0.0012, "AQUA": 0.0021, "MAT": 0.0007, "GEN": 0.0005, "HAVEN": 0.0, "ISLE": 0.0, "GRID": 0.0, "RECON": 0.0}
VARIANTS = {
    "current":            dict(buffer=0.001, flip=True,  syms=None),
    "buffer 0.5%":        dict(buffer=0.005, flip=True,  syms=None),
    "buffer 1%":          dict(buffer=0.010, flip=True,  syms=None),
    "no flip":            dict(buffer=0.001, flip=False, syms=None),
    "buffer 0.5% no flip": dict(buffer=0.005, flip=False, syms=None),
    "no PORT/HAVEN":      dict(buffer=0.001, flip=True,  syms=["ISLE", "AQUA", "GRID", "FERRY", "RECON", "MAT", "GEN"]),
    "buffer 1% no flip":  dict(buffer=0.010, flip=False, syms=None),
    "buffer 1.5%":        dict(buffer=0.015, flip=True,  syms=None),
    "buffer 2%":          dict(buffer=0.020, flip=True,  syms=None),
    "buffer 3%":          dict(buffer=0.030, flip=True,  syms=None),
    "1% no flip, start age 30": dict(buffer=0.010, flip=False, syms=None, min_age=30),
    "1% no flip, start age 50": dict(buffer=0.010, flip=False, syms=None, min_age=50),
    "1% no flip, start age 90": dict(buffer=0.010, flip=False, syms=None, min_age=90),
    "1% no flip, start age 130": dict(buffer=0.010, flip=False, syms=None, min_age=130),
}
name = sys.argv[1]; v = VARIANTS[name]
S.BUFFER = v["buffer"]
def tweak(st):
    st.allow_flip = v["flip"]
    st.min_age = v.get("min_age", 10)
    if v["syms"]:
        st.storm_syms = v["syms"]
prof = build(); rows = []; per_round = []
for n in range(501, 601):
    replay_season(n, Path(__file__).resolve().parent / "models_holdout100.pkl", 0.1, profile=prof, per_symbol=rows,
                  per_round=per_round, extra=EXTRA, tweak=tweak)
D = pd.DataFrame(rows); D.to_pickle(Path(__file__).resolve().parent / f"logs/variant_{name.replace(' ', '_').replace('/', '')}.pkl")
seas = D.groupby("season").pnl.sum()
by = D.groupby("round").pnl.sum() / 100
print(f"{name:22s} ${seas.mean():7,.0f}/season  P>0 {(seas > 0).mean():.2f}  rounds: " + "  ".join(f"r{r} ${x:,.0f}" for r, x in by.items()))
