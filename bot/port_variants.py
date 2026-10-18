"""Out-of-sample test of PORT-specific entry rules under the production config (200 held-out seasons, honest
models, live-calibrated costs). PORT is persistently worse live than in replay, so the question is whether a
higher PORT entry threshold (or no PORT) is better once PORT's extra live cost is included.

    OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3 -I bot/port_variants.py "<variant>" <port extra cost>
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategy as S
from replay import replay_season
from book_profile import build

BOT = Path(__file__).resolve().parent
HONEST = [(401, 500, BOT / "models_holdout_401_500.pkl"), (501, 600, BOT / "models_holdout100.pkl")]
VARIANTS = {"production": {}, "PORT buffer 2%": {"PORT": 0.02}, "PORT buffer 3%": {"PORT": 0.03},
            "no PORT": None}
name, port_cost = sys.argv[1], float(sys.argv[2])
if len(sys.argv) > 3 and sys.argv[3] == "61-160":                 # a third held-out set (covers the live seasons 71-82)
    HONEST = [(61, 160, BOT / "models_holdout_61_160.pkl")]
extra = {"PORT": port_cost, "FERRY": 0.0012, "AQUA": 0.0021, "MAT": 0.0007, "GEN": 0.0005}
S.BUFFER, S.EDGE_SHARE = 0.01, 0.7


def tweak(st):
    st.allow_flip, st.min_age = False, 30
    v = VARIANTS[name]
    if v is None:
        st.storm_syms = [s for s in st.storm_syms if s != "PORT"]
    else:
        st.sym_buffer = dict(v)


prof, rows = build(), []
for a, b, model in HONEST:
    for n in range(a, b + 1):
        replay_season(n, model, 0.1, profile=prof, per_symbol=rows, extra=extra, tweak=tweak)
D = pd.DataFrame(rows)
D.to_pickle(BOT / "logs" / f"portvar_{name.replace(' ', '_').replace('%', 'pct')}_{port_cost}_{HONEST[0][0]}.pkl")
seas = D.groupby("season").pnl.sum()
port = D[D.symbol == "PORT"].groupby("season").pnl.sum().reindex(seas.index, fill_value=0)
print(f"{name:16s} PORT cost {port_cost:.4f}: ${seas.mean():7,.0f}/season (se {seas.std() / len(seas) ** .5:,.0f})  "
      f"P>0 {(seas > 0).mean():.2f}  worst ${seas.min():,.0f}  PORT ${port.mean():,.0f}/season  seasons {seas.index.min()}-{seas.index.max()}", flush=True)
