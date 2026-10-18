"""Round-3 analysis frame: house prices at key times + value under different damage estimates."""
from __future__ import annotations
import numpy as np, pandas as pd
from load import tables, PHYS, STOCKS, SYMS
from wind import SITES
from damage import exposure, roll_season, company_damage
from value import values

def build():
    T = tables(); ST = T["storms"].sort_values(["season","round"]).reset_index(drop=True)
    z = np.load("storm_tensors.npz"); U, I = z["U"], z["I"]; p = np.load("damage_params.npy")
    pred = company_damage(roll_season(exposure(U, I, p[0], p[1]), np.exp(p[2:]), ST.season.values, ST["round"].values))/100
    keys = list(zip(ST.season, ST["round"])); R = T["reports"]
    pre = R[R.stage=="preliminary"].set_index(["season","round"]).loc[keys].reset_index()
    Y = ST[[f"d_{c}" for c in PHYS]].reset_index(drop=True)/100; Y.columns = PHYS
    Pr = pre[[f"d_{c}" for c in PHYS]]/100; Pr.columns = PHYS
    F = pd.read_pickle("site_features.pkl").merge(SITES[["company","weight"]], left_on="site_i", right_index=True)
    F["wc"] = F.closed_s*F.weight
    Cs = F.groupby(["season","company"]).wc.sum().unstack().sort_index()
    r3 = (ST["round"]==3).values; seas = ST.season.values[r3]
    Dprev = {c: Y[c].values.reshape(-1,3)[:, :2].sum(1) for c in PHYS}
    C = {c: Cs.loc[seas, c].values for c in PHYS}
    V = {name: values({c: Dprev[c] + d[c].values[r3] for c in PHYS}, C) for name, d in (("final", Y), ("prelim", Pr), ("model", pred))}
    P3 = T["prices"]; P3 = P3[P3["round"]==3].set_index(["season","t"])
    H = {t: P3.xs(float(t), level="t").loc[seas] for t in (58, 300, 420, 510, 538, 546, 600, 738)}
    return dict(seas=seas, V=V, H=H, S=T["settle"].set_index("season").loc[seas], Y=Y[r3].reset_index(drop=True),
                Pr=Pr[r3].reset_index(drop=True), M=pred[r3].reset_index(drop=True), Dprev=Dprev, C=C)
