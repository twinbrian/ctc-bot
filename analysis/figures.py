"""Figures for FINDINGS.md (static PNGs)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import tables, SYMS
from value import haven, rein
from features import posterior

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})
T = tables()


def insurance():
    ST, S = T["storms"], T["settle"].set_index("season")
    L = ST.groupby("season").loss.sum()
    x = np.linspace(0, 1900, 400)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(L, S.loc[L.index, "HAVEN"], s=10, color=BLUE, alpha=0.5, linewidths=0, label="HAVEN settlement")
    ax.scatter(L, S.loc[L.index, "REIN"], s=10, color=ORANGE, alpha=0.5, linewidths=0, label="REIN settlement")
    ax.plot(x, haven(x), color=BLUE, lw=2); ax.plot(x, rein(x), color=ORANGE, lw=2)
    for v in (400, 1300):
        ax.axvline(v, color=INK2, lw=0.8)
    ax.text(410, 50, "HAVEN retention $400M", color=INK2, fontsize=9)
    ax.text(1310, 50, "HAVEN limit = REIN attach $1,300M", color=INK2, fontsize=9)
    ax.text(150, 77, "HAVEN = 75 − 0.06·clip(L−400, 0, 900)", color=INK, fontsize=9)
    ax.text(150, 92, "REIN = 90 − 0.035·max(L−1300, 0)", color=INK, fontsize=9)
    ax.set_xlabel("Season insured loss L ($M)"); ax.set_ylabel("Settlement ($/share)")
    ax.set_title("Insurers are exact call-spreads on season loss (600 seasons)", loc="left", fontsize=11)
    ax.legend(loc="lower left"); ax.set_ylim(15, 98)
    fig.tight_layout(); fig.savefig("fig_insurance.png", dpi=150); plt.close(fig)


def house_path():
    P = T["prices"].sort_values(["season", "round", "t"])
    t = np.arange(0, 780, 2)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s, c in (("HAVEN", BLUE), ("PORT", ORANGE), ("RECON", AQUA)):
        h = P[s].values.reshape(1800, 390)
        m = (h - h[:, :1]).mean(0)
        ax.plot(t, m, color=c, lw=2, label=s)
        ax.text(782, m[-1], f" {s}", va="center", color=INK, fontsize=9)
    for x, lab in ((60, "storm forms"), (540, "dissipates"), (570, "prelim"), (740, "final")):
        ax.axvline(x, color=INK2, lw=0.8)
        if x == 60:
            ax.text(x - 4, 1.6, lab, color=INK2, fontsize=8, rotation=90, va="top", ha="right")
        else:
            ax.text(x + 4, 4.1, lab, color=INK2, fontsize=8, rotation=90, va="top")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Real seconds since the round open"); ax.set_ylabel("Mean change from open ($/share)")
    ax.set_title("House jumps optimistic 4 s after a storm forms, then drifts back (1,800 storms)", loc="left", fontsize=11)
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.6)); ax.set_xlim(0, 830)
    fig.tight_layout(); fig.savefig("fig_house_path.png", dpi=150); plt.close(fig)


def pattern_id():
    ST = T["storms"].set_index(["season", "round"]); A = T["official"]; Wr = T["wire"]
    msgs = pd.concat([A[["season", "round", "age", "outlook"]].assign(src="official"),
                      Wr[["season", "round", "age", "outlook"]].assign(src="wire")]).sort_values(["season", "round", "age"])
    ages = np.arange(30, 481, 30)
    acc = {"latest": np.zeros(len(ages)), "bayes": np.zeros(len(ages))}
    G = dict(list(msgs.groupby(["season", "round"])))
    for k, g in G.items():
        pat, flip = ST.at[k, "pattern"], ST.at[k, "flip"]
        for i, a in enumerate(ages):
            truth = "erratic" if (pd.notna(flip) and a >= flip) else pat
            h = g[g.age <= a]; off = h[h.src == "official"]
            post = posterior(list(zip(h.src, h.outlook)))
            acc["bayes"][i] += max(post, key=post.get) == truth
            acc["latest"][i] += off.outlook.iloc[-1] == truth
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for key, c, lab in (("latest", ORANGE, "latest official advisory"), ("bayes", BLUE, "Bayes: all advisories + wire")):
        y = acc[key] / len(G)
        ax.plot(ages, y, color=c, lw=2, marker="o", ms=4, label=lab)
        ax.text(486, y[-1], f" {y[-1]:.0%}", va="center", color=INK, fontsize=9)
    ax.set_ylim(0.6, 1.0); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Storm age (real seconds)"); ax.set_ylabel("True pattern identified")
    ax.set_title("Advisories are i.i.d. 70%-accurate draws: aggregate them", loc="left", fontsize=11)
    ax.legend(loc="lower right"); ax.set_xlim(20, 520)
    fig.tight_layout(); fig.savefig("fig_pattern_id.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    insurance(); house_path(); pattern_id()
