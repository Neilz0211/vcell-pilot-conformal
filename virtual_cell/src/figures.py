"""Figures from the cached results JSONs (no recomputation).
F1 ladder: source radius gain vs target coverage of the source-calibrated set (both directions).
F2 pilots: coverage and radius vs m for pilot-only / P2 / P3(g=20), disjoint library and K562->RPE1.
F3 natural library shift (day2): coverage vs essential share, unweighted / oracle / estimated weights.
usage: uv run python src/figures.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[1] / "figures"
FIG.mkdir(exist_ok=True)


def f1():
    L = json.loads((R / "ladder_metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, d in zip(axes, ("K562->RPE1", "RPE1->K562")):
        for n, v in L[d].items():
            c = "k" if n == "mean" else ("C0" if "GO" in n else "C1")
            ax.scatter(v["gainA_pct"], v["covB_src"], c=c, s=25)
            ax.scatter(v["gainA_pct"], v["gainB_pct"] / 100 + 0.6, marker="x", c=c, s=18, alpha=.5)
        ax.axhline(0.9, ls="--", c="gray", lw=.8)
        ax.set_title(f"{d}  (Spearman {L[d + '_spearman_qA_covB']:.2f})")
        ax.set_xlabel("source-line radius gain over mean (%)")
    axes[0].set_ylabel("target-line coverage of source-calibrated set")
    fig.suptitle("Better in the source line -> more overconfident in the new line (o coverage; x target gain/100+0.6)")
    fig.tight_layout(); fig.savefig(FIG / "F1_ladder.png", dpi=150); plt.close(fig)


def f2():
    P = json.loads((R / "pilot2_metrics.json").read_text())
    ms = (9, 10, 20, 50, 100)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for j, (grp, sc) in enumerate((("library", "disjoint"), ("context", "K562->RPE1"))):
        a = P[grp][sc]
        for p, c in (("pilot", "C2"), ("P2", "C0"), ("P3g20", "C3")):
            axes[0, j].plot(ms, [a[f"{p}_m{m}"][0] for m in ms], "o-", c=c, label=p)
            axes[1, j].plot(ms, [a[f"{p}_m{m}"][2] for m in ms], "o-", c=c, label=p)
        axes[0, j].axhline(a["src"][0], c="k", ls=":", label="source-calibrated")
        axes[0, j].axhline(0.9, c="gray", ls="--"); axes[0, j].set_title(sc); axes[0, j].set_ylabel("coverage")
        axes[1, j].axhline(a["src"][2], c="k", ls=":"); axes[1, j].set_ylabel("mean radius"); axes[1, j].set_xlabel("pilots m")
        for ax in axes[:, j]:
            ax.set_xscale("log")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "F2_pilots.png", dpi=150); plt.close(fig)


def f3():
    D = json.loads((R / "day2_metrics.json").read_text())["cases"]
    keys = sorted([k for k in D if k.startswith("mix_")], key=lambda k: float(k.split("test")[1]))
    xs = [float(k.split("test")[1]) for k in keys]
    fig, ax = plt.subplots(figsize=(5, 4))
    for f, c, lab in (("unw", "C3", "unweighted"), ("orc", "C0", "oracle weights"), ("est", "C1", "estimated weights")):
        ax.plot(xs, [D[k][f] for k in keys], "o-", c=c, label=lab)
    ax.axhline(0.9, c="gray", ls="--"); ax.set_xlabel("essential share of screening library"); ax.set_ylabel("coverage")
    ax.legend(); ax.set_title("Natural library shift (K562 gwps)")
    fig.tight_layout(); fig.savefig(FIG / "F3_library.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    f1(); f2(); f3(); print("FIGS_DONE")
