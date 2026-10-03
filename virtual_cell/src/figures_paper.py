"""Paper figures (RECOMB draft) from cached result JSONs only.
Fig 1  The paradox. (a) Source-line radius gain over predict-the-mean vs coverage after transfer of the
       source-calibrated set: 14-model ladder (K562<->RPE1, both directions) plus tuned GEARS (4 directions).
       (b) Across the 4-line transfer matrix: offset energy Delta^2 vs transferred coverage (ridge, mean).
Fig 2  Mechanism + efficiency law. (a) Coverage with source radius before vs after removing the oracle offset.
       (b) Prop C: predicted vs observed radius change of the exact adapted ball at m = 50 (all directed pairs).
Fig 3  Pilots: radius vs m for pilot-only and the exact adapted ball ((m+1)/m x LOO radius), K562<->RPE1.
Main-text versions (Figs 2-3 of the submission): fig2_transfer, fig3_repair (--main).
usage: uv run python src/figures_paper.py [--main]"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
FIG = Path(__file__).resolve().parents[2] / "paper" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7})
NAME = {"ess": "K562", "k562m": "K562", "gwps": "K562gw", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}


def pairs():
    out = {}
    for f in ("offset_pairs_metrics.json", "nadig_pairs_metrics.json", "matrix_pairs_metrics.json", "rpe1_ess_pair_metrics.json"):
        for k, v in json.loads((R / f).read_text()).items():
            a, b = k.split("->")
            if a == "k562m" and f"ess->{b}" in out:  # duplicate source pipeline, keep Replogle version
                continue
            ctrl = a in ("hepg2e", "hepg2o") or NAME.get(a) == NAME.get(b) or {a, b} <= {"ess", "gwps", "k562m"}
            out[k] = (v, ctrl)
    return out


def fig1():
    L = json.loads((R / "ladder_metrics.json").read_text())
    G12 = json.loads((R / "gears12_summary.json").read_text())  # 12 directed pairs x 10 seeds
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for d, mk in (("K562->RPE1", "o"), ("RPE1->K562", "s")):
        for n, v in L[d].items():
            c = "k" if n == "mean" else ("C0" if "GO" in n else "C7")
            ax[0].scatter(v["gainA_pct"], v["covB_src"] - L[d]["mean"]["covB_src"], c=c, marker=mk, s=14, alpha=.8, lw=0)
    for k, v in G12.items():
        ax[0].scatter(100 * (v["mqA"] - v["gqA"]) / v["mqA"], v["gcov"] - v["mcov"], c="C3", marker="*", s=45, zorder=5)
    ax[0].axhline(0, ls="--", c="gray", lw=.8)
    ax[0].set_xlabel("source-line radius gain over predict-the-mean (%)")
    ax[0].set_ylabel("coverage minus predict-the-mean\n(same transfer pair)")
    ax[0].set_title("(a) better in source, less covered after transfer")
    ax[0].scatter([], [], c="C0", s=14, label="GO-feature models"); ax[0].scatter([], [], c="C7", s=14, label="gwps-feature models")
    ax[0].scatter([], [], c="k", s=14, label="predict-the-mean"); ax[0].scatter([], [], c="C3", marker="*", s=40, label="GEARS (tuned)")
    ax[0].legend(loc="lower left", frameon=False)
    P = pairs()
    for k, (v, ctrl) in P.items():
        for mod, c in (("ridge10", "C0"), ("mean", "k")):
            ax[1].scatter(v[mod]["Delta2"], v[mod]["covB_src"], c=c, marker="x" if ctrl else "o", s=16, lw=1 if ctrl else 0)
    ax[1].axhline(0.9, ls="--", c="gray", lw=.8)
    ax[1].set_xlabel(r"context offset energy $\Delta^2$"); ax[1].set_ylabel("coverage in target line")
    ax[1].set_title("(b) 4-line transfer matrix (x = same-line controls)")
    ax[1].scatter([], [], c="C0", s=14, label="ridge"); ax[1].scatter([], [], c="k", s=14, label="predict-the-mean")
    ax[1].legend(loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig1_paradox.pdf"); fig.savefig(FIG / "fig1_paradox.png", dpi=200); plt.close(fig)


def fig2():
    P = pairs()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    xs, ys, cs = [], [], []
    for k, (v, ctrl) in P.items():
        r = v["ridge10"]
        ax[0].plot([0, 1], [r["covB_src"], r["covB_src_offset"]], c="gray" if ctrl else "C0", lw=.8, alpha=.8)
        rho = r["Delta2"] / r["sB2"]
        pred = 100 * (51 / 50 * np.sqrt((50 / 49) / (1 + rho)) - 1)  # outer ball vs pilot-only, energy approximation
        obs = 100 * (r["exact_m50_q"] / r["pilot_m50_q"] - 1)
        xs.append(pred); ys.append(obs); cs.append("gray" if ctrl else "C0")
    ax[0].axhline(0.9, ls="--", c="gray", lw=.8); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["source radius", "source radius\n+ oracle offset"])
    ax[0].set_ylabel("coverage in target line (ridge)"); ax[0].set_title("(a) removing the shared offset")
    ax[1].scatter(xs, ys, c=cs, s=16)
    lim = [min(xs + ys) - 1, max(xs + ys) + 1]; ax[1].plot(lim, lim, ls=":", c="k", lw=.8)
    ax[1].axhline(0, c="gray", lw=.5); ax[1].axvline(0, c="gray", lw=.5)
    ax[1].set_xlabel("predicted change, Prop. C (%)"); ax[1].set_ylabel("observed change (%)")
    ax[1].set_title(f"(b) exact adapted ball vs pilot-only, m=50 (r={np.corrcoef(xs, ys)[0, 1]:.2f})")
    fig.tight_layout(); fig.savefig(FIG / "fig2_mechanism.pdf"); fig.savefig(FIG / "fig2_mechanism.png", dpi=200); plt.close(fig)


def fig3():
    A = json.loads((R / "adapt_metrics.json").read_text())
    ms = (10, 20, 50, 100)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for j, d in enumerate(("K562->RPE1", "RPE1->K562")):
        pil = [A[f"{d}_m{m}"]["ridge_pilot"][1] for m in ms]
        exa = [A[f"{d}_m{m}"]["ridge_offset"][1] * (m + 1) / m for m in ms]
        ax[j].plot(ms, pil, "o-", c="C2", label="pilot-only split CP")
        ax[j].plot(ms, exa, "o-", c="C0", label="adapt-then-calibrate (exact ball)")
        ax[j].set_xscale("log"); ax[j].set_xticks(ms); ax[j].set_xticklabels(ms)
        ax[j].set_xlabel("pilot perturbations m"); ax[j].set_ylabel("set radius"); ax[j].set_title(d.replace("->", r" $\to$ "))
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig3_pilots.pdf"); fig.savefig(FIG / "fig3_pilots.png", dpi=200); plt.close(fig)


BIG = {"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 10, "xtick.labelsize": 10,
       "ytick.labelsize": 10}


def fig_transfer():
    """Main-text Fig 2: (a) shift energy vs transferred coverage; (b) the inversion (same data as fig1, panels swapped)."""
    L = json.loads((R / "ladder_metrics.json").read_text())
    G12 = json.loads((R / "gears12_summary.json").read_text())
    with plt.rc_context(BIG):
        fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75))
        for k, (v, ctrl) in pairs().items():
            for mod, c in (("ridge10", "C0"), ("mean", "k")):
                ax[0].scatter(v[mod]["Delta2"], v[mod]["covB_src"], c=c, marker="x" if ctrl else "o", s=18, lw=1 if ctrl else 0)
        ax[0].axhline(0.9, ls="--", c="gray", lw=.8)
        ax[0].set_xlabel(r"shift energy $\Delta^2$"); ax[0].set_ylabel("coverage in target line")
        ax[0].set_title("(a) loss tracks the shift")
        ax[0].scatter([], [], c="C0", s=18, label="ridge"); ax[0].scatter([], [], c="k", s=18, label="predict-the-mean")
        ax[0].scatter([], [], c="gray", marker="x", s=18, lw=1, label="same-line controls")
        ax[0].legend(loc="lower left", frameon=False)
        for d, mk in (("K562->RPE1", "o"), ("RPE1->K562", "s")):
            for n, v in L[d].items():
                c = "k" if n == "mean" else ("C0" if "GO" in n else "C7")
                ax[1].scatter(v["gainA_pct"], v["covB_src"] - L[d]["mean"]["covB_src"], c=c, marker=mk, s=16, alpha=.8, lw=0)
        for k, v in G12.items():
            ax[1].scatter(100 * (v["mqA"] - v["gqA"]) / v["mqA"], v["gcov"] - v["mcov"], c="C3", marker="*", s=50, zorder=5)
        ax[1].axhline(0, ls="--", c="gray", lw=.8)
        ax[1].set_xlabel("source radius gain (%)"); ax[1].set_ylim(-0.215, 0.012)
        ax[1].set_ylabel("coverage minus\npredict-the-mean")
        ax[1].set_title("(b) tighter in source, less covered")
        ax[1].scatter([], [], c="C0", s=16, label="GO-feature models"); ax[1].scatter([], [], c="C7", s=16, label="gwps-feature models")
        ax[1].scatter([], [], c="C3", marker="*", s=45, label="GEARS (tuned)")
        ax[1].legend(loc="lower left", frameon=False)
        fig.tight_layout(); fig.savefig(FIG / "fig2_transfer.pdf"); fig.savefig(FIG / "fig2_transfer.png", dpi=200); plt.close(fig)


def fig_repair():
    """Main-text Fig 3: (a, b) radius vs number of pilots, K562<->RPE1; (c) Prop C predicted vs observed change at m=50."""
    A = json.loads((R / "adapt_metrics.json").read_text())
    ms = (10, 20, 50, 100)
    with plt.rc_context(BIG):
        fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.45))
        for j, d in enumerate(("K562->RPE1", "RPE1->K562")):
            pil = [A[f"{d}_m{m}"]["ridge_pilot"][1] for m in ms]
            exa = [A[f"{d}_m{m}"]["ridge_offset"][1] * (m + 1) / m for m in ms]
            ax[j].plot(ms, pil, "o-", c="C2", label="pilot-only", ms=4)
            ax[j].plot(ms, exa, "o-", c="C0", label="adapted", ms=4)
            ax[j].set_xscale("log"); ax[j].set_xticks(ms); ax[j].set_xticklabels(ms); ax[j].minorticks_off()
            ax[j].set_xlabel("pilots $m$"); ax[j].set_title("(" + "ab"[j] + ") " + d.replace("->", r"$\to$"))
        ax[0].set_ylabel("90% radius (RMSE)"); ax[1].legend(frameon=False, loc="upper right")
        xs, ys, cs = [], [], []
        for k, (v, ctrl) in pairs().items():
            r = v["ridge10"]; rho = r["Delta2"] / r["sB2"]
            xs.append(100 * (51 / 50 * np.sqrt((50 / 49) / (1 + rho)) - 1)); ys.append(100 * (r["exact_m50_q"] / r["pilot_m50_q"] - 1))
            cs.append("gray" if ctrl else "C0")
        ax[2].scatter(xs, ys, c=cs, s=16)
        lim = [min(xs + ys) - 1, max(xs + ys) + 1]; ax[2].plot(lim, lim, ls=":", c="k", lw=.8)
        ax[2].axhline(0, c="gray", lw=.5); ax[2].axvline(0, c="gray", lw=.5)
        ax[2].set_xlabel("predicted change (%)"); ax[2].set_ylabel("observed change (%)")
        ax[2].set_title("(c) Prop. C, $m$=50")
        fig.tight_layout(); fig.savefig(FIG / "fig3_repair.pdf"); fig.savefig(FIG / "fig3_repair.png", dpi=200); plt.close(fig)
        print("fig3c r =", round(float(np.corrcoef(xs, ys)[0, 1]), 3), "n =", len(xs))


if __name__ == "__main__":
    import sys
    if "--main" in sys.argv:
        fig_transfer(); fig_repair()
    else:
        fig1(); fig2(); fig3()
    print("PAPER_FIGS_DONE", FIG)
