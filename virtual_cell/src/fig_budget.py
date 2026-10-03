"""Pilot-budget figure (main text): when does a learned cross-line map buy narrower exact sets?
(a) Single source line, 12 directed pairs: radius of the map's exact split-conformal set relative to the current recipe
    (Prop. A outer ball, adapted ridge, all pilots reused) on the same seeds and pilots; median over pairs and range;
    one line per fitting fraction (xsplit.py) and the PerturbMap-style blend at 1/2 (xmap.py).
(b) Coverage of the same sets (median over pairs, range).
(c) Three source lines, per target: average of the other lines + pilot offset (Prop. A, exact) and learned line weights
    + reduced-rank residual (split conformal, 1/2), relative to the Prop. A ball of the pilot mean.
Inputs: results/xsplit_*_s20.json, results/xpropa_s20_metrics.json, results/pod_x/{xmap_pairs,xmulti_pairs}.
usage: uv run python src/fig_budget.py"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"; FIG = Path(__file__).resolve().parents[2] / "paper" / "figs"
NM = {"ess": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
pa = json.loads((R / "xpropa_s20_metrics.json").read_text())
XS = {}
for f in glob.glob(str(R / "xsplit_*_s20.json")): XS.update(json.loads(Path(f).read_text()))
XM = {}
for f in glob.glob(str(R / "pod_x" / "xmap_pairs" / "*_s20.json")): XM.update(json.loads(Path(f).read_text()))
MU = {}
for f in glob.glob(str(R / "pod_x" / "xmulti_pairs" / "*_s20.json")): MU.update(json.loads(Path(f).read_text()))
MS = (50, 100, 200)


def rel(num, den):
    return float(np.mean(np.array(num) / np.array(den) - 1))


arms = [(f"rrr_f{f}", f"map, fit {f}%", c) for f, c in ((50, "C0"), (60, "C2"), (70, "C3"))]
summary = []
with plt.rc_context({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10}):
    fig, ax = plt.subplots(1, 3, figsize=(6.5, 3.05), gridspec_kw={"width_ratios": [1.15, 1, 1.15]})
    for key, lab, c in arms + [("pmap", "PerturbMap-st., 50%", "C7")]:
        med, lo, hi, cm, cl, ch, ms = [], [], [], [], [], [], []
        for m in MS:
            v, cv = [], []
            for pair in pa:
                if "->" not in pair or pair.startswith("others"): continue
                if key in ("pmap", "rrr_f50"):          # the 1/2 split is exactly xmap.py's run
                    src = XM[pair]["raw"]; kk = "pmap" if key == "pmap" else "rrr"
                else:
                    src = XS.get(pair, {}).get("raw", {}); kk = key
                k = f"{kk}_q_{m}"
                if k not in src: continue
                v.append(rel(src[k], pa[pair]["raw"][f"propA_ridge_q_{m}"])); cv.append(np.mean(src[f"{kk}_cov_{m}"]))
            if len(v) < 12: continue
            ms.append(m); med.append(np.median(v)); lo.append(min(v)); hi.append(max(v)); cm.append(np.median(cv)); cl.append(min(cv)); ch.append(max(cv))
            summary.append(f"{lab} m={m}: radius vs current recipe median {100 * np.median(v):+.1f}% [{100 * min(v):+.1f}, {100 * max(v):+.1f}], below 0 in {sum(x < 0 for x in v)}/12 (mean over seeds); coverage {min(cv):.3f}-{max(cv):.3f}")
        if not ms: continue
        ax[0].plot(ms, 100 * np.array(med), "o-", c=c, label=lab, ms=4); ax[0].fill_between(ms, 100 * np.array(lo), 100 * np.array(hi), color=c, alpha=0.12)
        ax[1].plot(ms, cm, "o-", c=c, ms=4); ax[1].fill_between(ms, cl, ch, color=c, alpha=0.12)
    for a in ax[:2]:
        a.set_xscale("log"); a.set_xticks(MS); a.set_xticklabels(MS); a.minorticks_off(); a.set_xlabel("pilots $m$")
    ax[0].axhline(0, c="k", lw=0.8); ax[0].set_ylabel("radius change (%)"); ax[0].set_title("(a) one line, vs current")
    ax[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.55, -0.24), ncol=2, columnspacing=0.8, handlelength=1.5)
    ax[1].axhline(0.9, c="k", lw=0.8, ls="--"); ax[1].set_ylabel("coverage"); ax[1].set_title("(b) coverage"); ax[1].set_ylim(0.85, 0.97)
    for i, (B, c) in enumerate(zip(("ess", "rpe1", "hepg2", "jurkat"), ("C0", "C1", "C2", "C3"))):
        key = f"others->{B}"; p = pa[key]["raw"]; x = MU[key]["raw"]; mm = (25, 50, 100, 200)
        avg = [100 * rel(p[f"propA_avg_q_{m}"], p[f"propA_mmean_q_{m}"]) for m in mm]
        lrn = [100 * rel(x[f"synth_rrr_q_{m}"], p[f"propA_mmean_q_{m}"]) for m in mm]
        ax[2].plot(mm, avg, "s--", c=c, ms=3.5); ax[2].plot(mm, lrn, "o-", c=c, ms=3.5, label=None)
        summary.append(f"three lines -> {NM[B]}: avg+offset (Prop A) " + " ".join(f"{v:+.1f}" for v in avg) + " | learned " + " ".join(f"{v:+.1f}" for v in lrn) + " (m=25,50,100,200; % vs Prop A pilot mean)")
    ax[2].set_xscale("log"); ax[2].set_xticks((25, 50, 100, 200)); ax[2].set_xticklabels((25, 50, 100, 200)); ax[2].minorticks_off()
    ax[2].axhline(0, c="k", lw=0.8); ax[2].set_xlabel("pilots $m$"); ax[2].set_ylabel("radius change (%)")
    ax[2].set_title("(c) three lines, vs mean")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], c=c, marker="o", ms=3.5) for c in ("C0", "C1", "C2", "C3")] + [Line2D([], [], c="k", ls="--", marker="s", ms=3.5), Line2D([], [], c="k", marker="o", ms=3.5)]
    ax[2].legend(h, ["K562", "RPE1", "HepG2", "Jurkat", "average + offset", "learned (split)"], frameon=False, fontsize=8,
                 loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, columnspacing=0.8, handlelength=1.5)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.36, wspace=0.55); fig.savefig(FIG / "fig5_budget.pdf"); fig.savefig(FIG / "fig5_budget.png", dpi=200)
(R / "fig_budget_summary.txt").write_text("\n".join(summary), encoding="utf-8"); print("\n".join(summary))
