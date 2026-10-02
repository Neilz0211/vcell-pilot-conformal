"""Figure 5: where the residual error comes from (results/ceiling_{all,top100,pc10}_metrics.json).
(a) Per directed pair, 90 % radii (whole-transcriptome readout, m = 50, leak-free): noise floor (perfect predictor of
the expected response), adapted ridge, adapted pilot mean, pilot-only ridge. (b) Noise share of the adapted-ridge
residual energy by readout. usage: uv run python src/fig_attribution.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT.parent / "paper" / "figs"
NAME = {"ess": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
R = {r: json.load(open(ROOT / "results" / f"ceiling_{r}_metrics.json")) for r in ("all", "top100", "pc10")}
pairs = sorted(R["all"], key=lambda k: R["all"][k]["floor"] / R["all"][k]["ad_ridge"])
lab = [f"{NAME[k.split('->')[0]]}$\\to${NAME[k.split('->')[1]]}" for k in pairs]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.1, 1]})
x = np.arange(len(pairs)); w = 0.2
for j, (key, nm, c) in enumerate((("floor", "noise floor", "0.6"), ("ad_ridge", "adapted ridge", "C0"),
                                    ("ad_mean", "adapted pilot mean", "C1"), ("pil_ridge", "pilot-only ridge", "C2"))):
    ax[0].bar(x + (j - 1.5) * w, [R["all"][k][key] for k in pairs], w, label=nm, color=c)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=60, ha="right", fontsize=6.5)
ax[0].set_ylabel("90% radius (RMSE)"); ax[0].legend(fontsize=6.5, frameon=False, ncol=2)
ax[0].set_title("(a) radii, whole-transcriptome readout", fontsize=8)
for i, (r, nm) in enumerate((("all", "1,000 genes"), ("top100", "top 100 genes"), ("pc10", "10 programmes"))):
    v = [R[r][k]["noise_share"] for k in pairs]
    ax[1].scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=10, color="C0")
    ax[1].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1)
ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["1,000 genes", "top 100", "10 PCs"], fontsize=7)
ax[1].set_ylabel("noise share of residual energy"); ax[1].set_ylim(0, 0.5)
ax[1].set_title("(b) measurement noise by readout", fontsize=8)
fig.tight_layout(); FIG.mkdir(exist_ok=True)
fig.savefig(FIG / "fig5_attribution.pdf"); fig.savefig(FIG / "fig5_attribution.png", dpi=200)
print("wrote", FIG / "fig5_attribution.pdf")
