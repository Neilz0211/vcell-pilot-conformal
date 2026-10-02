"""Figure: pathway-score intervals (non-RMSE scores), from results/pathway_metrics.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = json.loads((Path(__file__).resolve().parents[1] / "results" / "pathway_metrics.json").read_text())
FIG = Path(__file__).resolve().parents[2] / "paper" / "figs"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})
progs = ["cell cycle", "p53 signalling", "unfolded protein response", "ribosome biogenesis", "cholesterol biosynthesis"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
for i, pr in enumerate(progs):
    rat = [v[f"{pr}|m50"]["ratio"] for v in R.values() if f"{pr}|m50" in v]
    pc = [v[f"{pr}|m50"]["pil_cov"] for v in R.values() if f"{pr}|m50" in v]
    ac = [v[f"{pr}|m50"]["ad_cov"] for v in R.values() if f"{pr}|m50" in v]
    jit = np.linspace(-0.12, 0.12, len(rat))
    ax[0].scatter(np.full(len(rat), i) + jit, rat, s=12, c="C0")
    ax[0].hlines(np.median(rat), i - 0.25, i + 0.25, colors="k", lw=1.2)
    ax[1].scatter(np.full(len(pc), i) - 0.15 + jit / 2, pc, s=10, c="C2", label="pilot-only" if i == 0 else None)
    ax[1].scatter(np.full(len(ac), i) + 0.15 + jit / 2, ac, s=10, c="C0", label="adapted" if i == 0 else None)
for a in ax:
    a.set_xticks(range(len(progs)))
    a.set_xticklabels(["cell cycle", "p53", "unfolded\nprotein resp.", "ribosome\nbiogenesis", "cholesterol\nsynthesis"], fontsize=7)
ax[0].axhline(1, ls="--", c="gray", lw=.8); ax[0].set_ylabel("interval width,\nadapted / pilot-only"); ax[0].set_title("(a) width at m = 50 (bars: median over pairs)")
ax[1].axhline(0.9, ls="--", c="gray", lw=.8); ax[1].set_ylabel("empirical coverage"); ax[1].set_title("(b) coverage at m = 50"); ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig4_pathway.pdf"); fig.savefig(FIG / "fig4_pathway.png", dpi=200)
print("ok")
