"""Figure 1 (schematic, no data): the three calibration options and what limits the repaired sets.
(a) A set calibrated in the source line; (b) the same set reused in a new line whose responses are shifted;
(c) random pilots in the new line used to remove the mean offset and to calibrate (Prop. A);
(d) relative radii after repair (ranges quoted from Section 5.3 / Fig. 4, drawn as a cartoon).
usage: uv run python src/fig_schematic.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

FIG = Path(__file__).resolve().parents[2] / "paper" / "figs"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
rng = np.random.default_rng(0)
A, B, P, G = "0.55", "C1", "C1", "C0"   # source points, target points, pilots, sets
f = np.array([0.0, 0.0]); d = np.array([0.6, 0.3])
src = rng.normal(size=(60, 2)) * 0.62
tgt = rng.normal(size=(60, 2)) * 0.62 + d
q = np.quantile(np.linalg.norm(src, axis=1), 0.9)

fig = plt.figure(figsize=(6.5, 2.15))
W, H, y0 = 0.205, 0.80, 0.03
ax = [fig.add_axes([0.005 + i * 0.215, y0, W, H]) for i in range(3)] + [fig.add_axes([0.80, 0.19, 0.195, 0.64])]
for a in ax[:3]:
    a.set_xlim(-1.9, 2.6); a.set_ylim(-2.6, 3.2); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color("0.8")

a = ax[0]
a.scatter(*src.T, s=6, c=A, lw=0)
a.add_patch(Circle(f, q, fill=False, ec=G, lw=1.6)); a.plot(*f, "k*", ms=7)
a.text(0.35, -2.45, "covers 90%", ha="center", va="bottom", fontsize=10, color=G)
a.set_title("(a) calibrate in A")

a = ax[1]
inside = np.linalg.norm(tgt - f, axis=1) <= q
a.scatter(*tgt[inside].T, s=6, c=B, lw=0); a.scatter(*tgt[~inside].T, s=9, facecolors="none", edgecolors=B, lw=0.7)
a.add_patch(Circle(f, q, fill=False, ec=G, lw=1.6, ls="--")); a.plot(*f, "k*", ms=7)
a.add_patch(FancyArrowPatch(f, d, arrowstyle="-|>", mutation_scale=9, color="k", lw=1))
a.text(-0.75, 0.55, r"shift $d$", fontsize=10, ha="center")
a.text(0.35, -2.45, "covers 0.79–0.89", ha="center", va="bottom", fontsize=10, color=G)
a.set_title("(b) reuse in B")

a = ax[2]
a.scatter(*tgt.T, s=5, c=B, lw=0, alpha=0.25)
pil = tgt[rng.choice(len(tgt), 12, replace=False)]
a.scatter(*pil.T, s=16, c=P, edgecolors="k", lw=0.5, zorder=4)
ebar = pil.mean(0)
r = np.quantile(np.linalg.norm(tgt - ebar, axis=1), 0.9) * 1.04
a.plot(*f, "k*", ms=7)
a.add_patch(FancyArrowPatch(f, ebar, arrowstyle="-|>", mutation_scale=9, color="k", lw=1))
a.text(*(ebar / 2 + np.array([-0.25, -0.8])), r"$\bar e$", fontsize=10, ha="center")
a.add_patch(Circle(ebar, r, fill=False, ec=G, lw=1.6))
a.text(0.35, -2.45, r"valid, $\geq 1-\alpha$", ha="center", va="bottom", fontsize=10, color=G)
a.set_title("(c) $m$ pilots in B")

a = ax[3]
rows = [("pilot-only", 1.10, "C2"), ("adapted\npilot mean", 1.0, "C1"), ("adapted\nsource model", 0.95, G),
        ("noise floor\n(iid cells)", 0.55, "0.6")]
for i, (lab, v, c) in enumerate(rows):
    a.barh(-1.25 * i, v, color=c, height=0.75)
    a.text(-0.04, -1.25 * i, lab, ha="right", va="center", fontsize=10)
a.text(0.98, -2.5, r"$-{\leq}8\%$", va="center", fontsize=10)
a.text(0.58, -3.75, "0.45–0.65×", va="center", fontsize=10)
a.set_xlim(0, 1.6); a.set_ylim(-4.4, 0.6); a.set_yticks([]); a.set_xticks([])
for s in ("top", "right", "left"):
    a.spines[s].set_visible(False)
a.set_xlabel("radius (illustrative)", fontsize=10)
a.set_title("(d) what limits the sets", loc="right")

fig.savefig(FIG / "fig0_schematic.pdf"); fig.savefig(FIG / "fig0_schematic.png", dpi=220)
print("wrote", FIG / "fig0_schematic.pdf")
