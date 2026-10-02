"""Equivalent pilot count (GPT review): at the same nominal coverage, how many pilots does pilot-only split conformal
(unadapted ridge) need to reach the radius the adapted ball reaches with m_a pilots?

Per directed cross-line pair and seed: screening pool = the 30 % test split (~590-690 targets); pilots are the first
m of a random order, the evaluation set is the last 280 targets (fixed across m). Pilot-only radius at
m in {10,20,30,50,75,100,150,200,300}; adapted (Prop A outer ball) at m_a in {10,20,50}; reference = pilot-only radius
using ALL non-evaluation targets (the unadapted model's best attainable radius, i.e. full-screen calibration).
Equivalent count m_eq(m_a) = smallest m on the grid where the seed-averaged pilot-only radius <= the seed-averaged
adapted radius at m_a; "> 300" if never on the grid, and "unreachable" if even the full-screen radius is larger.
Also reports the seed-to-seed standard deviation of the radius (stability) and mean coverage.
usage: uv run python src/eq_pilots.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import fit_adapt, loo_radius  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MP = (10, 20, 30, 50, 75, 100, 150, 200, 300)
MA = (10, 20, 50)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    pairs = [(x, y) for x in lines for y in lines if x != y]
    go = go_annotations(); cache = {}; res = {}
    for A, B in pairs:
        YA, YB, targets = load_pair(A, B, compact, cache)
        Z, _, _, has_go = go_embedding(targets, go)
        YA, YB, Z = YA[has_go], YB[has_go], Z[has_go]; N = len(Z)
        rec = {}
        for s in range(a.seeds):
            rng = np.random.default_rng(s); r = rng.permutation(N)
            tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]
            ev, cand = pool[-280:], pool[:-280]
            f = Ridge(alpha=10.0).fit(Z[tr], YA[tr]); Pev = f.predict(Z[ev]); s_ev = rmse(Pev, YB[ev])
            put = lambda k, v: rec.setdefault(k, []).append(float(v))  # noqa: E731
            qfull = split_cp_q(rmse(f.predict(Z[cand]), YB[cand]), alpha); put("full_q", qfull); put("full_cov", (s_ev <= qfull).mean())
            for m in MP:
                if m > len(cand):
                    continue
                pil = cand[:m]; q = split_cp_q(rmse(f.predict(Z[pil]), YB[pil]), alpha)
                put(f"pil_q_{m}", q); put(f"pil_cov_{m}", (s_ev <= q).mean())
            for m in MA:
                pil = cand[:m]; Pp = f.predict(Z[pil])
                R = (m + 1) / m * loo_radius("offset", Pp, YB[pil], alpha)
                st = rmse(fit_adapt("offset", Pp, YB[pil])(Pev), YB[ev])
                put(f"ad_q_{m}", R); put(f"ad_cov_{m}", (st <= R).mean())
        mean = {k: float(np.mean(v)) for k, v in rec.items()}; sd = {k: float(np.std(v)) for k, v in rec.items()}
        eq = {}
        for m in MA:
            target = mean[f"ad_q_{m}"]
            hit = [mp for mp in MP if f"pil_q_{mp}" in mean and mean[f"pil_q_{mp}"] <= target]
            eq[m] = hit[0] if hit else ("unreachable" if mean["full_q"] > target else ">300")
        res[f"{A}->{B}"] = {"mean": mean, "sd": sd, "m_eq": eq, "n_cand": int(len(cand))}
        print(f"{A}->{B}: full-screen q {mean['full_q']:.3f} | adapted q m10/20/50 {mean['ad_q_10']:.3f}/{mean['ad_q_20']:.3f}/{mean['ad_q_50']:.3f} "
              f"| pilot-only q m20/50/100/300 {mean['pil_q_20']:.3f}/{mean['pil_q_50']:.3f}/{mean['pil_q_100']:.3f}/{mean.get('pil_q_300', float('nan')):.3f} "
              f"| m_eq(10,20,50) = {eq[10]}, {eq[20]}, {eq[50]} | sd q pilot50 {sd['pil_q_50']:.3f} adapted50 {sd['ad_q_50']:.3f} "
              f"| cov pilot50 {mean['pil_cov_50']:.3f} adapted50 {mean['ad_cov_50']:.3f}", flush=True)
    (OUT / "eq_pilots_metrics.json").write_text(json.dumps(res, indent=1, default=str))
    print("EQ_DONE", flush=True)


if __name__ == "__main__":
    main()
