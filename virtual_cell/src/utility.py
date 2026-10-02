"""What does a tighter set buy? (GPT review 2, item 3)

Downstream readout: a perturbation is a *certified responder* when its 90 % prediction ball in the target line excludes
the null response (the zero vector = no change vs controls), i.e. RMSE(prediction, 0) > radius. Compared at m = 20/50
pilots: pilot-only split CP (unadapted ridge) vs the exact adapted ball (offset-adapted ridge, (m+1)/m LOO radius).
Reported: fraction of test perturbations certified, and the precision of the call (fraction of certified ones whose
OBSERVED target response is also non-null, |Y| above the 50th percentile of observed norms -- a crude truth proxy),
for all 12 directed cross-line pairs.
usage: uv run python src/utility.py [--seeds 20]"""
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
KS = (0, 1, 5)  # 0 = all 1,000 genes; K = top-K source-line principal components


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
            tr, te_all = r[: int(0.4 * N)], r[int(0.7 * N):]
            te = te_all[100:]
            f = Ridge(alpha=10.0).fit(Z[tr], YA[tr])
            for K in KS:
              if K == 0:
                  W = np.eye(YA.shape[1])
              else:
                  Yc = YA[tr] - YA[tr].mean(0); W = np.linalg.svd(Yc, full_matrices=False)[2][:K].T  # source-line PCs
              YBp = YB @ W; pred = lambda idx: f.predict(Z[idx]) @ W  # noqa: E731
              Pte = pred(te)
              ynorm = np.linalg.norm(YBp[te], axis=1); truth = ynorm > np.median(ynorm)
              for m in (20, 50):
                pil = te_all[:m]; Pp = pred(pil)
                q0 = split_cp_q(rmse(Pp, YBp[pil]), alpha)
                R = (m + 1) / m * loo_radius("offset", Pp, YBp[pil], alpha)
                Pad = fit_adapt("offset", Pp, YBp[pil])(Pte)
                for name, P, q in (("pilot", Pte, q0), ("adapted", Pad, R)):
                    cert = (np.linalg.norm(P, axis=1) / np.sqrt(P.shape[1])) > q
                    cov = float((rmse(P, YBp[te]) <= q).mean())
                    d = rec.setdefault(f"K{K}_{name}_m{m}", {"frac": [], "prec": [], "q": [], "cov": []}); d["cov"].append(cov)
                    d["frac"].append(float(cert.mean())); d["q"].append(float(q))
                    d["prec"].append(float(truth[cert].mean()) if cert.any() else np.nan)
        res[f"{A}->{B}"] = {k: {kk: float(np.nanmean(vv)) for kk, vv in d.items()} for k, d in rec.items()}
        x = res[f"{A}->{B}"]
        print(f"{A}->{B}: " + " | ".join(f"K{K} m50 cert pilot {x[f'K{K}_pilot_m50']['frac']:.3f} vs adapted {x[f'K{K}_adapted_m50']['frac']:.3f} "
              f"(prec {x[f'K{K}_adapted_m50']['prec']:.2f}, cov {x[f'K{K}_pilot_m50']['cov']:.2f}/{x[f'K{K}_adapted_m50']['cov']:.2f})" for K in KS), flush=True)
    (OUT / "utility_metrics.json").write_text(json.dumps(res, indent=1))
    print("UTILITY_DONE", flush=True)


if __name__ == "__main__":
    main()
