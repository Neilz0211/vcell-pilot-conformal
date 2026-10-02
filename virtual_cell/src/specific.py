"""Perturbation-specific readout (priority 1 after GPT review 2).

Decision question in the target line B: does perturbation p respond differently from the typical perturbation, i.e. is
its perturbation-specific response s_p = Y_B(p) - mu_B non-zero? mu_B = mean target response over the screening panel
(the shared / systematic component, Systema), estimated by the pilot mean in deployment.

A perturbation is *certified* when its 90 % prediction ball for Y_B(p) excludes the estimated typical response mu_hat
(pilot mean of Y_B):
  pilot-only   ball centre f(z_p), radius = split-CP quantile of pilot scores -> distance ||f(z_p) - mu_hat||
               still contains the context offset, so the call mixes shared shift with specific effect
  adapted      ball centre f(z_p) + b_hat, radius = (m+1)/m * LOO quantile (Prop A outer ball) -> the offset cancels:
               centre - mu_hat = f(z_p) - mean_pilots f
Readouts in the full 1,000-gene space (K=0) and on the top-K principal components of the SOURCE line's training
responses (K = 1, 5, 20; projection fixed before pilots are seen, so Prop A applies).
Truth (evaluation only, never used by the methods): the true specific effect ||s_p|| with mu_B from ALL target
perturbations; a perturbation is a true responder if ||s_p|| is in the top 25 % of test perturbations.
Reported per pair (12 directed cross-line pairs), m = 20/50, 20 seeds: certified fraction, precision (true responders
among certified), recall (certified among true responders), and coverage of the sets.
usage: uv run python src/specific.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import loo_radius  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
KS = (0, 1, 5, 20)


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
            Yc = YA[tr] - YA[tr].mean(0); V = np.linalg.svd(Yc, full_matrices=False)[2]
            pool = np.r_[te_all]  # the screening panel = pilots + test
            for K in KS:
                W = np.eye(YA.shape[1]) if K == 0 else V[:K].T
                YBp = YB @ W; Pall = f.predict(Z) @ W
                muB = YBp[pool].mean(0)  # truth only
                spec = np.linalg.norm(YBp[te] - muB, axis=1) / np.sqrt(W.shape[1])
                truth = spec >= np.quantile(spec, 0.75)
                for m in (20, 50):
                    pil = te_all[:m]
                    mu_hat = YBp[pil].mean(0)
                    # pilot-only
                    q0 = split_cp_q(rmse(Pall[pil], YBp[pil]), alpha)
                    d0 = np.linalg.norm(Pall[te] - mu_hat, axis=1) / np.sqrt(W.shape[1])
                    cov0 = (rmse(Pall[te], YBp[te]) <= q0).mean()
                    # adapted (Prop A outer ball)
                    b = (YBp[pil] - Pall[pil]).mean(0)
                    R = (m + 1) / m * loo_radius("offset", Pall[pil], YBp[pil], alpha)
                    d1 = np.linalg.norm(Pall[te] + b - mu_hat, axis=1) / np.sqrt(W.shape[1])
                    cov1 = (rmse(Pall[te] + b, YBp[te]) <= R).mean()
                    for name, dist, q, cov in (("pilot", d0, q0, cov0), ("adapted", d1, R, cov1)):
                        cert = dist > q
                        d = rec.setdefault(f"K{K}_{name}_m{m}", {k: [] for k in ("frac", "prec", "rec", "cov")})
                        d["frac"].append(float(cert.mean())); d["cov"].append(float(cov))
                        d["prec"].append(float(truth[cert].mean()) if cert.any() else np.nan)
                        d["rec"].append(float(cert[truth].mean()))
        with np.errstate(all="ignore"):
            res[f"{A}->{B}"] = {k: {kk: float(np.nanmean(vv)) if np.isfinite(vv).any() else None for kk, vv in d.items()} for k, d in rec.items()}
        x = res[f"{A}->{B}"]
        def fmt(k):  # noqa: E306
            v = x[k]; p = "n/a" if v["prec"] is None else f"{v['prec']:.2f}"
            return f"{v['frac']:.3f}/prec {p}/rec {v['rec']:.2f}"
        print(f"{A}->{B}: " + " | ".join(f"K{K} m50 pilot {fmt(f'K{K}_pilot_m50')} adapted {fmt(f'K{K}_adapted_m50')}" for K in (1, 5, 20)), flush=True)
    (OUT / "specific_metrics.json").write_text(json.dumps(res, indent=1))
    print("SPECIFIC_DONE", flush=True)


if __name__ == "__main__":
    main()
