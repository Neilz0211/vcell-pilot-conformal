"""Reviewer-requested decomposition (GPT review, 2026-10-02), all 12 directed cross-line pairs + 2 same-line controls.

Per pair, seed and model (mean; ridge alpha 1/10/100/1000 = a same-family regularisation path; kNN20):
  src_rmse / tgt_rmse   mean per-perturbation RMSE on test targets in A and in B (point accuracy)
  qA / qB               90 % split-CP radius from A-calibration / B-calibration scores (qB is post-hoc only)
  ratio                 qB / qA (calibration correction each model would need)
  covB_src              coverage in B of the source-calibrated set
  cov_at_common         coverage in B at a COMMON radius (the mean model's qA) -> separates radius shrinkage from
                        centre degradation
Per pair and seed (ridge10) also: pilot-only and exact adapted (m+1)/m ball coverage + radius at m = 20/50, kept per
seed so the spread over pilot draws can be reported.
Writes results/decomp_metrics.json (per-seed arrays) and prints pair summaries + drop-RPE1 sensitivity.
usage: uv run python src/decomp.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import fit_adapt, loo_radius  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def models(Z, YA, tr):
    M = {"mean": (lambda P: np.tile(YA[tr].mean(0), (len(P), 1)))}
    for al in (1, 10, 100, 1000):
        r = Ridge(alpha=al).fit(Z[tr], YA[tr]); M[f"ridge_a{al}"] = (lambda P, r=r: r.predict(P))
    nn = NearestNeighbors(n_neighbors=20).fit(Z[tr]); Ytr = YA[tr]
    M["knn20"] = (lambda P: Ytr[nn.kneighbors(P)[1]].mean(1))
    return M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    pairs = [(x, y) for x in lines for y in lines if x != y] + [("ess", "gwps"), ("gwps", "ess")]
    go = go_annotations(); cache = {}; res = {}
    for A, B in pairs:
        YA, YB, targets = load_pair(A, B, compact, cache)
        Z, _, _, has_go = go_embedding(targets, go)
        YA, YB, Z = YA[has_go], YB[has_go], Z[has_go]; N = len(Z)
        rec = {}
        for s in range(a.seeds):
            rng = np.random.default_rng(s); r = rng.permutation(N)
            tr, ca, te_all = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
            te = te_all[100:]
            M = models(Z, YA, tr)
            q_common = None
            for name, f in M.items():
                Pca, Pte = f(Z[ca]), f(Z[te])
                qA = split_cp_q(rmse(Pca, YA[ca]), alpha); qB = split_cp_q(rmse(Pca, YB[ca]), alpha)
                sB = rmse(Pte, YB[te])
                if name == "mean":
                    q_common = qA
                d = rec.setdefault(name, {})
                for k, v in (("src_rmse", rmse(Pte, YA[te]).mean()), ("tgt_rmse", sB.mean()), ("qA", qA), ("qB", qB),
                             ("ratio", qB / qA), ("covB_src", (sB <= qA).mean()), ("cov_at_common", (sB <= q_common).mean())):
                    d.setdefault(k, []).append(float(v))
            f = M["ridge_a10"]; Pte = f(Z[te])
            for m in (20, 50):
                pil = te_all[:m]; Pp = f(Z[pil])
                q0 = split_cp_q(rmse(Pp, YB[pil]), alpha)
                R = (m + 1) / m * loo_radius("offset", Pp, YB[pil], alpha)
                st1 = rmse(fit_adapt("offset", Pp, YB[pil])(Pte), YB[te])
                d = rec.setdefault(f"pilots_m{m}", {})
                for k, v in (("pil_cov", (rmse(Pte, YB[te]) <= q0).mean()), ("pil_q", q0), ("ad_cov", (st1 <= R).mean()), ("ad_q", R)):
                    d.setdefault(k, []).append(float(v))
        key = f"{A}->{B}"; res[key] = rec
        mm = {n: {k: np.mean(v) for k, v in d.items()} for n, d in rec.items()}
        print(f"== {key} (N={N})", flush=True)
        for n in ("mean", "ridge_a1000", "ridge_a100", "ridge_a10", "ridge_a1", "knn20"):
            d = mm[n]
            print(f"  {n:12s} srcRMSE {d['src_rmse']:.3f} tgtRMSE {d['tgt_rmse']:.3f} qA {d['qA']:.3f} qB {d['qB']:.3f} qB/qA {d['ratio']:.3f} "
                  f"covB {d['covB_src']:.3f} cov@commonR {d['cov_at_common']:.3f}", flush=True)
        for m in (20, 50):
            d = rec[f"pilots_m{m}"]
            print(f"  pilots m={m}: pilot cov {np.mean(d['pil_cov']):.3f} [p5 {np.percentile(d['pil_cov'], 5):.3f}, p95 {np.percentile(d['pil_cov'], 95):.3f}] "
                  f"radius sd {np.std(d['pil_q']):.3f} | adapted cov {np.mean(d['ad_cov']):.3f} [p5 {np.percentile(d['ad_cov'], 5):.3f}, p95 {np.percentile(d['ad_cov'], 95):.3f}] "
                  f"radius sd {np.std(d['ad_q']):.3f}", flush=True)
    # sensitivity: drop RPE1
    cross = [k for k in res if "gwps" not in k]
    for label, keys in (("all 12", cross), ("no RPE1 (6)", [k for k in cross if "rpe1" not in k])):
        n_inv = sum(np.mean(res[k]["ridge_a10"]["covB_src"]) < np.mean(res[k]["mean"]["covB_src"]) for k in keys)
        rho = []
        for k in keys:
            qa = [np.mean(res[k][n]["qA"]) for n in ("mean", "ridge_a1000", "ridge_a100", "ridge_a10", "ridge_a1", "knn20")]
            cb = [np.mean(res[k][n]["covB_src"]) for n in ("mean", "ridge_a1000", "ridge_a100", "ridge_a10", "ridge_a1", "knn20")]
            rho.append(spearmanr(qa, cb).correlation)
        print(f"SENS {label}: ridge10 covers less than mean in {n_inv}/{len(keys)}; per-pair Spearman(qA, covB) over 6 models: "
              f"median {np.median(rho):.2f}, min {np.min(rho):.2f}", flush=True)
    (OUT / "decomp_metrics.json").write_text(json.dumps(res))
    print("DECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
