"""Audit of src/specific.py (owner asked: is the negative perturbation-specific result real or a bug?).

Same certification rule and metrics as specific.py (adapted Prop A outer ball, pilot mean as typical response, truth =
top 25 % of true specific effects), K in {1, 5}, m = 50, 10 seeds, applied to controls:
  oracle_s     POSITIVE CONTROL: predictor = true target response + Gaussian noise (sd s per gene) -- if the readout
               works, informative predictions must certify many perturbations with high precision
  insource     same-line control: train, calibrate and test within one line (no transfer) with ridge on GO features
  transfer     the real setting (ridge trained in the source line), for reference
Also prints, per setting, the median predicted specific-effect size ||centre - mu_hat|| against the median radius.
usage: uv run python src/audit_specific.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import loo_radius  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from rho_rule import load_pair  # noqa: E402

ALPHA = 0.1


def run(YA, YB, Z, predictor, K, m=50, seeds=10):
    N = len(Z); out = {"frac": [], "prec": [], "rec": [], "pred": [], "rad": []}
    for s in range(seeds):
        rng = np.random.default_rng(s); r = rng.permutation(N)
        tr, te_all = r[: int(0.4 * N)], r[int(0.7 * N):]; te = te_all[100:]
        V = np.linalg.svd(YA[tr] - YA[tr].mean(0), full_matrices=False)[2]; W = V[:K].T
        P = predictor(tr, rng) @ W; YBp = YB @ W
        muB = YBp[te_all].mean(0)
        spec = np.linalg.norm(YBp[te] - muB, axis=1) / np.sqrt(K); truth = spec >= np.quantile(spec, 0.75)
        pil = te_all[:m]; mu_hat = YBp[pil].mean(0)
        b = (YBp[pil] - P[pil]).mean(0)
        R = (m + 1) / m * loo_radius("offset", P[pil], YBp[pil], ALPHA)
        d = np.linalg.norm(P[te] + b - mu_hat, axis=1) / np.sqrt(K)
        cert = d > R
        out["frac"].append(cert.mean()); out["rec"].append(cert[truth].mean())
        out["prec"].append(truth[cert].mean() if cert.any() else np.nan)
        out["pred"].append(np.median(d)); out["rad"].append(R)
    return {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan") for k, v in out.items()}


def main():
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    go = go_annotations(); cache = {}
    for A, B in (("ess", "rpe1"), ("hepg2", "jurkat")):
        YA, YB, t = load_pair(A, B, compact, cache); Z, _, _, hg = go_embedding(t, go); YA, YB, Z = YA[hg], YB[hg], Z[hg]
        _, YBB, _ = load_pair(B, B, compact, cache) if False else (None, None, None)
        settings = {
            "transfer (ridge, A->B)": lambda tr, rng: Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z),
            "insource (ridge, B->B)": lambda tr, rng: Ridge(alpha=10.0).fit(Z[tr], YB[tr]).predict(Z),
            "oracle s=0.3": lambda tr, rng: YB + rng.normal(0, 0.3, YB.shape),
            "oracle s=0.7": lambda tr, rng: YB + rng.normal(0, 0.7, YB.shape),
            "oracle s=1.0": lambda tr, rng: YB + rng.normal(0, 1.0, YB.shape),
        }
        for K in (1, 5):
            print(f"== {A}->{B}, K={K}", flush=True)
            for name, pr in settings.items():
                x = run(YA, YB, Z, pr, K)
                print(f"  {name:24s} certified {x['frac']:.3f} precision {x['prec']:.2f} recall {x['rec']:.2f} | median predicted specific {x['pred']:.3f} vs radius {x['rad']:.3f}", flush=True)
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
