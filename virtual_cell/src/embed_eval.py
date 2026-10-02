"""Option B: do pretrained gene embeddings make a better source model, and does the inversion / the gap change?

Features for the perturbed gene: GO-SVD (as in the paper), ESM2-650M protein embedding, Geneformer V2-104M token
embedding (data/gene_emb.npz from pod/embed_genes.py), each reduced by SVD to 64 dimensions over the pair's targets
(unsupervised, symmetric), and their concatenation. Only targets with all three features are used, so every feature set
sees the same perturbations. Leak-free design of paired.py: source train 40 % (fit; readout genes and scales), source
'mid' 30 % (source calibration), target pool 30 % = fixed 280-target test set + calibration pool (pilots = first m).
Per model (predict-the-mean; ridge lambda = 10 on each feature set): source radius qA, source-calibrated target
coverage covB_src, oracle-offset coverage, and at m = 50 the adapted radius (Prop A outer ball) and coverage; with
data/noise.npz the noise floor of the target (as src/ceiling.py, same perturbations and genes).
usage: uv run python src/embed_eval.py [--seeds 10] [--pairs ess:rpe1,...] [--out F]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ceiling import load_pair_named  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "results"


def svd64(X):
    X = X - X.mean(0); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return TruncatedSVD(64, random_state=0).fit_transform(X)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--pairs", default=""); ap.add_argument("--out", default="embed_eval_metrics.json")
    a = ap.parse_args(); al = a.alpha
    only = set(a.pairs.split(",")) if a.pairs else None
    compact = dict(np.load(ROOT / "data" / "nadig_compact.npz", allow_pickle=True))
    EM = np.load(ROOT / "data" / "gene_emb.npz", allow_pickle=True); esym = {s: i for i, s in enumerate(EM["symbols"])}
    NZ = np.load(ROOT / "data" / "noise.npz", allow_pickle=True)
    g3 = {g: i for i, g in enumerate([str(x) for x in compact["genes"]])}
    lines = ("ess", "rpe1", "hepg2", "jurkat"); go = go_annotations(); cache = {}; res = {}
    feats_order = ["go", "esm2", "geneformer", "all"]
    for A in lines:
        for B in lines:
            if A == B or (only and f"{A}:{B}" not in only): continue
            YA0, YB0, targets, gnames = load_pair_named(A, B, compact, cache)
            Zgo, _, _, has = go_embedding(targets, go)
            ei = np.array([esym.get(t, -1) for t in targets])
            ok = has & (ei >= 0)
            ok &= np.array([np.isfinite(EM["esm2"][i]).all() and np.isfinite(EM["geneformer"][i]).all() if i >= 0 else False for i in ei])
            YA0, YB0, targets, ei = YA0[ok], YB0[ok], targets[ok], ei[ok]; N = len(targets)
            Zgo = go_embedding(targets, go)[0]
            F = {"go": Zgo, "esm2": svd64(EM["esm2"][ei]), "geneformer": svd64(EM["geneformer"][ei])}
            F["all"] = np.hstack([F["go"], F["esm2"], F["geneformer"]])
            pz = {p: i for i, p in enumerate(NZ[f"{B}_perts"])}; EPS = NZ[f"{B}_eps"].astype(np.float64)
            prow = np.array([pz.get(t, -1) for t in targets]); gcol = np.array([g3.get(g, -1) for g in gnames])
            raw = {}; put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
            for s in range(a.seeds):
                r = np.random.default_rng(s).permutation(N)
                tr, mid, pool = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
                test, cal = pool[-280:], pool[:-280]
                top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
                YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
                eps = np.full_like(YB, np.nan); okr = prow >= 0; okc = gcol[top] >= 0
                eps[np.ix_(okr, okc)] = EPS[np.ix_(prow[okr], gcol[top][okc])] / sd[okc]; eps[~np.isfinite(eps)] = np.nan
                usable = np.isfinite(eps).mean(1) >= 0.95
                gm = np.isfinite(eps[usable]).mean(0) >= 0.99
                YA, YB, eps = YA[:, gm], YB[:, gm], np.where(np.isnan(eps[:, gm]), 0.0, eps[:, gm])
                calu = cal[usable[cal]]; testu = test[usable[test]]
                put("floor", split_cp_q(np.sqrt((eps[calu] ** 2).mean(1)), al))
                preds = {"mean": np.tile(YA[tr].mean(0), (N, 1))}
                for fn in feats_order:
                    preds[f"ridge_{fn}"] = Ridge(alpha=10.0).fit(F[fn][tr], YA[tr]).predict(F[fn])
                delta = YB[mid].mean(0) - YA[mid].mean(0)
                for nm, P in preds.items():
                    qA = split_cp_q(rmse(P[mid], YA[mid]), al); st = rmse(P[testu], YB[testu])
                    put(f"{nm}_qA", qA); put(f"{nm}_covB", (st <= qA).mean())
                    put(f"{nm}_covB_off", (rmse(P[testu] + delta, YB[testu]) <= qA).mean())
                    put(f"{nm}_covA", (rmse(P[testu], YA[testu]) <= qA).mean())
                    m = 50; pil = calu[:m]; E = YB[pil] - P[pil]; b = E.mean(0); bl = (E.sum(0)[None] - E) / (m - 1)
                    R = (m + 1) / m * split_cp_q(np.sqrt(((E - bl) ** 2).mean(1)), al)
                    put(f"{nm}_adq", R); put(f"{nm}_adcov", (rmse(P[testu] + b, YB[testu]) <= R).mean())
                    put(f"{nm}_qBor", split_cp_q(rmse(P[calu] + (YB[calu] - P[calu]).mean(0), YB[calu]), al))
            x = {k: float(np.mean(v)) for k, v in raw.items()}; x["N"] = N
            names = ["mean"] + [f"ridge_{f}" for f in feats_order]
            x["inv_spearman"] = float(spearmanr([x[f"{n}_qA"] for n in names], [x[f"{n}_covB"] for n in names]).statistic)
            res[f"{A}->{B}"] = x
            print(f"{A}->{B} N={N} | qA " + " ".join(f"{n.replace('ridge_', '')}={x[f'{n}_qA']:.3f}" for n in names)
                  + " | covB " + " ".join(f"{x[f'{n}_covB']:.3f}" for n in names)
                  + f" | inv rho {x['inv_spearman']:.2f} | adq " + " ".join(f"{x[f'{n}_adq']:.3f}" for n in names)
                  + f" floor {x['floor']:.3f}", flush=True)
    (OUT / a.out).write_text(json.dumps(res, indent=1))
    print("EMBED_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
