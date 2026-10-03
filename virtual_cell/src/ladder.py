"""E2' model ladder x context (discussion_round1.md): does source efficiency transfer, and is the more source-efficient
model the more overconfident in a new cell line?

Context pair K562-essential <-> RPE1 (context.load_pair). Models, all fit on line-A responses of training targets:
  mean; ridge(GO z, alpha in 1/10/100/1000); kNN(GO z, k in 5/20/50); MLP(GO z);
  ridge(F) and kNN(F) where F = a functional embedding of the target gene learned in K562 from the genome-wide screen
  (gwps): PCA-64 of the gene's readout column across all gwps perturbations (how the gene itself moves when other
  genes are knocked down) -- the strong linear-baseline recipe; only targets measured as gwps readout genes are kept.
  For RPE1 -> K562 the F embedding is still K562-derived (it is a property of the gene, known before the experiment).
Per model and direction: source radius q_A (split CP on A cal scores), oracle target radius q_B (split CP on B cal
scores), coverage in B of the source-calibrated set, and the efficiency-transfer ratio
  (q_B(mean) - q_B(model)) / (q_A(mean) - q_A(model)).
usage: uv run python src/ladder.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context import load_pair  # noqa: E402
from data import DATA, go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def gwps_gene_embedding(genes, dim=64):
    a = ad.read_h5ad(DATA / "K562_gwps_normalized_bulk_01.h5ad")
    names = np.asarray(a.obs_names)
    keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
    X = np.asarray(a.X, dtype=np.float32)[keep]
    sym = np.asarray(a.var["gene_name"]).astype(str)
    del a
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    idx = {g: i for i, g in enumerate(sym)}
    have = np.array([g in idx for g in genes])
    cols = X[:, [idx[g] for g in genes[have]]].T  # genes x perturbations
    del X
    cols = np.clip(cols, -10, 10)
    E = PCA(dim, random_state=0).fit_transform(cols - cols.mean(0))
    E = E / (E.std(0) + 1e-8)
    return E, have


def knn_fit(Z, Y, k):
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    return lambda Zq: Y[nn.kneighbors(Zq)[1]].mean(1)


def models(Zg, Fe, Ytr, tr, s):
    M = {"mean": lambda idx: np.tile(Ytr.mean(0), (len(idx), 1))}
    for al in (1, 10, 100, 1000):
        r = Ridge(alpha=al).fit(Zg[tr], Ytr); M[f"ridgeGO_a{al}"] = lambda idx, r=r: r.predict(Zg[idx])
    for k in (5, 20, 50):
        f = knn_fit(Zg[tr], Ytr, k); M[f"knnGO_k{k}"] = lambda idx, f=f: f(Zg[idx])
    mlp = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=300, early_stopping=True, random_state=s).fit(Zg[tr], Ytr)
    M["mlpGO"] = lambda idx: mlp.predict(Zg[idx])
    for al in (10, 100, 1000):
        r = Ridge(alpha=al).fit(Fe[tr], Ytr); M[f"ridgeF_a{al}"] = lambda idx, r=r: r.predict(Fe[idx])
    for k in (10, 30):
        f = knn_fit(Fe[tr], Ytr, k); M[f"knnF_k{k}"] = lambda idx, f=f: f(Fe[idx])
    return M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    Ys, targets = load_pair()
    Zg, _, _, has_go = go_embedding(targets, go_annotations())
    Fe_all, have_f = gwps_gene_embedding(targets)
    keep = has_go & have_f
    Fe = np.zeros((len(targets), Fe_all.shape[1])); Fe[have_f] = Fe_all
    Ys = {k: v[keep] for k, v in Ys.items()}; Zg, Fe = Zg[keep], Fe[keep]; N = len(Zg)
    print(f"targets with GO and a gwps embedding: {N}", flush=True)
    res = {"N": int(N)}
    for A, B in (("K562", "RPE1"), ("RPE1", "K562")):
        rec = {}
        for s in range(a.seeds):
            r = np.random.default_rng(s).permutation(N)
            tr, ca, te = r[: int(0.5 * N)], r[int(0.5 * N): int(0.75 * N)], r[int(0.75 * N):]
            for name, f in models(Zg, Fe, Ys[A][tr], tr, s).items():
                pc, pt = f(ca), f(te)
                qA = split_cp_q(rmse(pc, Ys[A][ca]), alpha); qB = split_cp_q(rmse(pc, Ys[B][ca]), alpha)
                stB = rmse(pt, Ys[B][te]); stA = rmse(pt, Ys[A][te])
                d = rec.setdefault(name, {"qA": [], "qB": [], "covB_src": [], "covA": []})
                d["qA"].append(qA); d["qB"].append(qB); d["covB_src"].append(float((stB <= qA).mean())); d["covA"].append(float((stA <= qA).mean()))
        agg = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in rec.items()}
        m0 = agg["mean"]
        for n, d in agg.items():
            gA = m0["qA"] - d["qA"]
            d["gainA_pct"] = 100 * gA / m0["qA"]; d["gainB_pct"] = 100 * (m0["qB"] - d["qB"]) / m0["qB"]
            d["transfer_ratio"] = (m0["qB"] - d["qB"]) / gA if abs(gA) > 1e-3 else None
        res[f"{A}->{B}"] = agg
        print(f"== {A}->{B}", flush=True)
        for n, d in sorted(agg.items(), key=lambda kv: kv[1]["qA"]):
            print(f"  {n:14s} qA {d['qA']:.3f} gainA {d['gainA_pct']:5.1f}%  qB {d['qB']:.3f} gainB {d['gainB_pct']:5.1f}%  covA {d['covA']:.3f}  covB_src {d['covB_src']:.3f}", flush=True)
        qa = np.array([d["qA"] for d in agg.values()]); cb = np.array([d["covB_src"] for d in agg.values()])
        res[f"{A}->{B}_spearman_qA_covB"] = float(spearmanr(qa, cb)[0])  # tie-aware (average ranks); see ladder_spearman.py
        print("  spearman(qA, covB_src) =", round(res[f"{A}->{B}_spearman_qA_covB"], 3), flush=True)
    (OUT / "ladder_metrics.json").write_text(json.dumps(res, indent=1))
    print("LADDER_DONE", flush=True)


if __name__ == "__main__":
    main()
