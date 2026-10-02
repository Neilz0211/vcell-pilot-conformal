"""Pre-registered rho-hat adapt rule (discussion_round3.md §4 item 1), written BEFORE running.

From the m pilots only: residuals e_i = Y_B - f(z_i) of the source ridge10 model; e_bar = mean; sigma2_hat =
sum_i ||e_i - e_bar||^2 / ((m-1) G); Delta2_hat = ||e_bar||^2 / G - sigma2_hat / m; rho_hat = Delta2_hat / sigma2_hat.
Rule: adapt (use the exact adapted ball, radius (m+1)/m * LOO quantile) iff rho_hat > (3m+1)/(m(m-1)); else pilot-only
split CP. Arms reported: pilot-only, adapted, rule; plus rho_hat bias vs the oracle rho (from the full calibration
split). Pre-registered success: the rule arm picks the arm with the smaller mean radius in >= 13/15 pairs (here 12 cross-line
+ 2 same-line controls = 14 pairs, so >= 12/14), and rule-arm coverage >= 0.89 on average. Kill: |rho_hat - rho| > 0.2 at
m = 20 -> report the oracle rho only.
Caveat: choosing between two valid sets with the same pilots is not covered by Prop A; coverage is measured.
usage: uv run python src/rho_rule.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import fit_adapt, loo_radius  # noqa: E402
from data import DATA, go_annotations, go_embedding, target_gene  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
H5 = {"ess": "K562_essential_normalized_bulk_01.h5ad", "rpe1": "rpe1_normalized_bulk_01.h5ad", "gwps": "K562_gwps_normalized_bulk_01.h5ad"}


def load_line(k, compact, cache):
    if k in cache:
        return cache[k]
    if k in H5:
        a = ad.read_h5ad(DATA / H5[k])
        names = np.asarray(a.obs_names)
        keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
        t = np.array([target_gene(n) for n in names[keep]])
        _, first = np.unique(t, return_index=True); first = np.sort(first)
        g = np.asarray(a.var_names); gi = {x: i for i, x in enumerate(g)}
        cols = [gi[x] for x in compact["genes"] if x in gi]
        X = np.asarray(a.X[:, cols], dtype=np.float32)[keep][first]
        out = (X, t[first], g[cols])
    else:
        out = (compact[f"{k}_X"].astype(np.float32), np.array([n.split("_")[1] for n in compact[f"{k}_obs"]]), np.asarray(compact["genes"]))
    cache[k] = out
    return out


def load_pair(A, B, compact, cache, n_genes=1000, prep="pooled"):
    """prep="pooled": readout genes by summed A+B variance, each line standardised by its own per-gene SD (uses target
    responses; the original analysis). prep="source": leak-free deployment version -- genes chosen by source variance
    only and BOTH lines scaled by the source per-gene SD, so no target response enters preprocessing (the only target
    information used is which genes are finite, i.e. measurable, in the target assay)."""
    if prep == "source":
        return _load_pair_source(A, B, compact, cache, n_genes)
    if prep == "raw":  # all finite shared genes, unscaled: the caller selects genes / scales on the source TRAIN split
        return _load_pair_source(A, B, compact, cache, None)
    (XA, tA, gA), (XB, tB, gB) = load_line(A, compact, cache), load_line(B, compact, cache)
    ct = sorted(set(tA) & set(tB)); cg = sorted(set(gA) & set(gB))
    ia = {x: i for i, x in enumerate(tA)}; ib = {x: i for i, x in enumerate(tB)}
    ga = {x: i for i, x in enumerate(gA)}; gb = {x: i for i, x in enumerate(gB)}
    YA = XA[[ia[x] for x in ct]][:, [ga[x] for x in cg]].astype(np.float64)
    YB = XB[[ib[x] for x in ct]][:, [gb[x] for x in cg]].astype(np.float64)
    fin = np.isfinite(YA).all(0) & np.isfinite(YB).all(0)
    YA, YB = YA[:, fin], YB[:, fin]
    top = np.argsort(YA.var(0) + YB.var(0))[::-1][:n_genes]
    YA, YB = YA[:, top], YB[:, top]
    return YA / (YA.std(0) + 1e-8), YB / (YB.std(0) + 1e-8), np.array(ct)


def _load_pair_source(A, B, compact, cache, n_genes):
    (XA, tA, gA), (XB, tB, gB) = load_line(A, compact, cache), load_line(B, compact, cache)
    ct = sorted(set(tA) & set(tB)); cg = sorted(set(gA) & set(gB))
    ia = {x: i for i, x in enumerate(tA)}; ib = {x: i for i, x in enumerate(tB)}
    ga = {x: i for i, x in enumerate(gA)}; gb = {x: i for i, x in enumerate(gB)}
    YA = XA[[ia[x] for x in ct]][:, [ga[x] for x in cg]].astype(np.float64)
    YB = XB[[ib[x] for x in ct]][:, [gb[x] for x in cg]].astype(np.float64)
    fin = np.isfinite(YA).all(0) & np.isfinite(YB).all(0)
    YA, YB = YA[:, fin], YB[:, fin]
    if n_genes is None:
        return YA, YB, np.array(ct)
    top = np.argsort(YA.var(0))[::-1][:n_genes]
    YA, YB = YA[:, top], YB[:, top]
    sd = YA.std(0) + 1e-8
    return YA / sd, YB / sd, np.array(ct)


def rho_hat(E):
    m, G = E.shape; eb = E.mean(0)
    s2 = ((E - eb) ** 2).sum() / ((m - 1) * G)
    return ((eb ** 2).sum() / G - s2 / m) / s2


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
            f = Ridge(alpha=10.0).fit(Z[tr], YA[tr])
            Pca, Pte = f.predict(Z[ca]), f.predict(Z[te])
            rho_or = rho_hat(YB[ca] - Pca)  # same estimator on the large calibration split ~ oracle
            for m in (20, 50, 100):
                pil = te_all[:m]; Pp = f.predict(Z[pil]); E = YB[pil] - Pp
                q0 = split_cp_q(rmse(Pp, YB[pil]), alpha); st0 = rmse(Pte, YB[te])
                R = (m + 1) / m * loo_radius("offset", Pp, YB[pil], alpha)
                st1 = rmse(fit_adapt("offset", Pp, YB[pil])(Pte), YB[te])
                rh = rho_hat(E); adapt = rh > (3 * m + 1) / (m * (m - 1))
                d = rec.setdefault(m, {k: [] for k in ("pil_cov", "pil_q", "ad_cov", "ad_q", "rule_cov", "rule_q", "rule_adapt", "rho_hat", "rho_or")})
                d["pil_cov"].append((st0 <= q0).mean()); d["pil_q"].append(q0)
                d["ad_cov"].append((st1 <= R).mean()); d["ad_q"].append(R)
                d["rule_cov"].append((st1 <= R).mean() if adapt else (st0 <= q0).mean()); d["rule_q"].append(R if adapt else q0)
                d["rule_adapt"].append(float(adapt)); d["rho_hat"].append(rh); d["rho_or"].append(rho_or)
        key = f"{A}->{B}"; res[key] = {}
        for m, d in rec.items():
            row = {k: float(np.mean(v)) for k, v in d.items()}
            row["rho_bias"] = float(np.mean(np.array(d["rho_hat"]) - np.array(d["rho_or"])))
            row["rho_sd"] = float(np.std(d["rho_hat"]))
            better = "adapt" if row["ad_q"] < row["pil_q"] else "pilot"
            row["better_arm"] = better
            row["rule_majority"] = "adapt" if row["rule_adapt"] > 0.5 else "pilot"
            res[key][f"m{m}"] = row
            print(f"{key:14s} m={m:3d} rho_or {row['rho_or']:.3f} rho_hat {row['rho_hat']:.3f}+-{row['rho_sd']:.3f} adapt% {row['rule_adapt']:.2f} "
                  f"q pil/ad/rule {row['pil_q']:.3f}/{row['ad_q']:.3f}/{row['rule_q']:.3f} cov {row['pil_cov']:.3f}/{row['ad_cov']:.3f}/{row['rule_cov']:.3f} "
                  f"better={better} rule={row['rule_majority']}", flush=True)
    for m in ("m20", "m50", "m100"):
        ok = sum(res[k][m]["better_arm"] == res[k][m]["rule_majority"] for k in res)
        cov = np.mean([res[k][m]["rule_cov"] for k in res])
        print(f"SUMMARY {m}: rule picks the better arm in {ok}/{len(res)} pairs; mean rule coverage {cov:.3f}", flush=True)
    (OUT / "rho_rule_metrics.json").write_text(json.dumps(res, indent=1))
    print("RHO_RULE_DONE", flush=True)


if __name__ == "__main__":
    main()
