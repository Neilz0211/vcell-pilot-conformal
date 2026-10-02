"""Non-RMSE scores (reviewer suggestion): conformal intervals for scalar pathway scores.

A pathway score is the mean standardised response over the readout genes annotated to a GO Biological Process term
(a fixed linear projection, chosen before pilots are seen), so Proposition A applies with |residual| as the score.
Programmes: cell cycle (GO:0007049), ribosome biogenesis (GO:0042254), response to unfolded protein (GO:0006986),
signal transduction by p53 class mediator (GO:0072331), cholesterol biosynthetic process (GO:0006695); a programme is
used for a pair only if >= 8 of its genes are among the pair's readout genes.
Per directed cross-line pair, 20 seeds, m in {20, 50}: pilot-only split CP (unadapted ridge) vs the adapted Prop A outer
ball, both on the scalar score; reports empirical coverage and the interval half-width ratio adapted/pilot-only.
Preprocessing: leak-free (genes and scale from the source TRAIN split only), as in paired.py --prep sourcetrain.
usage: uv run python src/pathway.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import fit_adapt, loo_radius  # noqa: E402
from data import DATA, go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from offset_enrich import go_bp_with_names  # noqa: E402
from rho_rule import load_line  # noqa: E402


def load_raw_with_genes(A, B, compact, cache):
    """Same as rho_rule._load_pair_source with n_genes=None, but also returns the gene ids in column order."""
    (XA, tA, gA), (XB, tB, gB) = load_line(A, compact, cache), load_line(B, compact, cache)
    ct = sorted(set(tA) & set(tB)); cg = sorted(set(gA) & set(gB))
    ia = {x: i for i, x in enumerate(tA)}; ib = {x: i for i, x in enumerate(tB)}
    ga = {x: i for i, x in enumerate(gA)}; gb = {x: i for i, x in enumerate(gB)}
    YA = XA[[ia[x] for x in ct]][:, [ga[x] for x in cg]].astype(np.float64)
    YB = XB[[ib[x] for x in ct]][:, [gb[x] for x in cg]].astype(np.float64)
    fin = np.isfinite(YA).all(0) & np.isfinite(YB).all(0)
    return YA[:, fin], YB[:, fin], np.array(ct), np.array(cg)[fin]

OUT = Path(__file__).resolve().parents[1] / "results"
PROGRAMS = {"cell cycle": "GO:0007049", "ribosome biogenesis": "GO:0042254", "unfolded protein response": "GO:0006986",
            "p53 signalling": "GO:0072331", "cholesterol biosynthesis": "GO:0006695"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    v = ad.read_h5ad(DATA / "K562_essential_normalized_bulk_01.h5ad", backed="r").var
    ens2sym = dict(zip(v.index, v["gene_name"].astype(str)))
    g2t, names = go_bp_with_names()
    go = go_annotations(); cache = {}; res = {}
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    for A in lines:
        for B in lines:
            if A == B:
                continue
            YA0, YB0, targets, genes = load_raw_with_genes(A, B, compact, cache)
            Z, _, _, hg = go_embedding(targets, go); YA0, YB0, Z = YA0[hg], YB0[hg], Z[hg]; N = len(Z)
            rec = {}
            for s in range(a.seeds):
                rng = np.random.default_rng(s); r = rng.permutation(N)
                tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]; test, cal = pool[-280:], pool[:-280]
                top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
                YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
                syms = [ens2sym.get(g, g) for g in genes[top]]
                f = Ridge(alpha=10.0).fit(Z[tr], YA[tr]); P = f.predict(Z)
                for pname, term in PROGRAMS.items():
                    mask = np.array([term in g2t.get(sy, ()) for sy in syms])
                    if mask.sum() < 8:
                        continue
                    w = mask / mask.sum()
                    yb, p = (YB @ w)[:, None], (P @ w)[:, None]
                    for m in (20, 50):
                        pil = cal[:m]
                        q0 = split_cp_q(rmse(p[pil], yb[pil]), alpha)
                        R = (m + 1) / m * loo_radius("offset", p[pil], yb[pil], alpha)
                        st = rmse(fit_adapt("offset", p[pil], yb[pil])(p[test]), yb[test])
                        d = rec.setdefault(f"{pname}|m{m}", {"pil_cov": [], "ad_cov": [], "ratio": [], "n_genes": []})
                        d["pil_cov"].append(float((rmse(p[test], yb[test]) <= q0).mean())); d["ad_cov"].append(float((st <= R).mean()))
                        d["ratio"].append(float(R / q0)); d["n_genes"].append(int(mask.sum()))
            res[f"{A}->{B}"] = {k: {kk: float(np.mean(vv)) for kk, vv in d.items()} for k, d in rec.items()}
            x = res[f"{A}->{B}"]
            print(f"{A}->{B}: " + " | ".join(f"{k.split('|')[0][:12]} m50 cov {x[k]['pil_cov']:.3f}/{x[k]['ad_cov']:.3f} width x{x[k]['ratio']:.2f}"
                                          for k in x if k.endswith("m50")), flush=True)
    (OUT / "pathway_metrics.json").write_text(json.dumps(res, indent=1))
    print("PATHWAY_DONE", flush=True)


if __name__ == "__main__":
    main()
