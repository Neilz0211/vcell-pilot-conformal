"""Context shift: calibrate in one cell line, deploy in another (Replogle 2022 K562 essential <-> RPE1 pseudobulk).

Shared targets (one perturbation per target gene, first occurrence) and shared finite readout genes; readout = the
1000 genes with the largest mean variance across both lines, standardised per gene within each line.
Per seed, targets are split into train (40 %), calibration (30 %) and test (30 %) by target gene.
Models predict the response of target p in the DEPLOYMENT line B:
  ridge_src   ridge on z (GO SVD) trained on line-A responses of the training targets (a 'virtual cell' fit in A)
  mean_src    the mean line-A training response (predict-the-mean)
  transfer    the measured line-A response of p itself (cross-context transfer; p is unseen in B)
Calibration options at alpha = 0.1:
  src_cal     split CP with line-A scores of the calibration targets (exchangeability across lines assumed)
  pilot       split CP with m line-B pilot scores (valid; infinite if m < 1/alpha - 1)
  test        binomial test of src_cal coverage on the pilots (level 0.05) -> pilot if rejected
  oracle      split CP on line-B calibration targets (needs a full screen in B; reference only)
For `transfer` there is no line-A score; src_cal uses the A<->A analogue (score of A response vs A response is 0),
so only pilot/test/oracle are reported for it.
usage: uv run python src/context.py [--seeds 30]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DATA, go_annotations, go_embedding, target_gene  # noqa: E402
from day1 import fit_models, rmse, split_cp_q  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
FILES = {"K562": "K562_essential_normalized_bulk_01.h5ad", "RPE1": "rpe1_normalized_bulk_01.h5ad"}


def load_pair(n_genes=1000):
    mats, tg = {}, {}
    for k, f in FILES.items():
        a = ad.read_h5ad(DATA / f)
        names = np.asarray(a.obs_names)
        keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
        X = np.asarray(a.X, dtype=np.float64)[keep]
        t = np.array([target_gene(n) for n in names[keep]])
        _, first = np.unique(t, return_index=True)
        first = np.sort(first)
        mats[k] = (X[first], t[first], np.asarray(a.var_names))
    common_t = sorted(set(mats["K562"][1]) & set(mats["RPE1"][1]))
    common_g = sorted(set(mats["K562"][2]) & set(mats["RPE1"][2]))
    Ys = {}
    for k, (X, t, g) in mats.items():
        ti = {x: i for i, x in enumerate(t)}; gi = {x: i for i, x in enumerate(g)}
        Ys[k] = X[[ti[x] for x in common_t]][:, [gi[x] for x in common_g]]
    fin = np.isfinite(Ys["K562"]).all(0) & np.isfinite(Ys["RPE1"]).all(0)
    v = Ys["K562"][:, fin].var(0) + Ys["RPE1"][:, fin].var(0)
    top = np.where(fin)[0][np.argsort(v)[::-1][:n_genes]]
    out = {k: Y[:, top] / (Y[:, top].std(0) + 1e-8) for k, Y in Ys.items()}
    return out, np.array(common_t)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    Ys, targets = load_pair()
    Z, M, vocab, has_go = go_embedding(targets, go_annotations())
    Ys = {k: v[has_go] for k, v in Ys.items()}; Z = Z[has_go]; N = len(Z)
    cc = float(np.mean([np.corrcoef(Ys["K562"][i], Ys["RPE1"][i])[0, 1] for i in range(N)]))
    print(f"shared targets with GO {N}; readout {Ys['K562'].shape[1]}; mean per-target K562-RPE1 response corr {cc:.3f}", flush=True)
    res = {"N": int(N), "mean_cross_corr": cc, "alpha": alpha, "cases": {}}
    for A, B in (("K562", "RPE1"), ("RPE1", "K562")):
        for m in (10, 20, 50):
            rec = {}
            def put(k, v):  # noqa: E306
                rec.setdefault(k, []).append(float(v))
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                r = rng.permutation(N)
                tr, ca, te_all = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
                pil, te = te_all[:m], te_all[m:]
                fm = fit_models(Z[tr], Ys[A][tr], s)
                preds = {"ridge_src": fm["ridge"], "mean_src": fm["mean"], "transfer": lambda idx: Ys[A][idx]}
                for name, f in preds.items():
                    if name == "transfer":
                        pB = lambda idx: Ys[A][idx]  # noqa: E731
                        sA = None
                    else:
                        pB = lambda idx, f=f: f(Z[idx])  # noqa: E731
                        sA = rmse(pB(ca), Ys[A][ca])
                    sB_ca, sB_pil, sB_te = rmse(pB(ca), Ys[B][ca]), rmse(pB(pil), Ys[B][pil]), rmse(pB(te), Ys[B][te])
                    qo = split_cp_q(sB_ca, alpha); put(f"{name}_cov_oracle", (sB_te <= qo).mean()); put(f"{name}_q_oracle", qo)
                    qp = split_cp_q(sB_pil, alpha) if m >= int(np.ceil(1 / alpha)) - 1 else np.inf
                    put(f"{name}_cov_pilot", (sB_te <= qp).mean()); put(f"{name}_q_pilot", qp)
                    if sA is not None:
                        qs = split_cp_q(sA, alpha); put(f"{name}_cov_src", (sB_te <= qs).mean()); put(f"{name}_q_src", qs)
                        x = int((sB_pil <= qs).sum()); rej = binom.cdf(x, m, 1 - alpha) < 0.05
                        put(f"{name}_reject", rej)
                        put(f"{name}_cov_test", (sB_te <= (qp if rej else qs)).mean())
            row = {k: float(np.mean(v)) for k, v in rec.items()}
            res["cases"][f"{A}->{B}_m{m}"] = row
            print(f"{A}->{B} m={m}", json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
    (OUT / "context_metrics.json").write_text(json.dumps(res, indent=1))
    print("CONTEXT_DONE", flush=True)


if __name__ == "__main__":
    main()
