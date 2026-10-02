"""Deep virtual-cell model in the transfer matrix: GEARS (Roohani et al. 2023, MIT) trained on pseudobulk responses.

Each perturbation's z-normalised pseudobulk response is one 'cell' with condition '<GENE>+ctrl'; control 'cells' are
zero vectors (z-scores are relative to control cells, whose mean is 0). GEARS's DE step is skipped (one cell per
condition), its non-zero-gene filter is set to all genes, and model selection uses overall validation MSE (GEARS's
own selection uses the top-20 DE genes, which do not exist here). Same split protocol as src/offset_pairs.py:
40 % train (10 % of it as GEARS validation), 30 % calibration, 30 % test (first 100 of test = pilot pool).
Compared on identical splits with mean and ridge10(GO z). Metrics as offset_pairs: Delta2, qA, covB_src,
covB_src_offset, qB_oracle, pilot-only and exact adapted ball at m = 50.
usage (from virtual_cell/gears): uv run python gears_ctx.py --pairs ess:rpe1,rpe1:ess --seeds 3 [--epochs 15]"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from copy import deepcopy
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.linear_model import Ridge

VC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VC / "src"))
from adapt import fit_adapt, loo_radius  # noqa: E402
from data import DATA, go_annotations, go_embedding, target_gene  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402

from gears import GEARS, PertData  # noqa: E402
from gears.utils import loss_fct  # noqa: E402

OUT = VC / "results"
WORK = Path(__import__("os").environ.get("GEARS_WORK", str(VC / "gears" / "work")))
H5 = {"ess": "K562_essential_normalized_bulk_01.h5ad", "rpe1": "rpe1_normalized_bulk_01.h5ad"}


def load_line(k, compact):
    if k in H5:
        a = ad.read_h5ad(DATA / H5[k])
        names = np.asarray(a.obs_names)
        keep = np.array([not re.search(r"non-targeting|control", n, re.I) for n in names])
        t = np.array([target_gene(n) for n in names[keep]])
        _, first = np.unique(t, return_index=True); first = np.sort(first)
        X = np.asarray(a.X, dtype=np.float32)[keep][first]
        return X, t[first], np.asarray(a.var_names), dict(zip(a.var_names, a.var["gene_name"].astype(str)))
    X = compact[f"{k}_X"].astype(np.float32)
    t = np.array([n.split("_")[1] for n in compact[f"{k}_obs"]])
    return X, t, np.asarray(compact["genes"]), None


def load_pair(A, B, compact, n_genes=1000):
    (XA, tA, gA, symA), (XB, tB, gB, symB) = load_line(A, compact), load_line(B, compact)
    sym = symA or symB
    if sym is None:
        sym = dict(zip(*[np.asarray(x) for x in (ad.read_h5ad(DATA / H5["ess"], backed="r").var_names,
                                                 ad.read_h5ad(DATA / H5["ess"], backed="r").var["gene_name"].astype(str))]))
    ct = sorted(set(tA) & set(tB)); cg = sorted(set(gA) & set(gB) & set(compact["genes"]))
    ia = {x: i for i, x in enumerate(tA)}; ib = {x: i for i, x in enumerate(tB)}
    ga = {x: i for i, x in enumerate(gA)}; gb = {x: i for i, x in enumerate(gB)}
    YA = XA[[ia[x] for x in ct]][:, [ga[x] for x in cg]].astype(np.float64)
    YB = XB[[ib[x] for x in ct]][:, [gb[x] for x in cg]].astype(np.float64)
    fin = np.isfinite(YA).all(0) & np.isfinite(YB).all(0)
    YA, YB, cg = YA[:, fin], YB[:, fin], np.array(cg)[fin]
    top = np.argsort(YA.var(0) + YB.var(0))[::-1][:n_genes]
    YA, YB, cg = YA[:, top], YB[:, top], cg[top]
    return YA / (YA.std(0) + 1e-8), YB / (YB.std(0) + 1e-8), np.array(ct), np.array([sym.get(g, g) for g in cg])


def build_pertdata(name, YA, targets, gene_syms, n_ctrl=50):
    obs = pd.DataFrame({"condition": [f"{t}+ctrl" for t in targets] + ["ctrl"] * n_ctrl, "cell_type": "line"})
    X = sp.csr_matrix(np.r_[YA, np.zeros((n_ctrl, YA.shape[1]))].astype(np.float32))
    a = ad.AnnData(X, obs=obs, var=pd.DataFrame({"gene_name": gene_syms}, index=[f"g{i}" for i in range(len(gene_syms))]))
    pd_ = PertData(str(WORK))
    pd_.new_data_process(dataset_name=name, adata=a, skip_calc_de=True)
    cn = pd_.adata.obs["condition_name"].astype(str).values
    pd_.adata.uns["non_zeros_gene_idx"] = {c: np.arange(YA.shape[1]) for c in set(cn)}
    return pd_


def train_gears(pd_, tr_conds, va_conds, seed, epochs, device, hidden=64, lr=1e-3, step=1, loss="gears", dlam=None):
    split = {"train": tr_conds + ["ctrl"], "val": va_conds, "test": va_conds}
    sp_path = WORK / f"split_{seed}.pkl"; pickle.dump(split, open(sp_path, "wb"))
    pd_.prepare_split(split="custom", seed=seed, split_dict_path=str(sp_path))
    pd_.get_dataloader(batch_size=32, test_batch_size=128)
    torch.manual_seed(seed); np.random.seed(seed)
    Xs = pd_.adata.X; pd_.adata.X = Xs.toarray()  # GEARS 0.1.2 __init__ indexes X with a pandas mask (fails on new scipy)
    g = GEARS(pd_, device=device)
    pd_.adata.X = Xs  # prediction path calls .toarray() on control cells
    g.model_initialize(hidden_size=hidden) if dlam is None else g.model_initialize(hidden_size=hidden, direction_lambda=dlam)
    model = g.model.to(device); best, best_val = deepcopy(model), np.inf
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.5)
    for ep in range(epochs):
        model.train()
        for batch in pd_.dataloader["train_loader"]:
            batch.to(device); opt.zero_grad()
            out = model(batch)
            if loss == "mse":
                l_ = ((out - batch.y.reshape(out.shape)) ** 2).mean()
            else:
                l_ = loss_fct(out, batch.y, batch.pert, ctrl=g.ctrl_expression, dict_filter=g.dict_filter,
                              direction_lambda=g.config["direction_lambda"])
            loss_v = l_
            loss_v.backward(); torch.nn.utils.clip_grad_value_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval(); se, n = 0.0, 0
        with torch.no_grad():
            for batch in pd_.dataloader["val_loader"]:
                batch.to(device); p = model(batch)
                se += float(((p - batch.y.reshape(p.shape)) ** 2).sum()); n += p.numel()
        if se / n < best_val:
            best_val, best = se / n, deepcopy(model)
    g.best_model = best; g.saved_pred = {}
    return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default="ess:rpe1"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default="gears_ctx_metrics.json")
    ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--step", type=int, default=1); ap.add_argument("--loss", default="gears"); ap.add_argument("--dlam", type=float, default=None)
    a = ap.parse_args(); alpha = a.alpha; dev = "cuda" if torch.cuda.is_available() else "cpu"
    WORK.mkdir(parents=True, exist_ok=True)
    compact = dict(np.load(VC / "data" / "nadig_compact.npz", allow_pickle=True))
    go = go_annotations(); res = {}
    for A, B in [tuple(x.split(":")) for x in a.pairs.split(",")]:
        YA, YB, targets, syms = load_pair(A, B, compact)
        Z, _, _, has_go = go_embedding(targets, go)
        pd_ = build_pertdata(f"{A}_{B}", YA, targets, syms)
        inpert = np.array([t in set(pd_.pert_names) for t in targets])
        keep = has_go & inpert
        YA, YB, Z, targets = YA[keep], YB[keep], Z[keep], targets[keep]; N = len(targets)
        pd_ = build_pertdata(f"{A}_{B}_f", YA, targets, syms)
        print(f"== {A}->{B}: {N} targets with GO and in the GEARS graph", flush=True)
        rec = {}
        for s in range(a.seeds):
            rng = np.random.default_rng(s); r = rng.permutation(N)
            tr, ca, te_all = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
            te = te_all[100:]; pil = te_all[:50]
            nv = max(10, len(tr) // 10); tr_g, va_g = tr[nv:], tr[:nv]
            g = train_gears(pd_, [f"{t}+ctrl" for t in targets[tr_g]], [f"{t}+ctrl" for t in targets[va_g]], s, a.epochs, dev, a.hidden, a.lr, a.step, a.loss, a.dlam)
            pred_all = g.predict([[t] for t in targets])
            Pg = np.stack([pred_all[t] for t in targets])
            rg = Ridge(alpha=10.0).fit(Z[tr], YA[tr])
            preds = {"mean": np.tile(YA[tr].mean(0), (N, 1)), "ridge10": rg.predict(Z), "gears": Pg}
            delta = YB[ca].mean(0) - YA[ca].mean(0)
            for name, P in preds.items():
                d = rec.setdefault(name, {}); put = lambda k, v, d=d: d.setdefault(k, []).append(float(v))  # noqa: E731
                put("Delta2", (delta ** 2).mean())
                qA = split_cp_q(rmse(P[ca], YA[ca]), alpha); put("qA", qA)
                put("covA", (rmse(P[te], YA[te]) <= qA).mean())
                put("covB_src", (rmse(P[te], YB[te]) <= qA).mean())
                put("covB_src_offset", (rmse(P[te] + delta, YB[te]) <= qA).mean())
                put("qB_oracle", split_cp_q(rmse(P[ca], YB[ca]), alpha))
                q0 = split_cp_q(rmse(P[pil], YB[pil]), alpha)
                put("pilot_m50_q", q0); put("pilot_m50_cov", (rmse(P[te], YB[te]) <= q0).mean())
                R = 51 / 50 * loo_radius("offset", P[pil], YB[pil], alpha)
                st = rmse(fit_adapt("offset", P[pil], YB[pil])(P[te]), YB[te])
                put("exact_m50_q", R); put("exact_m50_cov", (st <= R).mean())
            print(f"  seed {s}: " + " ".join(f"{n}: qA {rec[n]['qA'][-1]:.3f} covB {rec[n]['covB_src'][-1]:.3f}" for n in preds), flush=True)
        agg = {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in rec.items()}
        res[f"{A}->{B}"] = {"N": int(N), **agg}
        for n, d in agg.items():
            print(f"  {n:8s} " + " ".join(f"{k}={v:.3f}" for k, v in d.items()), flush=True)
    (OUT / a.out).write_text(json.dumps(res, indent=1))
    print("GEARS_CTX_DONE", flush=True)


if __name__ == "__main__":
    main()
