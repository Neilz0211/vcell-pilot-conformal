"""Calibration transfer for STATE versus ridge and predict-the-mean, in STATE's own response space.

Response of a perturbation in a line = pseudobulk mean over its cells minus the mean of that line's control cells, over
the 2,000 HVGs (log-normalised), from truth.npz. STATE prediction = its pseudobulk prediction minus the same real
control mean. For each source line A and seed s, one STATE model was trained on A's train genes only (st_train.sh);
every other line was predicted zero-shot from its own control cells.
Per directed pair (A -> B): evaluation genes = held-out genes of seed s present (with GO features) in both lines; they
are split 50/50 into calibration and test, repeated over R random splits (the trained models are fixed). Readout:
top 1,000 HVGs by variance of A's train-gene responses, scaled by their SD (leak-free). Models: STATE, ridge
(lambda = 10) on the same 64-d GO-SVD features STATE was given, predict-the-mean (A's train-gene mean response).
Metrics as in offset_pairs.py / paired.py: source radius, source-calibrated target coverage, oracle-offset coverage,
Delta^2, pilot-only and adapted balls at m = 20, 50, 100.
usage: python st_eval.py /root/st --seeds 0 [--ckpt final.ckpt] --out st_eval.json
"""
import argparse
import glob
import json
import math

import anndata as ad
import numpy as np
from sklearn.linear_model import Ridge

ALPHA = 0.1


def split_cp_q(s, alpha=ALPHA):
    n = len(s); k = math.ceil((1 - alpha) * (n + 1))
    return float("inf") if k > n else float(np.sort(s)[k - 1])


def rmse(P, Y):
    return np.sqrt(((P - Y) ** 2).mean(1))


def loo_offset(P, Y):
    m = len(Y); E = Y - P; b = (E.sum(0)[None] - E) / (m - 1)
    return split_cp_q(np.sqrt(((E - b) ** 2).mean(1)))


def load_state_preds(run_dir, ckpt, ctrl):
    f = glob.glob(f"{run_dir}/eval_{ckpt}/adata_pred.h5ad")
    if not f: raise FileNotFoundError(f"{run_dir}/eval_{ckpt}/adata_pred.h5ad")
    P = ad.read_h5ad(f[0])
    X = np.asarray(P.X.toarray() if hasattr(P.X, "toarray") else P.X, np.float64)
    pert = P.obs["gene"].astype(str).values; line = P.obs["cell_line"].astype(str).values
    out = {}
    for l in np.unique(line):
        for p in np.unique(pert[line == l]):
            if p == "non-targeting": continue
            out[(l, p)] = X[(line == l) & (pert == p)].mean(0) - ctrl[l]   # mean over batches if pseudobulk is per batch
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--ckpt", default="final.ckpt"); ap.add_argument("--reps", type=int, default=20); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = np.load(f"{a.root}/truth.npz", allow_pickle=True); lines = [str(x) for x in T["lines"]]
    truth = {l: dict(zip([str(p) for p in T[f"{l}_perts"]], T[f"{l}_delta"].astype(np.float64))) for l in lines}
    ctrl = {l: T[f"{l}_ctrl"].astype(np.float64) for l in lines}
    F = np.load(f"{a.root}/st_feat.npz", allow_pickle=True); feat = dict(zip([str(g) for g in F["genes"]], F["Z"].astype(np.float64)))
    res = {}
    for s in a.seeds:
        sp = json.load(open(f"{a.root}/splits/seed{s}.json")); trs, hos = set(sp["train"]), set(sp["heldout"])
        for A in lines:
            pred = load_state_preds(f"{a.root}/runs/{A}_s{s}", a.ckpt, ctrl)
            trg = sorted(g for g in trs if g in truth[A] and g in feat)
            YA_tr_full = np.stack([truth[A][g] for g in trg])
            top = np.argsort(YA_tr_full.var(0))[::-1][:1000]; sd = YA_tr_full[:, top].std(0) + 1e-8
            sc = lambda v: v[..., top] / sd  # noqa: E731
            mu = sc(YA_tr_full).mean(0)
            rg = Ridge(alpha=10.0).fit(np.stack([feat[g] for g in trg]), sc(YA_tr_full))
            for B in lines:
                if B == A: continue
                ev = sorted(g for g in hos if g in truth[A] and g in truth[B] and g in feat and (A, g) in pred and (B, g) in pred)
                YA = sc(np.stack([truth[A][g] for g in ev])); YB = sc(np.stack([truth[B][g] for g in ev]))
                Zev = np.stack([feat[g] for g in ev])
                PA = {"state": sc(np.stack([pred[(A, g)] for g in ev])), "ridge10": rg.predict(Zev), "mean": np.tile(mu, (len(ev), 1))}
                PB = {"state": sc(np.stack([pred[(B, g)] for g in ev])), "ridge10": PA["ridge10"], "mean": PA["mean"]}
                rec = {}
                for r in range(a.reps):
                    rng = np.random.default_rng(1000 * s + r); perm = rng.permutation(len(ev)); h = len(ev) // 2
                    ca, te = perm[:h], perm[h:]
                    delta = YB[ca].mean(0) - YA[ca].mean(0)
                    for mname in PA:
                        d = rec.setdefault(mname, {}); put = lambda k, v, d=d: d.setdefault(k, []).append(float(v))  # noqa: E731
                        sA = rmse(PA[mname][ca], YA[ca]); qA = split_cp_q(sA); st = rmse(PB[mname][te], YB[te])
                        put("Delta2", (delta ** 2).mean()); put("qA", qA); put("covB_src", (st <= qA).mean())
                        put("covA_test", (rmse(PA[mname][te], YA[te]) <= qA).mean())
                        # oracle offset: the shift of the model's own target-line residual mean (state's B predictions differ from A's)
                        dm = (YB[ca] - PB[mname][ca]).mean(0) - (YA[ca] - PA[mname][ca]).mean(0)
                        put("covB_src_offset", (rmse(PB[mname][te] + dm, YB[te]) <= qA).mean()); put("model_shift2", (dm ** 2).mean())
                        put("qB_oracle", split_cp_q(rmse(PB[mname][ca], YB[ca])))
                        for m in (20, 50, 100):
                            pil = ca[:m]; Pp, Yp = PB[mname][pil], YB[pil]
                            q0 = split_cp_q(rmse(Pp, Yp)); put(f"pilot_q_{m}", q0); put(f"pilot_cov_{m}", (st <= q0).mean())
                            R = (m + 1) / m * loo_offset(Pp, Yp); b = (Yp - Pp).mean(0)
                            put(f"ad_q_{m}", R); put(f"ad_cov_{m}", (rmse(PB[mname][te] + b, YB[te]) <= R).mean())
                key = f"{A}->{B}|s{s}"
                res[key] = {"n_eval": len(ev), **{mn: {k: float(np.mean(v)) for k, v in d.items()} for mn, d in rec.items()}}
                x = res[key]
                print(f"{key:20s} n={len(ev)} D2 {x['state']['Delta2']:.3f} | qA st/rg/mn {x['state']['qA']:.3f}/{x['ridge10']['qA']:.3f}/{x['mean']['qA']:.3f}"
                      f" | covB st/rg/mn {x['state']['covB_src']:.3f}/{x['ridge10']['covB_src']:.3f}/{x['mean']['covB_src']:.3f}"
                      f" | +offset st {x['state']['covB_src_offset']:.3f} | ad50 cov st {x['state']['ad_cov_50']:.3f} q st/rg/mn "
                      f"{x['state']['ad_q_50']:.3f}/{x['ridge10']['ad_q_50']:.3f}/{x['mean']['ad_q_50']:.3f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1); print("ST_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
