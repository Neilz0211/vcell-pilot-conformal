"""Feasibility check (exploratory): transfer a per-perturbation DIFFICULTY model from the source line, calibrate the
normalised score on target pilots.

Idea: across cell lines the scale of the errors changes (pilots must recalibrate it), but WHICH perturbations are hard
may transfer. Difficulty sigma(z) is learned in the source line only: the source model f (ridge, lambda = 10, GO-SVD z)
is fitted on the source train split; its residual RMSE on the otherwise unused source split (40-70 %) is regressed
(log scale, ridge) on [z, log ||f(z) - mean||]. In the target, the normalised score rmse(pred, y) / sigma(z) is
calibrated on the m pilots, giving a per-perturbation radius q * sigma(z_i).

Arms (leak-free "sourcetrain" preprocessing and the paired design of paired.py: fixed 280-target test set, pilots =
first m of the calibration pool):
  P0  pilot-only split CP, constant radius                       (exact)
  P1  pilot-only split CP on the normalised score                (exact)
  A0  Prop A outer ball, constant radius                         (exact)
  A1s offset from the first m/2 pilots, normalised split CP on the other m/2   (exact)
  A1j offset from all m pilots, normalised leave-one-out scores, (m+1)/m factor (heuristic; coverage measured)
Reported: coverage overall and by difficulty quartile (conditional coverage), mean and median radius, Spearman of
sigma(z) with the target residual RMSE (does difficulty transfer?) next to the same in the source line, and the share of
test perturbations whose set excludes (a) the null response and (b) the pilot-mean response.
usage: uv run python src/difficulty.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MS = (20, 50, 100)


def difficulty_model(Z, P, mu, idx, res):
    X = np.column_stack([Z[idx], np.log(np.sqrt(((P[idx] - mu) ** 2).mean(1)) + 1e-3)])
    h = Ridge(alpha=10.0).fit(X, np.log(res + 1e-6))
    def sig(rows):
        Xr = np.column_stack([Z[rows], np.log(np.sqrt(((P[rows] - mu) ** 2).mean(1)) + 1e-3)])
        return np.exp(h.predict(Xr))
    return sig


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); al = a.alpha
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    pairs = [(x, y) for x in lines for y in lines if x != y]
    go = go_annotations(); cache = {}; res = {}
    for A, B in pairs:
        YA0, YB0, targets = load_pair(A, B, compact, cache, prep="raw")
        Z, _, _, has = go_embedding(targets, go)
        YA0, YB0, Z = YA0[has], YB0[has], Z[has]; N = len(Z)
        raw = {}
        put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
        for s in range(a.seeds):
            r = np.random.default_rng(s).permutation(N)
            tr, mid, pool = r[: int(0.4 * N)], r[int(0.4 * N): int(0.7 * N)], r[int(0.7 * N):]
            test, cal = pool[-280:], pool[:-280]
            top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
            YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
            f = Ridge(alpha=10.0).fit(Z[tr], YA[tr]); P = f.predict(Z); mu = YA[tr].mean(0)
            sig = difficulty_model(Z, P, mu, mid, rmse(P[mid], YA[mid]))
            st, sB = sig(test), rmse(P[test], YB[test])
            put("rank_target", spearmanr(st, sB).statistic)
            put("rank_source", spearmanr(st, rmse(P[test], YA[test])).statistic)
            q4 = np.quantile(st, [0.25, 0.5, 0.75]); qbin = np.digitize(st, q4)
            for m in MS:
                pil = cal[:m]; Pp, Yp = P[pil], YB[pil]; sp = sig(pil); ybar = Yp.mean(0)
                arms = {}
                q0 = split_cp_q(rmse(Pp, Yp), al); arms["P0"] = (P[test], np.full(len(test), q0))
                q1 = split_cp_q(rmse(Pp, Yp) / sp, al); arms["P1"] = (P[test], q1 * st)
                E = Yp - Pp; b = E.mean(0); bl = (E.sum(0)[None] - E) / (m - 1)
                R0 = (m + 1) / m * split_cp_q(np.sqrt(((E - bl) ** 2).mean(1)), al); arms["A0"] = (P[test] + b, np.full(len(test), R0))
                Rj = (m + 1) / m * split_cp_q(np.sqrt(((E - bl) ** 2).mean(1)) / sp, al); arms["A1j"] = (P[test] + b, Rj * st)
                h = m // 2; bh = E[:h].mean(0)
                qs = split_cp_q(rmse(Pp[h:] + bh, Yp[h:]) / sp[h:], al); arms["A1s"] = (P[test] + bh, qs * st)
                for name, (C, rad) in arms.items():
                    sc = rmse(C, YB[test]); cov = sc <= rad
                    put(f"{name}_cov_{m}", cov.mean()); put(f"{name}_rmean_{m}", rad.mean()); put(f"{name}_rmed_{m}", np.median(rad))
                    for k in range(4):
                        put(f"{name}_covq{k}_{m}", cov[qbin == k].mean())
                    put(f"{name}_excl0_{m}", (rmse(C, np.zeros_like(C)) > rad).mean())
                    put(f"{name}_exclbar_{m}", (rmse(C, np.tile(ybar, (len(test), 1))) > rad).mean())
        summ = {k: float(np.mean(v)) for k, v in raw.items()}
        res[f"{A}->{B}"] = {"summary": summ, "raw": raw}
        x = summ
        print(f"{A}->{B}: rank target {x['rank_target']:.2f} (source {x['rank_source']:.2f}) | m50 cov P0/P1/A0/A1s/A1j "
              + "/".join(f"{x[f'{n}_cov_50']:.3f}" for n in ('P0', 'P1', 'A0', 'A1s', 'A1j'))
              + " | mean r " + "/".join(f"{x[f'{n}_rmean_50']:.2f}" for n in ('P0', 'P1', 'A0', 'A1s', 'A1j'))
              + " | median r " + "/".join(f"{x[f'{n}_rmed_50']:.2f}" for n in ('P0', 'P1', 'A0', 'A1s', 'A1j'))
              + " | cov by difficulty P0 " + "/".join(f"{x[f'P0_covq{k}_50']:.2f}" for k in range(4))
              + " P1 " + "/".join(f"{x[f'P1_covq{k}_50']:.2f}" for k in range(4))
              + " | excl0 P0/P1 " + f"{x['P0_excl0_50']:.3f}/{x['P1_excl0_50']:.3f}"
              + " exclbar A0/A1s " + f"{x['A0_exclbar_50']:.3f}/{x['A1s_exclbar_50']:.3f}", flush=True)
    (OUT / "difficulty_metrics.json").write_text(json.dumps(res))
    print("DIFFICULTY_DONE", flush=True)


if __name__ == "__main__":
    main()
