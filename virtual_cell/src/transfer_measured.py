"""Exploratory T1: transfer the MEASURED source-line response of the same perturbation, not a feature model.

Scenario: a perturbation panel was screened in line A (e.g. genome-wide in K562); in line B only m random pilots are
measured. Predict every other perturbation's B response from its A response, adapted with the pilots, and attach a
conformal set. Paired design and leak-free preprocessing as paired.py (genes / scales from the source train split;
fixed 280-target test set; pilots = first m of the calibration pool).
Predictors (all adapted on the same pilots):
  mean     pilot mean of the B responses (adapted predict-the-mean)             Prop A outer ball
  ridge    ridge(GO-SVD) fitted in A + pilot offset                             Prop A outer ball
  meas     Y_A(p) + pilot offset                                                Prop A outer ball
  measc    c * Y_A(p) + b, scalar c and offset b by least squares on the pilots   leave-one-out radius x (m+1)/m (heuristic)
  measc_s  same, fitted on the first m/2 pilots, split CP on the other m/2      (exact)
Reported: coverage, radius, share of test perturbations whose set excludes the null response, and share whose set
excludes the pilot-mean response (a perturbation-specific call).
usage: uv run python src/transfer_measured.py [--seeds 20]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MS = (20, 50, 100)


def fit_c(X, Y):
    xm, ym = X.mean(0), Y.mean(0); xc, yc = X - xm, Y - ym
    c = (xc * yc).sum() / max((xc * xc).sum(), 1e-12)
    return c, ym - c * xm


def loo_c(X, Y, al):
    m = len(Y); R = np.empty(m)
    for i in range(m):
        k = np.arange(m) != i; c, b = fit_c(X[k], Y[k]); R[i] = rmse(c * X[i:i + 1] + b, Y[i:i + 1])[0]
    return split_cp_q(R, al)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); al = a.alpha
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    go = go_annotations(); cache = {}; res = {}
    for A in lines:
        for B in lines:
            if A == B: continue
            YA0, YB0, targets = load_pair(A, B, compact, cache, prep="raw")
            Z, _, _, has = go_embedding(targets, go); YA0, YB0, Z = YA0[has], YB0[has], Z[has]; N = len(Z)
            raw = {}; put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
            for s in range(a.seeds):
                r = np.random.default_rng(s).permutation(N); tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]
                test, cal = pool[-280:], pool[:-280]
                top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
                YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
                Pr = Ridge(alpha=10.0).fit(Z[tr], YA[tr]).predict(Z)
                put("corr_AB", np.mean([np.corrcoef(YA[i], YB[i])[0, 1] for i in test]))
                for m in MS:
                    pil = cal[:m]; Yp = YB[pil]; ybar = Yp.mean(0)
                    arms = {}
                    for name, Q in (("mean", np.zeros_like(YA)), ("ridge", Pr), ("meas", YA)):
                        E = Yp - Q[pil]; b = E.mean(0); bl = (E.sum(0)[None] - E) / (m - 1)
                        R = (m + 1) / m * split_cp_q(np.sqrt(((E - bl) ** 2).mean(1)), al)
                        arms[name] = (Q[test] + b, R)
                    c, b = fit_c(YA[pil], Yp); arms["measc"] = (c * YA[test] + b, (m + 1) / m * loo_c(YA[pil], Yp, al)); put(f"c_{m}", c)
                    h = m // 2; c2, b2 = fit_c(YA[pil[:h]], Yp[:h])
                    arms["measc_s"] = (c2 * YA[test] + b2, split_cp_q(rmse(c2 * YA[pil[h:]] + b2, Yp[h:]), al))
                    for nm, (C, R) in arms.items():
                        put(f"{nm}_cov_{m}", (rmse(C, YB[test]) <= R).mean()); put(f"{nm}_q_{m}", R)
                        put(f"{nm}_excl0_{m}", (rmse(C, np.zeros_like(C)) > R).mean())
                        put(f"{nm}_exclbar_{m}", (rmse(C, np.tile(ybar, (len(test), 1))) > R).mean())
            x = {k: float(np.mean(v)) for k, v in raw.items()}; res[f"{A}->{B}"] = {"summary": x, "raw": raw}
            print(f"{A}->{B} corrAB {x['corr_AB']:.2f} | m50 q mean/ridge/meas/measc/measc_s "
                  + "/".join(f"{x[f'{n}_q_50']:.2f}" for n in ("mean", "ridge", "meas", "measc", "measc_s"))
                  + " cov " + "/".join(f"{x[f'{n}_cov_50']:.3f}" for n in ("mean", "ridge", "meas", "measc", "measc_s"))
                  + f" c {x['c_50']:.2f} | excl0 meas/measc {x['meas_excl0_50']:.3f}/{x['measc_excl0_50']:.3f} exclbar meas/measc "
                  + f"{x['meas_exclbar_50']:.3f}/{x['measc_exclbar_50']:.3f}", flush=True)
    (OUT / "transfer_measured_metrics.json").write_text(json.dumps(res))
    print("TRANSFER_MEASURED_DONE", flush=True)


if __name__ == "__main__":
    main()
