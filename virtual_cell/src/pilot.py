"""Pilot-anchored checks for hidden library shift (K562 genome-wide pseudobulk).

Setting: calibration library (natural composition, or non-essential only) and a screening library (essential only).
Feature-based weights miss the shift (day2). Now m labelled PILOT perturbations from the screening library are
available (a small wet-lab pilot). Methods compared at alpha = 0.1:
  cal_w     weighted CP with classifier-estimated weights (no pilots; day2 baseline)
  pilot     split CP on the m pilot scores alone (valid; infinite when m < 1/alpha - 1)
  test      binomial test of the cal_w set's coverage on the pilots (H0: coverage >= 1-alpha, one-sided, level 0.05);
            if rejected -> use `pilot` (or abstain when it is infinite); else keep cal_w
  oracle_w  oracle weights (library design known), reference
Metrics on the remaining screening perturbations: coverage, finite-set rate, mean radius over finite sets, and for
`test` the rejection rate (power to detect the failure, and the false-alarm rate under no shift).
usage: uv run python src/pilot.py [--seeds 30]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import go_annotations, go_embedding  # noqa: E402
from day1 import fit_models, rmse, split_cp_q, weighted_quantile_cp  # noqa: E402
from day2 import draw, load_gwps  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args(); alpha = a.alpha
    Y, targets, ctrl, is_ess = load_gwps()
    Zg, M, vocab, has_go = go_embedding(targets, go_annotations())
    k = has_go
    Y, Zg, M, ctrl, is_ess = Y[k], Zg[k], M[k], ctrl[k], is_ess[k]
    Z = np.c_[Zg, (ctrl - ctrl.mean()) / ctrl.std()]; F = np.c_[M, Z]
    N = len(Y)
    res = {}
    for scen, (cs, ts) in {"natural_to_ess": (0.2, 1.0), "disjoint": (0.0, 1.0), "no_shift": (0.2, 0.2)}.items():
        for m in (5, 10, 20, 50):
            rec = {x: [] for x in ("cov_calw", "cov_pilot", "fin_pilot", "q_pilot", "cov_test", "fin_test", "q_test", "reject", "cov_oracle")}
            for s in range(a.seeds):
                rng = np.random.default_rng(s)
                r = rng.permutation(N)
                tr, pool = r[: int(0.3 * N)], r[int(0.3 * N):]
                f = fit_models(Z[tr], Y[tr], s)["ridge"]
                ca = draw(rng, pool, is_ess, cs, 1500)
                rest = np.setdiff1d(pool, ca)
                te_all = rng.permutation(draw(rng, rest, is_ess, ts, 600 + m))  # draw() returns essential first: shuffle
                pil, te = te_all[:m], te_all[m:]
                sc, sp, st = rmse(f(Z[ca]), Y[ca]), rmse(f(Z[pil]), Y[pil]), rmse(f(Z[te]), Y[te])
                c2 = draw(rng, pool, is_ess, cs, 1500); t2 = draw(rng, pool, is_ess, ts, 1500)
                clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.r_[F[c2], F[t2]], np.r_[np.zeros(len(c2)), np.ones(len(t2))])
                pr = lambda x: np.clip(clf.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4)  # noqa: E731
                wc = pr(F[ca]) / (1 - pr(F[ca]))
                q_te = np.array([weighted_quantile_cp(sc, wc, float(pr(F[i:i + 1])[0] / (1 - pr(F[i:i + 1])[0])), alpha)[0] for i in te])
                q_pi = np.array([weighted_quantile_cp(sc, wc, float(pr(F[i:i + 1])[0] / (1 - pr(F[i:i + 1])[0])), alpha)[0] for i in pil])
                rec["cov_calw"].append(float((st <= q_te).mean()))
                qp = split_cp_q(sp, alpha) if m >= int(np.ceil(1 / alpha)) - 1 else np.inf
                rec["cov_pilot"].append(float((st <= qp).mean())); rec["fin_pilot"].append(float(np.isfinite(qp)))
                rec["q_pilot"].append(float(qp) if np.isfinite(qp) else np.nan)
                # binomial test on pilot coverage of the cal_w sets
                x = int((sp <= q_pi).sum())
                pval = binom.cdf(x, m, 1 - alpha)  # P(X <= x | coverage = 1-alpha)
                rej = pval < 0.05
                rec["reject"].append(float(rej))
                if rej:
                    rec["cov_test"].append(float((st <= qp).mean())); rec["fin_test"].append(float(np.isfinite(qp)))
                    rec["q_test"].append(float(qp) if np.isfinite(qp) else np.nan)
                else:
                    rec["cov_test"].append(float((st <= q_te).mean())); rec["fin_test"].append(float(np.isfinite(q_te).mean()))
                    rec["q_test"].append(float(np.nanmean(np.where(np.isfinite(q_te), q_te, np.nan))))
                w_or = lambda e: np.where(e, ts / max(cs, 1e-12) if cs > 0 else np.inf, (1 - ts) / max(1 - cs, 1e-12))  # noqa: E731
                if cs == 0:
                    rec["cov_oracle"].append(1.0)  # oracle abstains everywhere (support violation)
                else:
                    wo = w_or(is_ess[ca]); qo = np.array([weighted_quantile_cp(sc, wo, float(w_or(is_ess[i:i + 1])[0]), alpha)[0] for i in te])
                    rec["cov_oracle"].append(float((st <= qo).mean()))
            row = {kk: float(np.nanmean(v)) for kk, v in rec.items()}
            res[f"{scen}_m{m}"] = row
            print(scen, m, json.dumps({kk: round(v, 3) for kk, v in row.items()}), flush=True)
    (OUT / "pilot_metrics.json").write_text(json.dumps(res, indent=1))
    print("PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
