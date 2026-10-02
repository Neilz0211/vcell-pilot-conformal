"""Paired, leak-checked budget comparison (GPT review 3).

Design per directed cross-line pair and seed (all methods share the SAME pilots; paired by seed):
  targets split by gene: 40 % train (source model), 30 % unused, last 30 % = target screening pool;
  of the pool, the last 280 targets = fixed independent TEST set; the rest = CALIBRATION pool (~310-410 targets);
  pilots = the first m of the calibration pool (random order), m in {20, 50, 100}.
Methods (alpha = 0.1), all evaluated on the same test set:
  full      unadapted ridge, split CP on the WHOLE calibration pool (best the unadapted model can do here)
  pilot     unadapted ridge, split CP on the m pilots
  adapted   Prop A outer ball: offset from the m pilots, (m+1)/m * LOO quantile on the same pilots (reuse)
  split     same budget, no reuse: offset from the first m/2 pilots, split CP on the other m/2 (standard, valid)
Preprocessing: --prep pooled (original: genes/SD from both lines) or --prep source (leak-free: genes and per-gene SD
from the source line only).
Reported per pair: mean radius and empirical coverage of each method; paired ratios
  adapted_m / full - 1, adapted_m / split_m - 1, adapted_m / pilot_m - 1
with mean, 95 % t-interval over seeds and the number of seeds where adapted is narrower. Seeds measure sampling
stability of splits and pilot draws, not independent biological replicates.
usage: uv run python src/paired.py [--seeds 20] [--prep source]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapt import fit_adapt, loo_radius  # noqa: E402
from data import go_annotations, go_embedding  # noqa: E402
from day1 import rmse, split_cp_q  # noqa: E402
from rho_rule import load_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MS = (20, 50, 100)


def ci(x):
    x = np.asarray(x); m = x.mean(); h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(m), float(m - h), float(m + h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=20); ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--prep", default="pooled", choices=("pooled", "source", "sourcetrain"))
    a = ap.parse_args(); alpha = a.alpha
    compact = dict(np.load(Path(__file__).resolve().parents[1] / "data" / "nadig_compact.npz", allow_pickle=True))
    lines = ("ess", "rpe1", "hepg2", "jurkat")
    pairs = [(x, y) for x in lines for y in lines if x != y]
    go = go_annotations(); cache = {}; res = {}
    for A, B in pairs:
        YA0, YB0, targets = load_pair(A, B, compact, cache, prep="raw" if a.prep == "sourcetrain" else a.prep)
        Z, _, _, has_go = go_embedding(targets, go)
        YA0, YB0, Z = YA0[has_go], YB0[has_go], Z[has_go]; N = len(Z)
        raw = {}
        put = lambda k, v: raw.setdefault(k, []).append(float(v))  # noqa: E731
        for s in range(a.seeds):
            rng = np.random.default_rng(s); r = rng.permutation(N)
            tr, pool = r[: int(0.4 * N)], r[int(0.7 * N):]
            test, cal = pool[-280:], pool[:-280]
            if a.prep == "sourcetrain":  # genes and per-gene scale fitted on the source TRAIN split only, then frozen
                top = np.argsort(YA0[tr].var(0))[::-1][:1000]; sd = YA0[tr][:, top].std(0) + 1e-8
                YA, YB = YA0[:, top] / sd, YB0[:, top] / sd
            else:
                YA, YB = YA0, YB0
            assert len(np.intersect1d(test, cal)) == 0
            f = Ridge(alpha=10.0).fit(Z[tr], YA[tr]); Pt = f.predict(Z[test]); s_t = rmse(Pt, YB[test])
            qf = split_cp_q(rmse(f.predict(Z[cal]), YB[cal]), alpha); put("full_q", qf); put("full_cov", (s_t <= qf).mean())
            put("n_cal", len(cal))
            for m in MS:
                pil = cal[:m]; Pp = f.predict(Z[pil]); Yp = YB[pil]
                q0 = split_cp_q(rmse(Pp, Yp), alpha); put(f"pilot_q_{m}", q0); put(f"pilot_cov_{m}", (s_t <= q0).mean())
                R = (m + 1) / m * loo_radius("offset", Pp, Yp, alpha); st = rmse(fit_adapt("offset", Pp, Yp)(Pt), YB[test])
                put(f"adapted_q_{m}", R); put(f"adapted_cov_{m}", (st <= R).mean())
                h = m // 2; b = (Yp[:h] - Pp[:h]).mean(0)
                qs = split_cp_q(rmse(Pp[h:] + b, Yp[h:]), alpha); ss = rmse(Pt + b, YB[test])
                put(f"split_q_{m}", qs); put(f"split_cov_{m}", (ss <= qs).mean())
                # predict-the-mean baseline on the same pilots: unadapted (pilot-only CP) and adapted (= the pilot mean of
                # the target responses, with the same Prop A outer ball). Answers "does the source model add anything once
                # the offset is removed?"
                mu = YA[tr].mean(0); Mp = np.tile(mu, (m, 1)); Mt = np.tile(mu, (len(test), 1))
                qm0 = split_cp_q(rmse(Mp, Yp), alpha); put(f"mpilot_q_{m}", qm0); put(f"mpilot_cov_{m}", (rmse(Mt, YB[test]) <= qm0).mean())
                Rm = (m + 1) / m * loo_radius("offset", Mp, Yp, alpha); sm = rmse(fit_adapt("offset", Mp, Yp)(Mt), YB[test])
                put(f"madapted_q_{m}", Rm); put(f"madapted_cov_{m}", (sm <= Rm).mean())
        summ = {"n_cal_mean": float(np.mean(raw["n_cal"])), "full_q": float(np.mean(raw["full_q"])), "full_cov": float(np.mean(raw["full_cov"]))}
        for m in MS:
            ad = np.array(raw[f"adapted_q_{m}"])
            for ref in ("full", "split", "pilot", "madapted"):
                rq = np.array(raw["full_q"] if ref == "full" else raw[f"{ref}_q_{m}"])
                rel = 100 * (ad / rq - 1)
                summ[f"ad{m}_vs_{ref}"] = {"mean_pct": ci(rel)[0], "ci_pct": ci(rel)[1:], "wins": int((ad < rq).sum()), "n": len(ad)}
            for meth in ("pilot", "adapted", "split", "mpilot", "madapted"):
                summ[f"{meth}_q_{m}"] = float(np.mean(raw[f"{meth}_q_{m}"])); summ[f"{meth}_cov_{m}"] = float(np.mean(raw[f"{meth}_cov_{m}"]))
        res[f"{A}->{B}"] = {"summary": summ, "raw": raw}
        x = summ
        print(f"{A}->{B} [{a.prep}] ncal {x['n_cal_mean']:.0f} full {x['full_q']:.3f}/{x['full_cov']:.3f} | "
              + " | ".join(f"m{m}: ad {x[f'adapted_q_{m}']:.3f}/{x[f'adapted_cov_{m}']:.3f} split {x[f'split_q_{m}']:.3f}/{x[f'split_cov_{m}']:.3f} "
                           f"pil {x[f'pilot_q_{m}']:.3f}/{x[f'pilot_cov_{m}']:.3f}" for m in (20, 50))
              + f" || ad50 vs full {x['ad50_vs_full']['mean_pct']:+.1f}% [{x['ad50_vs_full']['ci_pct'][0]:+.1f},{x['ad50_vs_full']['ci_pct'][1]:+.1f}] wins {x['ad50_vs_full']['wins']}/{x['ad50_vs_full']['n']}"
              + f" ; ad50 vs split {x['ad50_vs_split']['mean_pct']:+.1f}% [{x['ad50_vs_split']['ci_pct'][0]:+.1f},{x['ad50_vs_split']['ci_pct'][1]:+.1f}] wins {x['ad50_vs_split']['wins']}"
              + f" ; ad50 vs adapted-mean {x['ad50_vs_madapted']['mean_pct']:+.1f}% [{x['ad50_vs_madapted']['ci_pct'][0]:+.1f},{x['ad50_vs_madapted']['ci_pct'][1]:+.1f}] wins {x['ad50_vs_madapted']['wins']}"
              + f" (adapted-mean cov {x['madapted_cov_50']:.3f})", flush=True)
    (OUT / f"paired_{a.prep}_metrics.json").write_text(json.dumps(res))
    print("PAIRED_DONE", flush=True)


if __name__ == "__main__":
    main()
