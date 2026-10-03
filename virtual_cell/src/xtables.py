"""LaTeX fragments for the supplementary (2026-10-03): learned cross-line maps (Appendix C) and the Tahoe cross-line
analysis (Appendix D). Every number is computed here from the result files; nothing is typed by hand.
Inputs: results/pod_x/{xmap_pairs,xmulti_pairs,xdesign_pairs}, results/pod_x/tahoe_xline.json, results/xpropa_s20_metrics.json,
results/xsplit_all_s20.json, results/xsens_s20_metrics.json, results/xdesign_pairs/*_ct.json.
Outputs: paper/supp_xmaps.tex, paper/supp_xtahoe.tex.
usage: uv run python src/xtables.py"""
import glob
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results"; P = R / "pod_x"; PAPER = Path(__file__).resolve().parents[2] / "paper"
NM = {"ess": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
ORDER = [(a, b) for a in ("ess", "rpe1", "hepg2", "jurkat") for b in ("ess", "rpe1", "hepg2", "jurkat") if a != b]


def ci(v):
    v = np.asarray(v, float); m = v.mean(); h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)); return m, m - h, m + h


def pc(x, d=1):
    return f"${100 * x:+.{d}f}$"


pa = json.loads((R / "xpropa_s20_metrics.json").read_text())
XM = {}
for f in glob.glob(str(P / "xmap_pairs" / "*_s20.json")): XM.update(json.loads(Path(f).read_text()))
XS = json.loads((R / "xsplit_all_s20.json").read_text())
MU = {}
for f in glob.glob(str(P / "xmulti_pairs" / "*_s20.json")): MU.update(json.loads(Path(f).read_text()))
SE = json.loads((R / "xsens_s20_metrics.json").read_text())

L = [r"""\subsection{Learned cross-line maps (exploratory)}\label{app:xmaps}
\emph{Design.} Leak-free design of Appendix~\ref{app:readout} with the split-half noise subset of
Appendix~\ref{app:attrib} (perturbations and genes with a noise estimate in the target line), 20 seeds, the same
train/calibration/test split, pilots (first $m$ of the calibration pool) and test set as the Prop.~A runs. The map is a
reduced-rank ridge regression from the measured source response of a perturbation to its target response (rank and
penalty chosen by 5-fold cross-validation inside the fitting pilots; \texttt{src/xmap.py}). ``Gap closed'' is
$(\mathrm{MSE}_{\rm mean}-\mathrm{MSE}_{\rm map})/(\mathrm{MSE}_{\rm mean}-\text{noise energy})$ on the test set, for a map
fitted on all $m$ pilots. Radii are those of exact split-conformal sets: the map is refitted (and retuned) on a fixed
fraction of the pilots and calibrated on the rest. They are compared seed by seed with the current recipe, the outer
ball of Proposition~A for adapted ridge on the same $m$ pilots (\texttt{src/xpropa.py}). A PerturbMap-style blend
(full-rank ridge expert mixed with ridge-GO plus offset, weight chosen by cross-validation) follows
\cite{cui2026perturbmap} only in spirit; it is not the published method. Summary script \texttt{src/xsummary.py},
tables \texttt{src/xtables.py}, figure \texttt{src/fig\_budget.py}.
"""]
# ---- per-pair table (half/half) ----
L.append(r"""\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Learned map, one source line, half/half split: gap closed (map fitted on all $m$ pilots) and radius of the
exact split-conformal set relative to the current recipe, mean over 20 seeds [95\% seed interval]; last column:
coverage at $m=50/100/200$.}\label{tab:xmap_pairs}
\begin{tabular}{lcccccccc}\toprule
Pair & gap 50 & gap 100 & gap 200 & radius 50 (\%) & radius 100 (\%) & radius 200 (\%) & coverage\\\midrule""")
for a, b in ORDER:
    k = f"{a}->{b}"; s = XM[k]["summary"]; raw = XM[k]["raw"]; p = pa[k]["raw"]
    cells = []
    for m in (50, 100, 200):
        c = ci(np.array(raw[f"rrr_q_{m}"]) / np.array(p[f"propA_ridge_q_{m}"]) - 1)
        cells.append(f"{pc(c[0])} [{100 * c[1]:+.1f}, {100 * c[2]:+.1f}]")
    L.append(f"{NM[a]} $\\to$ {NM[b]} & {100 * s['rrr_gap_50']:.0f}\\% & {100 * s['rrr_gap_100']:.0f}\\% & {100 * s['rrr_gap_200']:.0f}\\% & "
             + " & ".join(cells) + f" & {s['rrr_cov_50']:.3f}/{s['rrr_cov_100']:.3f}/{s['rrr_cov_200']:.3f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
# ---- allocation + PerturbMap-style + sensitivity summary ----
L.append(r"""\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{4pt}
\caption{Radius relative to the current recipe, median over the 12 pairs [range] of seed means, by fitting fraction
(60\% and 70\% were run after the 50\% results), for the PerturbMap-style blend, and without the noise-availability
filter (half/half; \texttt{src/xsens.py}); coverage range over pairs.}\label{tab:xmap_alloc}
\begin{tabular}{lccccc}\toprule
Arm & $m=50$ (\%) & $m=100$ (\%) & $m=200$ (\%) & coverage\\\midrule""")


def row(name, getter):
    cells, covs = [], []
    for m in (50, 100, 200):
        v = []
        for a, b in ORDER:
            q, cv = getter(f"{a}->{b}", m); v.append(np.mean(np.array(q) / np.array(pa[f'{a}->{b}']["raw"][f"propA_ridge_q_{m}"]) - 1)); covs.append(np.mean(cv))
        v = np.array(v); cells.append(f"{pc(np.median(v))} [{100 * v.min():+.1f}, {100 * v.max():+.1f}]")
    return f"{name} & " + " & ".join(cells) + f" & {min(covs):.3f}--{max(covs):.3f}\\\\"


L.append(row("map, fit 50\\%", lambda k, m: (XM[k]["raw"][f"rrr_q_{m}"], XM[k]["raw"][f"rrr_cov_{m}"])))
L.append(row("map, fit 60\\%", lambda k, m: (XS[k]["raw"][f"rrr_f60_q_{m}"], XS[k]["raw"][f"rrr_f60_cov_{m}"])))
L.append(row("map, fit 70\\%", lambda k, m: (XS[k]["raw"][f"rrr_f70_q_{m}"], XS[k]["raw"][f"rrr_f70_cov_{m}"])))
L.append(row("PerturbMap-style, 50\\%", lambda k, m: (XM[k]["raw"][f"pmap_q_{m}"], XM[k]["raw"][f"pmap_cov_{m}"])))
sens = []
for m in (50, 100, 200):
    v = np.array([np.mean(np.array(SE[f"{a}->{b}"]["raw"][f"rrr_q_{m}"]) / np.array(SE[f"{a}->{b}"]["raw"][f"propA_ridge_q_{m}"]) - 1) for a, b in ORDER])
    sens.append(f"{pc(np.median(v))} [{100 * v.min():+.1f}, {100 * v.max():+.1f}]")
scov = [SE[f"{a}->{b}"]["summary"][f"rrr_cov_{m}"] for a, b in ORDER for m in (50, 100, 200)]
L.append("map, 50\\%, no noise filter & " + " & ".join(sens) + f" & {min(scov):.3f}--{max(scov):.3f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
# ---- multi-source ----
L.append(r"""\paragraph{Three source lines.} Targets shared by all four lines (\texttt{src/xmulti.py}); readout genes and
scales from the source training split pooled over the three source lines. ``Average'' = mean of the three source
responses plus the pilot offset, i.e.\ Proposition~A with another base predictor (outer ball, exact); ``learned'' = one
weight per source line plus a reduced-rank ridge correction of the residual, split conformal (half/half). Radii relative
to the outer ball of the adapted pilot mean.
\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Three source lines: radius change relative to the adapted pilot mean (mean over seeds), average / learned, and
coverage of the learned set.}\label{tab:xmulti}
\begin{tabular}{lccccc}\toprule
Target & $m=25$ (\%) & $m=50$ (\%) & $m=100$ (\%) & $m=200$ (\%) & coverage (learned)\\\midrule""")
for B in ("ess", "rpe1", "hepg2", "jurkat"):
    k = f"others->{B}"; p = pa[k]["raw"]; x = MU[k]["raw"]; s = MU[k]["summary"]
    cells = [f"{pc(np.mean(np.array(p[f'propA_avg_q_{m}']) / np.array(p[f'propA_mmean_q_{m}']) - 1))} / "
             f"{pc(np.mean(np.array(x[f'synth_rrr_q_{m}']) / np.array(p[f'propA_mmean_q_{m}']) - 1))}" for m in (25, 50, 100, 200)]
    L.append(f"{NM[B]} & " + " & ".join(cells) + f" & {min(s[f'synth_rrr_cov_{m}'] for m in (25, 50, 100, 200)):.3f}--{max(s[f'synth_rrr_cov_{m}'] for m in (25, 50, 100, 200)):.3f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
# ---- designs ----
L.append(r"""\paragraph{Designed fitting pilots} (\texttt{src/xdesign.py}, map fitted on $m_{\rm fit}$ designed pilots, split
conformal on 50 random calibration pilots). Designs use only source responses: k-means++ seeds, max-min distance, a
greedy volume (D-optimal-like) rule, and the perturbations with the largest source response (``strongest''). Two
evaluations: (i) the test set is the panel that remains after the fitting and calibration pilots are removed (the
deployment estimand, for which validity holds); (ii) a common test set fixed before the design chooses
(\texttt{--common-test}), which isolates the learning effect but is not exchangeable with the calibration pilots.
\begin{table}[h]\centering\scriptsize\setlength{\tabcolsep}{3pt}
\caption{Test MSE of the map relative to random fitting pilots, median over 12 pairs [range]; coverage range over pairs
and designs.}\label{tab:xdesign}
\begin{tabular}{llcccc}\toprule
Evaluation & $m_{\rm fit}$ & k-means++ (\%) & strongest (\%) & coverage, random design & coverage, all designs\\\midrule""")
for label, files, key_ok in (("remaining panel", sorted(glob.glob(str(P / "xdesign_pairs" / "*_s20.json"))), None),
                             ("common test set", sorted(glob.glob(str(R / "xdesign_pairs" / "*_ct.json"))), None)):
    for mf in (25, 50, 100):
        km, bg, cr, ca = [], [], [], []
        for f in files:
            d = json.loads(Path(f).read_text()); x = d[next(iter(d))]["summary"]
            if f"random_rrr_mse_{mf}" not in x: continue
            km.append(x[f"kmeans_rrr_mse_{mf}"] / x[f"random_rrr_mse_{mf}"] - 1); bg.append(x[f"big_rrr_mse_{mf}"] / x[f"random_rrr_mse_{mf}"] - 1)
            cr.append(x[f"random_rrr_cov_{mf}"]); ca += [x[f"{dn}_rrr_cov_{mf}"] for dn in ("random", "kmeans", "maxmin", "dpp", "big")]
        L.append(f"{label} & {mf} & {pc(np.median(km))} [{100 * min(km):+.1f}, {100 * max(km):+.1f}] & {pc(np.median(bg))} [{100 * min(bg):+.1f}, {100 * max(bg):+.1f}] & "
                 f"{min(cr):.3f}--{max(cr):.3f} & {min(ca):.3f}--{max(ca):.3f}\\\\")
L.append(r"\bottomrule\end{tabular}\end{table}")
L.append(r"""\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figs/fig5_budget.pdf}
\caption{Pilot budget. (a) Radius of the map's exact split-conformal set relative to the current recipe, median over
the 12 pairs (band: range of seed means), by fitting fraction, and the PerturbMap-style blend. (b) Coverage. (c) Three
source lines: average plus offset (Prop.~A, dashed) and learned weights plus reduced-rank residual (split, solid),
relative to the adapted pilot mean.}\label{fig:budget_supp}\end{figure}""")
# reference numbers quoted in the main text
gr = {m: np.median([XM[f"{a}->{b}"]["summary"][f"ridge_gap_{m}"] for a, b in ORDER]) for m in (50, 100, 200)}
PS = json.loads((R / "paired_sourcetrain_metrics.json").read_text())
ex = {k: 100 * np.mean(np.array(PS[k]["raw"]["adapted_q_50"]) / np.array(PS[k]["raw"]["pilot_q_50"]) - 1) for k in ("ess->hepg2", "ess->rpe1")}
excov = {k: np.mean(PS[k]["raw"]["pilot_cov_50"]) for k in ("ess->hepg2", "ess->rpe1")}
L.append(rf"""\paragraph{{Reference numbers.}} For comparison with the maps, ridge on GO features plus the pilot offset
closes a median {100 * gr[50]:.0f}, {100 * gr[100]:.0f} and {100 * gr[200]:.0f}\% of the same gap at $m=50$, 100 and 200.
Worked example of the Discussion (leak-free paired design, \texttt{{results/paired\_sourcetrain\_metrics.json}}, 20
seeds, $m=50$): the adapted ridge set is {ex['ess->hepg2']:+.1f}\% (K562$\to$HepG2) and {ex['ess->rpe1']:+.1f}\%
(K562$\to$RPE1) relative to pilot-only split conformal, mean of per-seed ratios; pilot-only coverage
{excov['ess->hepg2']:.3f} and {excov['ess->rpe1']:.3f}.""")
(PAPER / "supp_xmaps.tex").write_text("\n".join(L) + "\n\\clearpage\n", encoding="utf-8")

# ---------------- Tahoe cross-line (Appendix D) ----------------
T = json.loads((P / "tahoe_xline.json").read_text())["results"]
g = lambda k: np.array([r[k] for r in T if k in r])  # noqa: E731
lines_cov = {}
for r in T: lines_cov.setdefault(r["t"], []).append(r["cov0_other"])
lc = np.array([np.mean(v) for v in lines_cov.values()])
rr = np.array([r["q_otheroff_50"] / r["q_pmean_50"] - 1 for r in T]); rs = np.array([r["q_synth_50"] / r["q_pmean_50"] - 1 for r in T])
TL = [r"""\paragraph{Exploratory: predicting a line from the other lines} (\texttt{pod/tahoe\_xline.py}; not pre-registered).
For each target line with at least 400 units (plate $\times$ drug $\times$ dose) having at least 30 cells, the units are
split at random into 300 test units and a pilot pool (so pilots and test units are exchangeable within the line; unlike
the registered analysis, splits are not grouped by drug). Each gene is scaled by its standard deviation over the other
lines. Predictors: the other lines' mean response to the unit; that mean plus the pilot offset (Prop.~A form); the
target's pilot mean; a ``synthetic control'' with one ridge weight per other line fitted on the pilots. Radii are exact
split conformal (half/half) for all arms. Zero-pilot calibration pools the other target lines' normalised scores and
rescales by a scale predicted from the line's median control-cell count; the other lines' predictors may average over
the held-out line, so this pipeline is not strictly free of target labels.""",
      rf"""\begin{{table}}[h]\centering\scriptsize
\caption{{Tahoe cross-line prediction, {len(lc)} target lines $\times$ 5 seeds (means; 5th percentile in brackets).}}\label{{tab:tahoe_xline}}
\begin{{tabular}}{{lcc}}\toprule
Quantity & $m=50$ & $m=100$\\\midrule
test MSE, pilot mean & {g('mse_pmean_50').mean():.3f} & {g('mse_pmean_100').mean():.3f}\\
test MSE, other lines + offset & {g('mse_otheroff_50').mean():.3f} & {g('mse_otheroff_100').mean():.3f}\\
test MSE, synthetic control & {g('mse_synth_50').mean():.3f} & {g('mse_synth_100').mean():.3f}\\
radius, pilot mean & {g('q_pmean_50').mean():.3f} & {g('q_pmean_100').mean():.3f}\\
radius, other lines + offset & {g('q_otheroff_50').mean():.3f} & {g('q_otheroff_100').mean():.3f}\\
radius, synthetic control & {g('q_synth_50').mean():.3f} & {g('q_synth_100').mean():.3f}\\
coverage, other lines + offset & {g('cov_otheroff_50').mean():.3f} [{np.percentile(g('cov_otheroff_50'), 5):.3f}] & {g('cov_otheroff_100').mean():.3f} [{np.percentile(g('cov_otheroff_100'), 5):.3f}]\\
\bottomrule\end{{tabular}}\end{{table}}
With zero pilots the other lines' mean has test MSE {g('mse_other').mean():.3f}; restricting it to the 5 or 20 lines
most similar in baseline expression gives {g('mse_simw5').mean():.3f} and {g('mse_simw20').mean():.3f}. At $m=50$ the
radius of other lines + offset is {100 * rr.mean():+.1f}\% relative to the pilot mean (synthetic control
{100 * rs.mean():+.1f}\%). Zero-pilot borrowed calibration covers {lc.mean():.3f} on average; {(lc < 0.85).sum()} of
{len(lc)} lines fall below $0.85$ (lowest {lc.min():.3f}).
"""]
TA = json.loads((R / "tahoe_all_metrics.json").read_text())["pairs"]
fr = np.mean([abs(p["ridge10"]["sB2"] - p["ridge10"]["sA2"]) > p["ridge10"]["Delta2"] for p in TA.values() if "ridge10" in p])
TL.append(rf"""\paragraph{{Residual scale versus shift.}} Over the {sum('ridge10' in p for p in TA.values())} registered line
pairs (ridge, $\lambda=10$; \texttt{{results/tahoe\_all\_metrics.json}}), the change in residual energy after removing the
shift, $|\sigma_B^2-\sigma_A^2|$, exceeds the shift energy $\Delta^2$ in {100 * fr:.1f}\% of pairs (not pre-registered).""")
(PAPER / "supp_xtahoe.tex").write_text("\n".join(TL) + "\n\\clearpage\n", encoding="utf-8")
print("wrote supp_xmaps.tex and supp_xtahoe.tex")
