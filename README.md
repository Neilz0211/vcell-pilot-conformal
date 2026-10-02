# Calibrating and repairing virtual-cell predictions across cell lines

Code, result summaries and paper sources for:

> **Better in One Cell Line, More Overconfident in the Next: Calibrating and Repairing Virtual-Cell Predictions with a
> Few Pilot Perturbations.** Jianguang Zhuang, Zhuowen Ye. Submitted to RECOMB 2027.

The paper studies conformal prediction sets for perturbation-response ("virtual cell") models that are calibrated in
one cell line and used in another.
- **Finding.** Source-calibrated sets lose coverage in the new line, and they lose more for models that are better in
  the source line.
- **Mechanism.** A shared cell-context response shift drives much of the loss.
- **Method.** A few target-line pilot perturbations can be used twice, to estimate the shift and to calibrate. For
  the mean-offset adapter the full-conformal set has a closed form with finite-sample coverage ≥ 1 − α.

## Layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX sources (`main.tex`, `supplementary.tex`, `references.bib`), figures and compiled PDFs |
| `virtual_cell/src/` | All analysis code (Python) |
| `virtual_cell/gears/` | GEARS experiments (separate environment with CUDA PyTorch) |
| `virtual_cell/pod/` | Scripts used on rented cloud machines: raw single-cell → pseudobulk for HepG2/Jurkat, parallel GEARS runs, the Tahoe-100M replication and STATE retraining |
| `virtual_cell/prereg/` | Pre-registration of the Tahoe-100M replication |
| `virtual_cell/results/` | Result summaries (`*.md`) and metrics (`*.json`) behind every number in the paper |
| `virtual_cell/figures/` | Exploratory figures |

## Data (public, not included)

Download into `virtual_cell/data/`:

| File | Source |
|---|---|
| `K562_essential_normalized_bulk_01.h5ad` | Replogle et al. 2022, figshare file 35780870 (`https://ndownloader.figshare.com/files/35780870`) |
| `K562_gwps_normalized_bulk_01.h5ad` | figshare file 35773217 |
| `rpe1_normalized_bulk_01.h5ad` | figshare file 35775512 |
| HepG2 / Jurkat raw single cell | Nadig et al. 2025, GEO GSE264667 (`GSE264667_hepg2_raw_singlecell_01.h5ad`, `GSE264667_jurkat_raw_singlecell_01.h5ad`) → `pod/pseudobulk.py` → `pod/export_compact.py` gives `nadig_compact.npz` |
| `goa_human.gaf.gz`, `go-basic.obo` | Gene Ontology (`http://current.geneontology.org`), downloaded 2026-10-01; ontology `data-version: releases/2026-07-26`; MD5 `3d16e91814f541afc3cdb03153ae0f55` (gaf.gz), `26de643e50af9a1588f9a31062348cc4` (obo) |

## Environment

Uses [`uv`](https://docs.astral.sh/uv/).

```bash
cd virtual_cell && uv sync          # main analysis (CPU)
cd virtual_cell/gears && uv sync    # GEARS (CUDA 12.8 PyTorch)
```

## Reproducing the paper

All scripts run from `virtual_cell/` as `uv run python src/<script>.py [--seeds N]`. Each writes its JSON and a log
into `results/` (logs are tracked as `results/*.log`).

| Result in the paper | Command (from `virtual_cell/`) | Output in `results/` |
|---|---|---|
| Transfer matrix (Table 1, Fig. 1b, Fig. 2) | `src/offset_pairs.py --seeds 20` (K562/RPE1 pairs, same-line controls); `src/offset_pairs.py --seeds 20 --pairs rpe1:ess --out rpe1_ess_pair_metrics.json`; HepG2/Jurkat pairs on a cloud CPU machine with `pod/run_analysis.sh` and `pod/run2.sh` | `offset_pairs_metrics.json`, `nadig_pairs_metrics.json`, `matrix_pairs_metrics.json`, `rpe1_ess_pair_metrics.json`; summaries `transfer_matrix.md`, `nadig_pairs.md` |
| HepG2/Jurkat pseudobulk | `pod/pseudobulk.py` on GEO raw files, then `pod/export_compact.py` | `data/nadig_compact.npz` (not tracked) |
| Model ladder (Fig. 1a) | `src/ladder.py --seeds 20` | `ladder_metrics.json`, `ladder_run20.log` |
| Decomposition, same-family path, drop-RPE1, pilot-draw spread | `src/decomp.py --seeds 20` | `decomp_metrics.json`, `decomp_run20.log` |
| GEARS, 4 directions (sweep and tuned) | `gears/gears_ctx.py --pairs ess:rpe1,rpe1:ess,hepg2:jurkat,jurkat:hepg2 --seeds 3 [--loss mse --hidden 64 --epochs 60 --step 20 --lr 3e-3]` (from `virtual_cell/gears/`) | `gears_ctx_metrics.json`, `gears_tuned_metrics.json`, `gears_ctx.md` |
| GEARS, 12 pairs x 10 seeds (Fig. 1a stars, Table S) | `pod/setup_gears.sh` then `pod/run_gears12.sh` on a cloud GPU; per-seed lines collected into `results/gears12/seed_lines.txt`; summary `src/gears12_summary.py` | `gears12/`, `gears12_summary.json` |
| Adapt-then-calibrate, exact region vs outer ball | `src/adapt.py`, `src/adapt_exact.py` | `adapt_metrics.json`, `adapt_exact_metrics.json` |
| Paired leak-free comparison, adapted pilot mean | `src/paired.py --seeds 20 --prep sourcetrain` and `--prep pooled` | `paired_*_metrics.json`, `paired.md` |
| Low-rank adapters | `src/lowrank.py --seeds 20` | `lowrank_metrics.json`, `lowrank_run20.log` |
| Equivalent pilot budget | `src/eq_pilots.py --seeds 20` | `eq_pilots_metrics.json`, `eq_pilots.md` |
| Pre-registered rho-hat rule | `src/rho_rule.py --seeds 20` | `rho_rule_metrics.json`, `rho_rule.md` |
| Library shift | `src/day2.py`, `src/pilot.py`, `src/pilot2.py`, `src/lib_adapt.py` | `day2_natural_shift.md`, `pilot_anchored.md`, `pilot2_E1.md`, `lib_adapt_run30.log` |
| Perturbation-specific readout and audit | `src/specific.py --seeds 20`, `src/audit_specific.py`, `src/utility.py` | `specific_metrics.json`, `specific.md`, `audit_specific.log`, `utility_metrics.json` |
| Pathway scores (Fig. 4) | `src/pathway.py --seeds 20`, `src/fig_pathway.py` | `pathway_metrics.json`, `pathway_run20.log` |
| Numerical checks of the theory | `src/theory_checks.py` | `theory_checks.json` |
| Paper figures 1-3 | `src/figures_paper.py` | `paper/figs/` |
| Tahoe-100M pre-registered replication (Section 5.5, Appendix D) | registration `prereg/PREREG_tahoe.md` (git 6561021); on a 32-vCPU cloud CPU machine: `pod/tahoe_extract.py OUT` (downloads the CC0 DE table from Hugging Face), `pod/tahoe_compact.py OUT`, `pod/tahoe_pairs.py OUT/compact --pairs all --models core --seeds 20`, `--pairs subset100 --models ladder --seeds 10`, `--pairs all --models core --seeds 20 --dose high`, then `pod/tahoe_summary.py` and `pod/tahoe_posthoc.py` | `tahoe_*_metrics.json`, `tahoe_summary.md` (P1-P6), `tahoe_posthoc.md` (exploratory), `tahoe/` (gene pool, units, lines) |
| STATE retrained per source line (Section 5.1, Appendix E) | on a GPU machine with `pip install arc-state`: `pod/st_prep.py` (Arc's `Replogle-Nadig-Preprint/replogle.h5ad`, GO-SVD features), `pod/st_train.sh SRC SEED STEPS` (seed 0: 20,000 steps; seeds 1-2: 8,000), `state tx predict --predict-only` (set `OMP_NUM_THREADS`), `pod/st_eval.py`, `pod/st_summary.py` | `st_eval_*.json`, `st_summary.txt` |
| Split-half noise floor and error attribution (Section 5.4, Fig. 5, Appendix F) | on a CPU machine: `pod/pseudobulk.py RAW.h5ad OUT.h5ad --parity=0|1` for each line (Arc's `Replogle-Nadig-Preprint` single-cell files), `pod/noise_export.py genes3000.txt noise.npz` -> `data/noise.npz`; then `src/ceiling.py --seeds 20 --readout all|top100|pc10`, `src/fig_attribution.py` | `ceiling_*_metrics.json`, `ceiling_*_run20.log`, `paper/figs/fig5_attribution.pdf` |
| Pretrained gene embeddings as source features (Section 5.1, Appendix F) | GPU: `pod/embed_genes.py targets.tsv emb.npz` (ESM2-650M, Geneformer V2-104M) -> `data/gene_emb.npz`; then `src/embed_eval.py --seeds 10` | `embed_eval_metrics.json` |
| Difficulty transfer (Section 5.5, Appendix F) | `src/difficulty.py --seeds 20` | `difficulty_metrics.json`, `difficulty_run20.log` |
| Transfer signals that did not close the gap (Appendix F) | `src/transfer_measured.py --seeds 20`, `src/decision.py --seeds 3` | `transfer_measured_metrics.json`, `decision_metrics.json` |
| Tahoe zero/few-pilot pooling (Section 5.6, Appendix D) | `pod/tahoe_zeroshot.py OUT --sources 12 --seeds 5` after the Tahoe extraction | `tahoe_zeroshot.md` |

Build the paper with any LaTeX engine, for example `tectonic main.tex` inside `paper/`.

## License

MIT (see `LICENSE`). The datasets are subject to their own terms. GEARS is MIT-licensed and is installed as a
dependency, not redistributed here.
