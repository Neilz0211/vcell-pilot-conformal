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
| `virtual_cell/pod/` | Scripts used on a rented cloud machine: raw single-cell → pseudobulk for HepG2/Jurkat, and parallel GEARS runs |
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
| `goa_human.gaf.gz`, `go-basic.obo` | Gene Ontology (`http://current.geneontology.org`) |

## Environment

Uses [`uv`](https://docs.astral.sh/uv/).

```bash
cd virtual_cell && uv sync          # main analysis (CPU)
cd virtual_cell/gears && uv sync    # GEARS (CUDA 12.8 PyTorch)
```

## Reproducing the paper

All scripts run from `virtual_cell/` as `uv run python src/<script>.py [--seeds N]`. Each writes its JSON and a log
into `results/`.

| Result in the paper | Script | Summary |
|---|---|---|
| Transfer matrix, controls (Table 1, Fig. 1b) | `offset_pairs.py` (K562/RPE1 pairs and same-line controls; HepG2/Jurkat pairs run on the cloud machine with `pod/run_analysis.sh`) | `transfer_matrix.md`, `offset_pairs.md`, `nadig_pairs.md` |
| Model ladder (Fig. 1a) | `ladder.py` | `ladder_metrics.json` |
| Calibration-transfer decomposition, same-family path, drop-RPE1 | `decomp.py` | `decomp_run20.log` |
| Tuned GEARS | `gears/gears_ctx.py` | `gears_ctx.md` |
| Adapt-then-calibrate, Prop. A ball | `adapt.py`, `offset_pairs.py` | `adapt_idea1.md`, `offset_pairs.md` |
| Paired budget comparison, leak-free preprocessing | `paired.py --prep sourcetrain` | `paired.md` |
| Pre-registered ρ̂ adapt rule | `rho_rule.py` | `rho_rule.md` |
| Library shift | `day2.py`, `pilot.py`, `pilot2.py`, `lib_adapt.py` | `day2_natural_shift.md`, `pilot_anchored.md`, `pilot2_E1.md` |
| Perturbation-specific readout and its audit | `specific.py`, `audit_specific.py` | `specific.md` |
| Equivalent pilot budget | `eq_pilots.py` | `eq_pilots.md` |
| Numerical checks of the theory | `theory_checks.py` | `theory_checks.json` |
| Paper figures | `figures_paper.py` | `paper/figs/` |

Build the paper with any LaTeX engine, for example `tectonic main.tex` inside `paper/`.

## License

MIT (see `LICENSE`). The datasets are subject to their own terms. GEARS is MIT-licensed and is installed as a
dependency, not redistributed here.
