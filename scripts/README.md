# TEP-SLR Analysis Pipeline

## Overview

The TEP-SLR pipeline downloads Satellite Laser Ranging (SLR) data from CDDIS,
computes range residuals using SP3 precise orbits, and performs
Magnitude-Weighted Phase Correlation (MWPC) analysis to test for TEP
conformal-sector signatures in the optical domain.

## Prerequisites

1. Python 3.10+
2. Dependencies: `pip install -r requirements.txt`
3. CDDIS/Earthdata credentials (for data download only):
   - `~/.netrc` with `machine urs.earthdata.nasa.gov`, or
   - `export CDDIS_USER` and `export CDDIS_PASS`

## Quick Start

```bash
cd TEP-SLR
bash reproduce_analysis.sh
```

This runs the full pipeline: download, residuals, MWPC, figures, simulation,
and site build.

## Pipeline Steps

### Step 1.0: Data Acquisition

```bash
python3 scripts/steps/step_1_0_data_acquisition.py \
    --start 2015-01-01 --end 2022-12-31 \
    --data-type npt --crd-version crd --source allsat --workers 10

python3 scripts/steps/step_1_0_data_acquisition.py \
    --start 2022-01-01 --end 2025-12-31 \
    --data-type npt --crd-version crd_v2 --source allsat --workers 10
```

Downloads NP2/NPT normal-point files from CDDIS. Two CRD format versions are
used:
- CRD v1 (`npt_crd`): 2015-2022
- CRD v2 (`npt_crd_v2`): 2022-2025

2022 appears in both (transition year). Total: ~91,000 files, ~15 GB.

### Step 2.1: Residual Calculation

```bash
python3 scripts/helpers/process_residuals_yearly.py
```

Processes residuals year-by-year (2015-2025) to avoid memory issues. For each
year, calls `step_2_1_slr_residuals.py` which:
- Downloads SP3 precise orbits from CDDIS
- Downloads SLRF2020 station coordinates (SINEX)
- Computes range residuals with full corrections (troposphere, Shapiro,
  Sagnac, solid Earth tide, ocean loading, center-of-mass)
- Saves yearly CSVs, then merges into master CSV

### Step 2.2: Summary

```bash
python3 scripts/helpers/generate_full_summary.py
```

Generates summary statistics for the full dataset.

### Step 2.3: MWPC Analysis

```bash
python3 scripts/steps/step_2_3_mwpc_analysis.py
```

Performs Magnitude-Weighted Phase Correlation analysis:
- Pass-based inter-station correlations (per-satellite, distance-binned)
- Daily-aggregation inter-station correlations
- Irregular-sampling phase coherence
- Range-dependent lag-1 coherence
- Spectral concentration in the TEP band (10-500 µHz)
- Family-wise permutation null tests
- Threshold sweep robustness (0.3m, 0.5m, 1.0m)
- Year-block stability

### Step 2.4-2.5: Figures

```bash
python3 scripts/steps/step_2_4_plot_results.py
python3 scripts/steps/step_2_5_enhanced_figures.py
```

### Step 3.0: Anti-Echo Simulation

```bash
python3 scripts/steps/step_3_0_sim_antiecho.py
```

### Step 4.0: Build Site

```bash
cd site && npm run build
```

Builds the static publication site and generates the markdown manuscript.

## Output Locations

- `results/outputs/` — JSON analysis outputs
- `results/figures/` — PNG figures
- `site/dist/` — Built static site
- `8-TEP-SLR-v0.4-Mombasa.md` — Generated markdown manuscript
