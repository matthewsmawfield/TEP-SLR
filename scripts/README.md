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

### Step 2.7: Orbit-Error / Common-Mode Confound Controls

```bash
python3 scripts/steps/step_2_7_orbit_commonmode_control.py
```

Controls the dominant common-mode confound in the pass-bin inter-station
statistic (two stations ranging the *same* satellite share that satellite's
orbit-model error):

- **Cross-satellite pairing**: correlates residuals of stations ranging
  *different* satellites in the same 15-min window — shared orbit error is
  absent by construction, so surviving correlation is station-referenced
  (spatial field) rather than product-referenced.
- **Network common-mode subtraction**: removes the per-day network mean from
  the daily-aggregation series (with the mechanical −1/(N−1) bias disclosed).
- **Per-satellite daily aggregation**: LAGEOS-1 vs LAGEOS-2 splits test
  orbit-solution dependence of the daily-aggregation feature.

Output: `results/outputs/step_2_7_orbit_commonmode_control.json`.

### Step 2.8: Sampling-Matched Coloured-Noise Nulls

```bash
python3 scripts/steps/step_2_8_sampling_matched_nulls.py \
    --n-surrogates 60 --workers 8
```

Supplies the missing controls for the Step 2.3 TEP-band spectral
diagnostic. The published band/broadband ratio is evaluated on a stitched
series (5-minute resample, interpolation limit=2, gaps dropped); with a
sub-percent observing duty cycle the estimator mixes the data's spectrum
with the sampling kernel. The step builds surrogate families — white
noise, station-matched AR(1), canonical flicker 1/f, a power-law grid, a
station-matched power law, and random walk — as continuous processes on
each station's full observing grid, samples them at the actual observation
bins, and passes them through the identical pipeline, so each null prices
the resampling channel exactly.

A second channel evaluates the same families under a real-epoch
Lomb–Scargle estimator (no interpolation or stitching) at the actual
5-minute observation bins, plus the pairwise structure function
SF(tau) over lags spanning 20 minutes to 24 hours, and computes the
delta-A -> residual amplitude map under the timer-rate channel
(delta-R = R * delta-A) against Earth's surface conformal depth
u_earth ~ 6.95e-10.

Result: the observed 14.12x ratio is reproduced by white noise through the
same sampling (null 13.97), every coloured alternative returns a larger
concentration, the real-epoch spectrum is flat (ratio 1.04 vs white null
1.05), and the conformal in-band excursion is bounded at delta-A <~ 2e-8.

Output: `results/outputs/step_2_8_sampling_matched_nulls.json`.

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
- `8-TEP-SLR-v0.5-Mombasa.md` — Generated markdown manuscript
