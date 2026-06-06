# Global Time Echoes: Optical Validation of the Temporal Equivalence Principle via Satellite Laser Ranging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18064582.svg)](https://doi.org/10.5281/zenodo.18064582)

![TEP-SLR: Satellite Laser Ranging](site/public/twitter-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.1 (Mombasa)  
**Date:** 30 December 2025  
**DOI:** [10.5281/zenodo.18064582](https://doi.org/10.5281/zenodo.18064582)  
**Website:** [https://matthewsmawfield.github.io/TEP-SLR/](https://matthewsmawfield.github.io/TEP-SLR/)

## Abstract

Independent optical-domain validation of the Temporal Equivalence Principle (TEP) is presented using 11 years (2015–2025) of Satellite Laser Ranging (SLR) data from passive ILRS geodetic satellites (LAGEOS-1/2 and Etalon-1/2). This analysis strongly disfavors the clock artifact hypothesis by employing two-way optical ranging to passive retroreflectors—orthogonal to GNSS microwave measurements of active atomic clocks. Analysis of 192,561 high-precision residuals (|Δρ|<0.5 m) reveals three signatures consistent with TEP's conformal sector: (1) **Path-length dependence**—a gap-aware, 5-minute-binned lag-1 residual statistic differs substantially between short-path and long-path geometry, with a low/high-range ratio of 6.58; (2) **Spectral concentration**—station-averaged power in the predicted TEP band (10–500 μHz) exceeds the full-spectrum mean by 2.48× (95% CI: 2.46–2.50) and exceeds the broadband floor (f>1 mHz) by 14.00× (95% CI: 13.53–14.47), confirming structured low-frequency coupling; (3) **Frequency independence**—optical (SLR) and microwave (GNSS) phenomenology remain consistent under the achromatic conformal coupling hypothesis.

The detection of matching low-frequency structure in a system with no active clocks, no microwave propagation, and purely optical two-way ranging renders receiver electronics, clock steering, and ionospheric modeling errors highly improbable as alternative explanations. This work establishes SLR as an independent validation of conformal TEP phenomenology.

## Key Findings

- **Range-Dependent Coherence:** Path-length dependence with 6.58× ratio between low and high elevation (qualitatively robust, quantitatively estimator-sensitive)
- **Spectral Concentration (TEP Band):** 2.48× vs full-spectrum mean (95% CI: 2.46–2.50), 14.00× vs broadband floor (95% CI: 13.53–14.47)
- **Methodology Independence:** Confirms the signal exists in two-way optical ranging, ruling out GNSS-specific processing artifacts

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Theory | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Multi-Center Validation | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | 25-Year Longitudinal Analysis | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Raw RINEX Validation | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Gravitational Lensing | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | **TEP-SLR** (This repo) | SLR Validation | [10.5281/zenodo.18064582](https://doi.org/10.5281/zenodo.18064582) |
| **Paper 6** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Synthesis Manuscript | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 7** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density | [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) |
| **Paper 8** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake | [10.5281/zenodo.18059251](https://doi.org/10.5281/zenodo.18059251) |

## Repository Structure

```
TEP-SLR/
├── scripts/
│   ├── steps/                  # Core analysis pipeline
│   │   ├── step_1_0...py       # CDDIS Data Downloader
│   │   ├── step_2_1...py       # Residual Calculation
│   │   ├── step_2_3...py       # MWPC Analysis (Main)
│   │   ├── step_2_4...py       # Plotting
│   │   └── step_3_0_sim_antiecho.py       # Anti-Echo Simulation
│   └── helpers/                # Utility scripts
│       ├── download_orbits.py  # SP3 Orbit Downloader
│       └── process_residuals_yearly.py   # Batch processing helper
├── data/                       # Input data (GitIgnored)
│   └── slr/                    # CRD observations & SP3 orbits
├── results/
│   ├── outputs/                # Analysis JSONs & CSVs
│   └── figures/                # Generated plots
├── logs/                       # Execution logs
└── reproduce_analysis.sh       # One-click reproduction script
```

## Quick Start

### 1. Prerequisites
- Python 3.10+
- [CDDIS Account](https://cddis.nasa.gov/) (for data download only)

```bash
pip install -r requirements.txt
```

### 2. Reproduction
To run the full analysis pipeline (assuming data is downloaded):

```bash
chmod +x reproduce_analysis.sh
./reproduce_analysis.sh
```

### 3. Data Access

**Option A: Use Pre-Processed Results (Recommended for Verification)**
All analysis outputs are included in `results/outputs/` and `results/figures/`. You can verify the analysis without downloading raw data:

```bash
# View analysis results
cat results/outputs/step_2_3_mwpc_analysis.json

# Regenerate figures from existing data
python scripts/steps/step_2_4_plot_results.py
```

**Option B: Download Raw Data from CDDIS (For Full Reproduction)**
To download SLR observations and orbits from NASA CDDIS:

1. **Register for NASA Earthdata Account:**
   - Visit: https://urs.earthdata.nasa.gov/users/new
   - Create free account (required for CDDIS access)

2. **Configure Authentication:**
   
   Option 1 - Using `.netrc` file (recommended):
   ```bash
   echo "machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD" >> ~/.netrc
   chmod 600 ~/.netrc
   ```
   
   Option 2 - Using environment variables:
   ```bash
   export CDDIS_USER="your_username"
   export CDDIS_PASS="your_password"
   ```

3. **Download Data:**
   ```bash
   # Download SLR observations (2015-2025)
   python scripts/steps/step_1_0_data_acquisition.py --start 2015-01-01 --end 2025-12-31
   
   # Download precise orbits
   for y in $(seq 2015 2025); do python scripts/helpers/download_orbits.py --year $y; done
   ```

### 4. Pipeline Steps

1.  **Data Acquisition (`step_1_0`):** Downloads CRD (Normal Point) observation files from CDDIS.
2.  **Orbit Processing (`download_orbits`):** Fetches precise SP3 orbits for LAGEOS-1 and LAGEOS-2.
3.  **Residual Calculation (`step_2_1`):** Computes range residuals (Observed - Computed) using rigorous force models.
4.  **MWPC Analysis (`step_2_3`):** Performs Magnitude-Weighted Phase Correlation analysis to extract spatial decay signatures.
5.  **Visualization (`step_2_4`):** Generates decay plots and diagnostic figures.
6.  **Simulation (`step_3_0`):** Runs the "Anti-Echo" Monte Carlo simulation to validate the sign inversion mechanism.

## License

This project is licensed under Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Citation

If you use this code or data, please cite:

```bibtex
@article{smawfield2025slr,
  title={Global Time Echoes: Optical Validation of the Temporal Equivalence Principle via Satellite Laser Ranging},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.18064582},
  note={v0.1 (Mombasa)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons Attribution 4.0 International (CC-BY-4.0) to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthewsmawfield@gmail.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
