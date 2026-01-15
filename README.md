# Global Time Echoes: Optical Validation of the Temporal Equivalence Principle via Satellite Laser Ranging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18064582.svg)](https://doi.org/10.5281/zenodo.18064582)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-SLR: Satellite Laser Ranging](site/public/twitter-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.1 (Mombasa)  
**Date:** 30 December 2025  
**Status:** Preprint  
**DOI:** [10.5281/zenodo.18064582](https://doi.org/10.5281/zenodo.18064582)  
**Website:** [https://mlsmawfield.com/tep/slr/](https://mlsmawfield.com/tep/slr/)

## Abstract

An independent, optical-domain test of the Temporal Equivalence Principle (TEP) is presented using 11 years (2015–2025) of Satellite Laser Ranging (SLR) data from passive ILRS geodetic satellites (LAGEOS-1/2 and Etalon-1/2). This analysis constrains "clock-artifact" explanations by employing two-way optical ranging to passive retroreflectors—a methodology orthogonal to the microwave measurements of active atomic clocks used in Global Navigation Satellite Systems (GNSS).

Under strict 5-minute contemporaneous binning, distance-binned mean pass-correlations fluctuate with high variance. However, widening the overlap window to 15 minutes (thereby increasing multi-station overlap) reveals statistically significant, distance-structured inter-station correlations (Fisher-combined $\chi^2=15.35$ with 4 d.o.f.; $p=0.0040$) under a family-wise circular-shift test.

This signal is driven primarily by LAGEOS-2 ($p=0.0005$), which exhibits a strong negative correlation ($r \approx -0.59$) in the 5,000–7,500 km distance bin, whereas LAGEOS-1 remains consistent with the null hypothesis ($p \approx 0.93$). Although observation counts and temporal overlap are comparable, this asymmetry likely reflects a combination of orbital geometry—LAGEOS-2's prograde $52.6^\circ$ orbit versus LAGEOS-1's retrograde $109.8^\circ$ orbit—and small-number statistics in the critical distance bin.

To validate this finding with more robust statistics, a daily-aggregation analysis ($N=190$ station pairs) was performed. This confirmed a subtler but statistically significant negative correlation at shorter ranges (500–1,000 km, $p=0.017$), suggesting a persistent global background structure independent of the high-amplitude LAGEOS-2 events.

The detection of matching low-frequency structure in a system devoid of active clocks and microwave propagation challenges receiver electronics, clock steering, and ionospheric modeling errors as complete explanations. While current network sparsity limits testing to the conformal sector, this work demonstrates SLR as an independent, technology-orthogonal line of evidence for TEP phenomenology.

## Summary of Key Results and Findings

### Primary Results Table

| Metric | Value | Uncertainty | Significance |
|--------|-------|-------------|--------------|
| **Dataset Coverage** | 11 years | 2015–2025 | LAGEOS-1/2, Etalon-1/2 |
| **Methodology** | Two-way optical ranging | Passive retroreflectors | Independent of GNSS |
| **Inter-Station p-value (15-min)** | p = 0.0040 | Fisher χ² = 15.35, 4 d.o.f. | Significant |
| **Daily Aggregation p-value** | p = 0.017 | N = 190 station pairs | Significant |

### Range-Dependent Coherence

| Observable | Value | 95% CI | Interpretation |
|------------|-------|--------|----------------|
| **Elevation Ratio (Low/High)** | 6.58× | — | Path-length dependence |
| **TEP Band Concentration** | 2.48× | 2.46–2.50 | vs full-spectrum mean |
| **Broadband Floor Ratio** | 14.00× | 13.53–14.47 | Spectral specificity |

### Satellite-Specific Results

| Satellite | Orbital Inclination | p-value | Key Finding |
|-----------|---------------------|---------|-------------|
| **LAGEOS-2** | 52.6° (prograde) | p = 0.0005 | Strong detection (r ≈ −0.59 at 5,000–7,500 km) |
| **LAGEOS-1** | 109.8° (retrograde) | p ≈ 0.93 | Null (geometry/averaging effects) |

### Spatial Coherence by Distance Bin

| Distance Bin | Correlation | p-value | Notes |
|--------------|-------------|---------|-------|
| **500–1,000 km** | r ≈ −0.027 | p = 0.017 | Daily aggregation |
| **5,000–7,500 km** | r ≈ −0.59 | p = 0.0005 | LAGEOS-2 dominant |

### Key Validation

| Test | Result | Interpretation |
|------|--------|----------------|
| **Clock Artifact Exclusion** | ✓ | Passive retroreflectors (no active clocks) |
| **Microwave Independence** | ✓ | Optical domain only |
| **Technology Orthogonality** | ✓ | Different from GNSS methodology |

### Key Interpretation

SLR provides a critical independent test of TEP because it uses passive retroreflectors with no active clocks or electronics—eliminating receiver artifacts as an explanation. The detection of distance-structured correlations in optical two-way ranging, completely independent of GNSS microwave methodology, suggests the signal is a property of spacetime rather than instrumentation. The LAGEOS-2/LAGEOS-1 asymmetry (prograde vs retrograde orbits) hints at velocity-dependent effects consistent with TEP's predictions. This technology-orthogonal confirmation strengthens the case that GNSS findings reflect genuine physical phenomena.

---

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Temporal Evolution of Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Validation of Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Synthesis** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Validation of the Temporal Equivalence Principle | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 7** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Unifying Atomic, Galactic, and Compact Object Scales | [10.5281/zenodo.18064366](https://doi.org/10.5281/zenodo.18064366) |
| **Paper 8** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: A Runaway Black Hole as a Gravitational Soliton | [10.5281/zenodo.18059251](https://doi.org/10.5281/zenodo.18059251) |
| **Paper 9** | **TEP-SLR** (This repo) | Global Time Echoes: Optical Validation of the Temporal Equivalence Principle via Satellite Laser Ranging | [10.5281/zenodo.18064582](https://doi.org/10.5281/zenodo.18064582) |
| **Paper 10** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109761](https://doi.org/10.5281/zenodo.18109761) |

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
