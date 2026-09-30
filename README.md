# Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18064581.svg)](https://doi.org/10.5281/zenodo.18064581)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-SLR: Satellite Laser Ranging](site/public/twitter-image.jpg)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.6 (Mombasa)  
**First published:** 30 December 2025 · **Last updated:** 30 September 2026
**Status:** Preprint  
**DOI:** [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581)  
**Website:** [https://mlsmawfield.com/tep/slr/](https://mlsmawfield.com/tep/slr/)

## Abstract





An optical-domain consistency test of TEP is presented using 11 years (2015–2025) of Satellite Laser Ranging (SLR) data from passive ILRS geodetic satellites (LAGEOS-1/2, Etalon-1/2, and LARES). This analysis constrains "clock-artifact" explanations by employing two-way optical ranging to passive retroreflectors—a measurement chain orthogonal in hardware and systematic-error class to the microwave atomic-clock chain used in Global Navigation Satellite Systems (GNSS), while probing, under conformal null-cone invariance, the same clock-amplitude channel through its ground-station leg. Frequency-domain analysis is carried out under sampling-matched control. On 5-minute resampled station series the TEP-band (10–500 μHz) mean PSD exceeds the broadband floor (f>1 mHz) by 14.12 ×  (95% CI: 13.55–14.67; N=46 stations); however, with the ILRS observing duty cycle below one percent, the diagnostic is dominated by the resampling kernel rather than by the data. Surrogate processes generated on each station's actual observing grid and passed through the identical resample–interpolate–concatenate pipeline show that white noise alone reproduces the measured ratio (sampling-matched null 13.97 vs observed 14.12; two of 46 stations above p Inter-station pass-correlation analysis under 15-minute contemporaneous binning yields a nominally significant Fisher-combined result (χ²=16.45, 4 d.o.f.; p=0.0025) that concentrates in three same-satellite LAGEOS-2 pairs sharing an orbit arc — orbit-model error enters both stations' residuals as a common mode, and is the controlling confound for any same-satellite pairing. A dedicated control therefore forms the contemporaneous pairings across different satellites in the same window — two stations ranging different targets share no orbit solution, so the channel is closed by construction while the baseline structure is preserved. At the 5,000–7,500 km bin — pre-specified as the first bin entirely beyond the simulated λ_T turnover under either corpus scale (4,201 km GPS-PPP; 1,862 km MGEX) — cross-satellite pairs return r̄=-0.23 over 32 baselines (epoch-preserving label-swap null p=0.046 one-sided; circular-shift synchrony null p ≤ 5 × 10⁻⁴; station-clustered bootstrap 95% CI [-0.43,-0.06]) — though the feature appears in only one of nine threshold/bin-width configurations, falls to r̄=-0.054 when restricted to pairs sharing five or more bins, and alternates sign year to year, so it is reported as a tail event rather than a detection. A daily-aggregation statistic (N=190 pairs, p_FWER=0.020) reverses sign between the two LAGEOS orbit solutions and is retained as exploratory only. Taken together — a spectral bound, an amplitude budget sitting seven or more orders of magnitude above the 10⁻¹⁵–10⁻¹⁸ fractional stabilities of GNSS clock comparisons, and a nominal turnover-bin candidate short of detection strength — the absence of detectable in-band structure in a constellation carrying no onboard clocks and no microwave propagation chain constrains artifact explanations specific to satellite atomic clocks, onboard steering electronics, and ionospheric modeling at microwave frequencies, and demonstrates SLR as an instrumentally and systematically independent measurement of the same conformal clock-amplitude channel as GNSS.


## Key Findings

Analysis of 11 years of SLR data from passive ILRS satellites finds that the apparent 14.12 ×  TEP-band spectral concentration is reproduced by white noise through the identical sparse-sampling pipeline (sampling-matched null 13.97), so the spectral channel is carried as a bound (δA ≲ 2 × 10⁻⁸ in-band coherent amplitude) rather than a detection. The surviving positive evidence is distance-structured: confound-controlled cross-satellite pairings (independent orbit solutions) return r̄=−0.228 at the pre-specified 5,000–7,500 km turnover bin, a nominal one-sided excess under the epoch-preserving label-swap null (p=0.046; circular-shift null p ≤ 5 × 10⁻⁴; bootstrap 95% CI [−0.43,−0.06]), concentrated in the sparsely sampled pairings. A daily-aggregation signal (p_FWER=0.020) is retained as exploratory since its driving bin reverses sign between orbit solutions. SLR uses passive retroreflectors with no active clocks or electronics, so the measurement constrains the conformal clock-amplitude channel through its ground-station leg—an instrumentally and systematically independent check on the GNSS findings rather than an independent physical channel.


---

## The TEP Research Program

| Paper | Repository | Title | DOI |
|-------|-----------|-------|-----|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed | [10.5281/zenodo.16921911](https://doi.org/10.5281/zenodo.16921911) |
| **Paper 1** | [TEP-GNSS](https://github.com/matthewsmawfield/TEP-GNSS) | Global Time Echoes: Distance-Structured Correlations in GNSS Clocks | [10.5281/zenodo.17127229](https://doi.org/10.5281/zenodo.17127229) |
| **Paper 2** | [TEP-GNSS-II](https://github.com/matthewsmawfield/TEP-GNSS-II) | Global Time Echoes: 25-Year Analysis of CODE Precise Clock Products | [10.5281/zenodo.17517141](https://doi.org/10.5281/zenodo.17517141) |
| **Paper 3** | [TEP-GNSS-RINEX](https://github.com/matthewsmawfield/TEP-GNSS-RINEX) | Global Time Echoes: Raw RINEX Consistency Test | [10.5281/zenodo.17860166](https://doi.org/10.5281/zenodo.17860166) |
| **Paper 4** | [TEP-GL](https://github.com/matthewsmawfield/TEP-GL) | Temporal-Spatial Coupling in Gravitational Lensing: A Reinterpretation of Dark Matter Observations | [10.5281/zenodo.17982540](https://doi.org/10.5281/zenodo.17982540) |
| **Paper 5** | [TEP-GTE](https://github.com/matthewsmawfield/TEP-GTE) | Global Time Echoes: Empirical Synthesis | [10.5281/zenodo.18004832](https://doi.org/10.5281/zenodo.18004832) |
| **Paper 6** | [TEP-UCD](https://github.com/matthewsmawfield/TEP-UCD) | Universal Critical Density: Cross-Scale Consistency of ρ_T | [10.5281/zenodo.18064365](https://doi.org/10.5281/zenodo.18064365) |
| **Paper 7** | [TEP-RBH](https://github.com/matthewsmawfield/TEP-RBH) | The Soliton Wake: Exploring RBH-1 as a Temporal Topology Candidate | [10.5281/zenodo.18059250](https://doi.org/10.5281/zenodo.18059250) |
| **Paper 8** | **TEP-SLR** (This repo) | Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging | [10.5281/zenodo.18064581](https://doi.org/10.5281/zenodo.18064581) |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? | [10.5281/zenodo.18109760](https://doi.org/10.5281/zenodo.18109760) |
| **Paper 10** | [TEP-COS](https://github.com/matthewsmawfield/TEP-COS) | The Temporal Equivalence Principle: Suppressed Density Scaling in Globular Cluster Pulsars | [10.5281/zenodo.18165798](https://doi.org/10.5281/zenodo.18165798) |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension | [10.5281/zenodo.18209702](https://doi.org/10.5281/zenodo.18209702) |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | The Temporal Equivalence Principle: A Unified Resolution to the JWST High-Redshift Anomalies | [10.5281/zenodo.19000827](https://doi.org/10.5281/zenodo.19000827) |
| **Paper 13** | [TEP-WB](https://github.com/matthewsmawfield/TEP-WB) | The Temporal Equivalence Principle: Temporal Shear Recovery in Gaia DR3 Wide Binaries | [10.5281/zenodo.19102061](https://doi.org/10.5281/zenodo.19102061) |
| **Paper 15** | [TEP-EFA](https://github.com/matthewsmawfield/TEP-EFA) | Temporal Equivalence Principle: Temporal Shear in the Earth Flyby Anomaly | [10.5281/zenodo.19454862](https://doi.org/10.5281/zenodo.19454862) |
| **Paper 16** | [TEP-J0437](https://github.com/matthewsmawfield/TEP-J0437) | Synchronization Holonomy in Pulsar Scintillation | [10.5281/zenodo.19454620](https://doi.org/10.5281/zenodo.19454620) |
| **Paper 17** | [TEP-LLR](https://github.com/matthewsmawfield/TEP-LLR) | Lunar Laser Ranging and the Nordtvedt Effect | [10.5281/zenodo.19446028](https://doi.org/10.5281/zenodo.19446028) |

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
  title={Global Time Echoes: Optical-Domain Consistency Test via Satellite Laser Ranging},
  author={Smawfield, Matthew Lukin},
  journal={Zenodo},
  year={2025},
  doi={10.5281/zenodo.18064581},
  note={v0.6 (Mombasa)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science—all manuscripts, analysis code, and data products are openly available under Creative Commons Attribution 4.0 International (CC-BY-4.0) to encourage and facilitate replication. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
