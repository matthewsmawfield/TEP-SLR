#!/bin/bash
set -e

# TEP-SLR Reproduction Script
# ---------------------------
# This script runs the full analysis pipeline for the TEP-SLR paper.
# Pre-requisites:
#   1. Python 3.10+ installed
#   2. Dependencies installed: pip install -r requirements.txt
#   3. CDDIS Credentials (for Step 1 only):
#      - ~/.netrc file with machine urs.earthdata.nasa.gov
#      - OR export CDDIS_USER and CDDIS_PASS environment variables

# Date range for the full 11-year analysis
START_DATE="2015-01-01"
END_DATE="2025-12-31"

echo "============================================================"
echo "TEP-SLR Analysis Pipeline (${START_DATE} to ${END_DATE})"
echo "============================================================"

# Step 1.0: Data Acquisition (auto-download if missing)
NEEDS_DOWNLOAD=false
if [ ! -d "data/slr/npt_crd_v2/allsat" ] && [ ! -d "data/slr/npt_crd/allsat" ]; then
    NEEDS_DOWNLOAD=true
else
    # Check if we have data for the full date range (at least one file per year)
    for year in $(seq 2015 2025); do
        if [ ! -d "data/slr/npt_crd_v2/allsat/${year}" ] && [ ! -d "data/slr/npt_crd/allsat/${year}" ]; then
            echo "[!] Missing data for year ${year}"
            NEEDS_DOWNLOAD=true
            break
        fi
    done
fi

if [ "$NEEDS_DOWNLOAD" = true ]; then
    echo -e "\n[Step 1.0] Downloading SLR observation data from CDDIS..."
    echo "    Date range: ${START_DATE} to ${END_DATE}"
    echo "    This requires CDDIS/Earthdata credentials (~/.netrc or CDDIS_USER/CDDIS_PASS)."
    echo "    Downloading ~91,000 NP2/NPT files (~1.4 GB per year, ~15 GB total)..."
    echo ""
    echo "    CDDIS uses two CRD format versions:"
    echo "      - CRD v1 (npt_crd): 2015-2022"
    echo "      - CRD v2 (npt_crd_v2): 2022-2025"
    echo "    Both are downloaded; 2022 appears in both (transition year)."
    echo ""

    # Download CRD v1 data (2015-2022)
    echo "    [1/2] Downloading CRD v1 (npt_crd) for 2015-2022..."
    python3 scripts/steps/step_1_0_data_acquisition.py \
        --start "2015-01-01" \
        --end "2022-12-31" \
        --data-type npt \
        --crd-version crd \
        --source allsat \
        --workers 10

    # Download CRD v2 data (2022-2025)
    echo "    [2/2] Downloading CRD v2 (npt_crd_v2) for 2022-2025..."
    python3 scripts/steps/step_1_0_data_acquisition.py \
        --start "2022-01-01" \
        --end "${END_DATE}" \
        --data-type npt \
        --crd-version crd_v2 \
        --source allsat \
        --workers 10
else
    echo "[+] SLR observation data found for full date range."
fi

# Step 2.1: Residual Calculation (Year-by-Year)
echo -e "\n[Step 2.1] Processing Residuals (2015-2025)..."
python3 scripts/helpers/process_residuals_yearly.py

# Step 2.2: Generate Full Dataset Summary
echo -e "\n[Step 2.2] Generating Full Dataset Summary..."
python3 scripts/helpers/generate_full_summary.py

# Step 2.3: MWPC Analysis
echo -e "\n[Step 2.3] Running Magnitude-Weighted Phase Correlation Analysis..."
python3 scripts/steps/step_2_3_mwpc_analysis.py

# Step 2.4: Plotting
echo -e "\n[Step 2.4] Generating Figures..."
python3 scripts/steps/step_2_4_plot_results.py

# Step 2.5: Enhanced Figures
echo -e "\n[Step 2.5] Generating Enhanced Figures..."
python3 scripts/steps/step_2_5_enhanced_figures.py

# Step 2.7: Orbit-Error / Common-Mode Confound Controls
echo -e "\n[Step 2.7] Running orbit-error / common-mode confound controls..."
python3 scripts/steps/step_2_7_orbit_commonmode_control.py

# Step 2.8: Sampling-Matched Coloured-Noise Nulls
echo -e "\n[Step 2.8] Running sampling-matched coloured-noise nulls..."
python3 scripts/steps/step_2_8_sampling_matched_nulls.py --n-surrogates 60

# Step 3.0: Simulation
echo -e "\n[Step 3.0] Running Anti-Echo Simulation..."
python3 scripts/steps/step_3_0_sim_antiecho.py

# Step 4.0: Build Site (Publication)
echo -e "\n[Step 4.0] Building Publication Site..."
node site/build.js

echo -e "\n[+] Analysis Complete!"
echo "    Results: results/outputs/"
echo "    Figures: results/figures/"
echo "    Website: site/dist/index.html"
