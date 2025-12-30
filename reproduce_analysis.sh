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

echo "============================================================"
echo "TEP-SLR Analysis Pipeline (2015–2025)"
echo "============================================================"

# Check for data
if [ ! -d "data/slr/npt_crd_v2" ] && [ ! -d "data/slr/npt_crd" ]; then
    echo "[!] SLR observation data not found."
    echo "    Run Step 1 to download data (requires CDDIS credentials)."
    echo "    Command: python scripts/steps/step_1_0_data_acquisition.py --start 2015-01-01 --end 2025-12-31"
    read -p "    Skip download and proceed? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "[+] SLR observation data found."
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
