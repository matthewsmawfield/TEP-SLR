#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.logger import TEPLogger, set_step_logger
from utils.plot_style import apply_paper_style

# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# Set up logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logger = TEPLogger("step_2_4", log_file_path=LOGS_DIR / "step_2_4_plot_results.log")
set_step_logger(logger)

def exponential_decay(d, A, lam, C):
    return A * np.exp(-d / lam) + C

def main():
    logger.info("Starting Step 2.4: Plotting Results")
    apply_paper_style()
    
    # Load MWPC results
    json_path = RESULTS_DIR / "step_2_3_mwpc_analysis.json"
    if not json_path.exists():
        logger.error(f"Error: {json_path} not found")
        return 1

    with open(json_path) as f:
        data = json.load(f)

    # 1. Plot Inter-Station Phase Alignment vs Distance
    # Use interstation_irregular_phase (has data) with fallback to interstation_mwpc
    src = data.get('interstation_irregular_phase') or data.get('interstation_mwpc') or {}
    pairs = src.get('pairs', [])
    if pairs:
        logger.info(f"Plotting phase alignment for {len(pairs)} station pairs")
        df_pairs = pd.DataFrame(pairs)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(df_pairs['baseline_km'], df_pairs['phase_alignment'], alpha=0.7, label='Station Pairs')
        
        # Plot fit if available
        fit = src.get('fit_results', {}).get('phase_alignment')
        if fit:
            dist_range = np.linspace(0, 15000, 100)
            y_fit = exponential_decay(dist_range, fit['amplitude'], fit['lambda_km'], fit['offset'])
            label = f"Fit: λ={fit['lambda_km']:.0f} km"
            plt.plot(dist_range, y_fit, linestyle='-', linewidth=2, color="#4A90C2", label=label)
            
            # Plot GNSS prediction
            # GNSS: A ~ 0.5, lambda = 4200, offset ~ 0
            # We use the fitted amplitude/offset to show the shape comparison
            y_gnss = exponential_decay(dist_range, fit['amplitude'], 4200, fit['offset'])
            plt.plot(dist_range, y_gnss, linestyle='--', linewidth=2, color="#6B73A1", label="GNSS Prediction (λ=4200 km)")
            
        plt.xlabel("Baseline Distance (km)")
        plt.ylabel("Phase Alignment (cos φ)")
        plt.title("SLR Inter-Station Phase Alignment (2015-2025)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        outfile = FIGURES_DIR / "slr_phase_alignment_decay.png"
        plt.tight_layout()
        plt.savefig(outfile)
        logger.success(f"Saved {outfile}")
        plt.close()

    # 2. Plot Pass-Based Correlation vs Distance
    # Prefer the confound-controlled split from Step 2.7: cross-satellite
    # pairings (no shared orbit solution) against matched same-satellite
    # pairs, with the epoch-preserving label-swap floor per bin.  Falls back
    # to the pooled pass_correlations bins when step_2_7 has not run.
    c27_path = RESULTS_DIR / "step_2_7_orbit_commonmode_control.json"
    if c27_path.exists():
        try:
            with open(c27_path) as f:
                c27 = json.load(f)['control_A_cross_satellite_pairing']

            def _bin_arrays(split):
                bins = split['distance_binned']
                mids, means, errs = [], [], []
                for k, v in bins.items():
                    lo, hi = k.replace('km', '').split('-')
                    mids.append((float(lo) + float(hi)) / 2)
                    means.append(v['mean_correlation'])
                    n = v.get('n_pairs', 0)
                    errs.append(v['std_correlation'] / np.sqrt(n) if n > 0 else 0)
                order = np.argsort(mids)
                return (np.array(mids)[order], np.array(means)[order],
                        np.array(errs)[order])

            def _floor_arrays(split):
                fl = split['label_swap_null']['null_bin_means']
                mids, vals = [], []
                for k, v in fl.items():
                    lo, hi = k.replace('km', '').split('-')
                    mids.append((float(lo) + float(hi)) / 2)
                    vals.append(v)
                order = np.argsort(mids)
                return np.array(mids)[order], np.array(vals)[order]

            x_c, m_c, e_c = _bin_arrays(c27['cross_satellite'])
            x_s, m_s, e_s = _bin_arrays(c27['same_satellite'])
            fx_c, fm_c = _floor_arrays(c27['cross_satellite'])
            fx_s, fm_s = _floor_arrays(c27['same_satellite'])
            pv = c27['cross_satellite']['label_swap_null'][
                'p_values_one_sided_le_by_bin'].get('5000-7500km')

            plt.figure(figsize=(10, 6))
            plt.axvspan(5000, 7500, color="#f0e6f5", zorder=0,
                        label="Pre-specified turnover bin")
            plt.axhline(0, color="#495773", linestyle='--', alpha=0.5)
            plt.errorbar(x_c, m_c, yerr=e_c, fmt='o-', capsize=5,
                         linewidth=2, markersize=8, color="#3d1155",
                         label="Cross-satellite pairs", zorder=3)
            plt.errorbar(x_s, m_s, yerr=e_s, fmt='s--', capsize=5,
                         linewidth=1.5, markersize=7, color="#4A90C2",
                         alpha=0.85, label="Same-satellite pairs", zorder=2)
            plt.plot(fx_c, fm_c, 'x:', color="#3d1155", alpha=0.55,
                     linewidth=1, label="Label-swap floor (cross)", zorder=2)
            plt.plot(fx_s, fm_s, '+:', color="#4A90C2", alpha=0.55,
                     linewidth=1, label="Label-swap floor (same)", zorder=2)
            if pv is not None:
                plt.annotate(f"$p={pv:.3f}$ (label-swap)",
                             xy=(6250, -0.2275), xytext=(7000, -0.40),
                             arrowprops=dict(arrowstyle='->',
                                             color="#3d1155", lw=1),
                             fontsize=10, color="#3d1155")
            plt.xlabel("Baseline Distance (km)")
            plt.ylabel("Pass Residual Correlation")
            plt.title("SLR Cross-Satellite Residual Correlation (2022-2025)")
            plt.legend(loc='upper right', fontsize=9)
            plt.grid(True, alpha=0.3)
            outfile = FIGURES_DIR / "slr_pass_correlation_decay.png"
            plt.tight_layout()
            plt.savefig(outfile)
            logger.success(f"Saved {outfile} (confound-controlled split)")
            plt.close()
            logger.success("Plotting complete")
            return
        except Exception as exc:
            logger.warning(
                f"Step 2.7 split figure failed ({exc}); "
                "falling back to pooled bins")

    pass_bins = data['pass_correlations'].get('distance_binned', {})
    if pass_bins:
        logger.info("Plotting pass-based correlations")
        dists = []
        corrs = []
        errs = []

        for k, v in pass_bins.items():
            # Parse range "0-5000km" -> 2500
            try:
                bounds = k.replace('km', '').split('-')
                mid = (float(bounds[0]) + float(bounds[1])) / 2
                dists.append(mid)
                corrs.append(v['mean_correlation'])
                errs.append(v['std_correlation'] / np.sqrt(v['n_pairs']) if v['n_pairs'] > 0 else 0)
            except Exception:
                continue

        plt.figure(figsize=(10, 6))
        plt.errorbar(dists, corrs, yerr=errs, fmt='o-', capsize=5, linewidth=2, markersize=8)
        plt.axhline(0, color="#495773", linestyle='--', alpha=0.5)
        plt.xlabel("Baseline Distance (km)")
        plt.ylabel("Pass Residual Correlation")
        plt.title("SLR Pass-Based Residual Correlation (2022-2025)")
        plt.grid(True, alpha=0.3)
        outfile = FIGURES_DIR / "slr_pass_correlation_decay.png"
        plt.tight_layout()
        plt.savefig(outfile)
        logger.success(f"Saved {outfile}")
        plt.close()

    logger.success("Plotting complete")

if __name__ == "__main__":
    main()
