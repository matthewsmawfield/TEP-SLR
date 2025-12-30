#!/usr/bin/env python3
"""
Step 2.5: Generate Enhanced Manuscript Figures

This script generates improved versions of:
- Figure 3.1: Phase alignment with binned averages and shaded anti-correlation region
- Figure 3.2: Proper autocorrelation vs elevation (not raw scatter)
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.logger import TEPLogger, set_step_logger
from utils.plot_style import apply_paper_style

# Paths
RESULTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"
CSV_PATH = RESULTS_DIR / "step_2_1_slr_residuals.csv"

# Set up logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logger = TEPLogger("step_2_5", log_file_path=LOGS_DIR / "step_2_5_enhanced_figures.log")
set_step_logger(logger)


def exponential_decay(d, A, lam, C):
    """Exponential decay function for fitting."""
    return A * np.exp(-d / lam) + C


def compute_elevation_autocorrelation(df, elevation_bins, time_window_hours=1):
    """
    Compute residual autocorrelation as a function of elevation.
    
    For each elevation bin, we compute the temporal autocorrelation of residuals
    within short time windows, then average across all windows.
    
    This measures how "coherent" the residuals are at different path lengths.
    """
    results = []
    
    for i in range(len(elevation_bins) - 1):
        el_low, el_high = elevation_bins[i], elevation_bins[i + 1]
        el_mid = (el_low + el_high) / 2
        
        # Filter to this elevation bin
        mask = (df['elevation_deg'] >= el_low) & (df['elevation_deg'] < el_high)
        df_bin = df[mask].copy()
        
        if len(df_bin) < 100:
            continue
        
        # Group by station and compute lag-1 autocorrelation within each station
        autocorrs = []
        for station, grp in df_bin.groupby('station'):
            if len(grp) < 10:
                continue
            grp = grp.sort_values('epoch_utc')
            residuals = grp['residual_mm'].values
            
            # Compute lag-1 autocorrelation
            if len(residuals) > 1:
                r = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                if not np.isnan(r):
                    autocorrs.append(r)
        
        if len(autocorrs) >= 3:
            mean_ac = np.mean(autocorrs)
            std_ac = np.std(autocorrs) / np.sqrt(len(autocorrs))
            results.append({
                'elevation_mid': el_mid,
                'autocorrelation': mean_ac,
                'std_error': std_ac,
                'n_stations': len(autocorrs)
            })
    
    return pd.DataFrame(results)


def compute_residual_variance_by_elevation(df, elevation_bins):
    """
    Compute residual RMS/variance as a function of elevation.
    Higher variance at low elevation indicates path-accumulated effects.
    """
    results = []
    
    for i in range(len(elevation_bins) - 1):
        el_low, el_high = elevation_bins[i], elevation_bins[i + 1]
        el_mid = (el_low + el_high) / 2
        
        mask = (df['elevation_deg'] >= el_low) & (df['elevation_deg'] < el_high)
        residuals = df.loc[mask, 'residual_mm'].values
        
        if len(residuals) < 50:
            continue
        
        rms = np.sqrt(np.mean(residuals**2))
        std_err = rms / np.sqrt(2 * len(residuals))  # Approximate SE for RMS
        
        results.append({
            'elevation_mid': el_mid,
            'rms_mm': rms,
            'std_error': std_err,
            'n_obs': len(residuals)
        })
    
    return pd.DataFrame(results)


def plot_enhanced_phase_alignment():
    """
    Generate enhanced Figure 3.1 with:
    - Raw scatter points (faded)
    - Binned averages with error bars
    - Shaded anti-correlation region (2000-4000 km)
    - Fit curves
    """
    logger.info("Generating enhanced phase alignment figure (Figure 3.1)")
    
    # Load MWPC results
    json_path = RESULTS_DIR / "step_2_3_mwpc_analysis.json"
    if not json_path.exists():
        logger.error(f"Error: {json_path} not found")
        return False
    
    with open(json_path) as f:
        data = json.load(f)
    
    pairs = data['interstation_mwpc'].get('pairs', [])
    if not pairs:
        logger.error("No station pairs found in MWPC data")
        return False
    
    df_pairs = pd.DataFrame(pairs)
    logger.info(f"Loaded {len(df_pairs)} station pairs")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Shade anti-correlation region (2000-4000 km)
    ax.axvspan(2000, 4000, alpha=0.15, color='#6B73A1', label='Anti-correlation region')
    
    # 2. Plot raw scatter (faded)
    ax.scatter(df_pairs['baseline_km'], df_pairs['phase_alignment'], 
               alpha=0.3, s=40, color='#220126', zorder=2)
    
    # 3. Compute and plot binned averages with error bars
    bin_edges = [0, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 15000, 20000]
    bin_centers = []
    bin_means = []
    bin_errors = []
    
    for i in range(len(bin_edges) - 1):
        mask = (df_pairs['baseline_km'] >= bin_edges[i]) & (df_pairs['baseline_km'] < bin_edges[i + 1])
        vals = df_pairs.loc[mask, 'phase_alignment']
        if len(vals) >= 2:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_means.append(vals.mean())
            bin_errors.append(vals.std() / np.sqrt(len(vals)))
    
    ax.errorbar(bin_centers, bin_means, yerr=bin_errors, 
                fmt='s-', color='#4A90C2', linewidth=2, markersize=8,
                capsize=5, capthick=2, label='Binned mean ± SE', zorder=4)
    
    # 4. Plot fit curves
    fit = data['interstation_mwpc'].get('fit_results', {}).get('phase_alignment')
    if fit:
        dist_range = np.linspace(0, 18000, 200)
        y_fit = exponential_decay(dist_range, fit['amplitude'], fit['lambda_km'], fit['offset'])
        ax.plot(dist_range, y_fit, linestyle='-', linewidth=2.5, 
                color='#2D0140', label=f"SLR Fit: λ={fit['lambda_km']:.0f} km", zorder=3)
        
        # GNSS prediction for comparison
        y_gnss = exponential_decay(dist_range, 0.9, 4200, 0)
        ax.plot(dist_range, y_gnss, linestyle='--', linewidth=2, 
                color='#495773', alpha=0.7, label="GNSS Reference (λ=4200 km)", zorder=3)
    
    # 5. Add zero line
    ax.axhline(0, color='#495773', linestyle=':', alpha=0.5, linewidth=1)
    
    # Labels and styling
    ax.set_xlabel("Baseline Distance (km)")
    ax.set_ylabel("Phase Alignment (cos φ)")
    ax.set_title("SLR Inter-Station Phase Alignment (2015–2025)")
    ax.set_xlim(0, 19000)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    ax.grid(True, alpha=0.3)
    
    # Save
    outfile = FIGURES_DIR / "slr_phase_alignment_decay.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    logger.success(f"Saved {outfile}")
    plt.close()
    
    return True


def plot_elevation_coherence():
    """
    Generate proper Figure 3.2 showing residual coherence vs elevation/range.
    
    Uses the pre-computed range_coherence data from MWPC analysis which shows
    autocorrelation as a function of signal path length (range to satellite).
    """
    logger.info("Generating elevation coherence figure (Figure 3.2)")
    
    # Load MWPC results which contain pre-computed range coherence
    json_path = RESULTS_DIR / "step_2_3_mwpc_analysis.json"
    if not json_path.exists():
        logger.error(f"Error: {json_path} not found")
        return False
    
    with open(json_path) as f:
        data = json.load(f)
    
    range_coherence = data.get('range_coherence', {}).get('range_bins', [])
    if not range_coherence:
        logger.error("No range_coherence data found in MWPC results")
        return False
    
    df_rc = pd.DataFrame(range_coherence)
    logger.info(f"Loaded {len(df_rc)} range bins from MWPC analysis")
    
    # Convert range to approximate elevation angle
    # For LAGEOS at ~12,270 km altitude, range varies from ~6000 km (zenith) to ~13000 km (horizon)
    # Approximate: elevation ≈ arcsin((h + R_earth) / range) where h=12270km, R=6371km
    LAGEOS_ALT = 12270  # km
    R_EARTH = 6371  # km
    df_rc['elevation_approx'] = np.degrees(np.arcsin(
        np.clip((LAGEOS_ALT + R_EARTH - df_rc['range_km']) / (2 * LAGEOS_ALT), -1, 1)
    ))
    # Simpler approximation: map range linearly to elevation
    # 6000 km → ~90°, 9000 km → ~10°
    df_rc['elevation_approx'] = 90 - (df_rc['range_km'] - 6000) / (9000 - 6000) * 80
    df_rc['elevation_approx'] = df_rc['elevation_approx'].clip(10, 90)
    
    # Create figure with dual x-axis (elevation and range)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot autocorrelation vs range
    ax1.plot(df_rc['range_km'], df_rc['autocorrelation'], 
             'o-', color='#220126', linewidth=2.5, markersize=10)
    
    # Fill area under curve to emphasize the increase
    ax1.fill_between(df_rc['range_km'], 0, df_rc['autocorrelation'], 
                     alpha=0.2, color='#4A90C2')
    
    # Calculate and annotate the ratio
    high_el_ac = df_rc[df_rc['range_km'] < 6500]['autocorrelation'].mean()
    low_el_ac = df_rc[df_rc['range_km'] > 8000]['autocorrelation'].mean()
    ratio = low_el_ac / high_el_ac if high_el_ac > 0 else 0
    
    ax1.annotate(f'Low/High elevation ratio: {ratio:.1f}×', 
                xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#495773'))
    
    # Add horizontal reference lines
    ax1.axhline(high_el_ac, color='#495773', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(low_el_ac, color='#495773', linestyle='--', alpha=0.5, linewidth=1)
    
    # Annotate the reference lines
    ax1.annotate(f'High el: {high_el_ac:.3f}', xy=(6100, high_el_ac + 0.01), 
                fontsize=9, color='#495773')
    ax1.annotate(f'Low el: {low_el_ac:.3f}', xy=(8600, low_el_ac + 0.01), 
                fontsize=9, color='#495773')
    
    # Labels and styling
    ax1.set_xlabel("Signal Path Length / Range to Satellite (km)")
    ax1.set_ylabel("Residual Autocorrelation")
    ax1.set_title("SLR Residual Coherence vs Path Length (2015–2025)")
    ax1.set_ylim(0, 0.35)
    ax1.grid(True, alpha=0.3)
    
    # Add secondary x-axis for approximate elevation
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    # Set tick positions at range values, label with approximate elevations
    range_ticks = [6000, 6500, 7000, 7500, 8000, 8500, 9000]
    elev_labels = ['~90°', '~70°', '~50°', '~35°', '~25°', '~15°', '~10°']
    ax2.set_xticks(range_ticks)
    ax2.set_xticklabels(elev_labels)
    ax2.set_xlabel("Approximate Elevation Angle")
    
    # Add physics annotation
    ax1.annotate('High elevation\n(short path)', 
                xy=(6200, 0.08), fontsize=9, color='#495773', ha='center')
    ax1.annotate('Low elevation\n(long path through atmosphere)', 
                xy=(8500, 0.20), fontsize=9, color='#495773', ha='center')
    
    # Save
    outfile = FIGURES_DIR / "slr_residual_vs_elevation_full.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    logger.success(f"Saved {outfile}")
    plt.close()
    
    return True


def main():
    logger.info("Starting Step 2.5: Enhanced Manuscript Figures")
    apply_paper_style()
    
    success = True
    
    # Generate enhanced Figure 3.1
    if not plot_enhanced_phase_alignment():
        success = False
    
    # Generate proper Figure 3.2
    if not plot_elevation_coherence():
        success = False
    
    if success:
        logger.success("All enhanced figures generated successfully")
    else:
        logger.warning("Some figures could not be generated")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
