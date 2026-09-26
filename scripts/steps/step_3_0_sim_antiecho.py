#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.plot_style import apply_paper_style

def generate_tep_field(n_stations=50, size_km=10000, correlation_length_km=4200):
    """
    Generate a spatially correlated conformal-sector Temporal-Topology delay
    field: the common-mode propagation delay produced by the A(phi) coupling
    on two-way optical transits, correlated on the Temporal Topology scale
    lambda_T ~ 4,200 km.
    """
    # Random station locations
    x = np.random.uniform(-size_km/2, size_km/2, n_stations)
    y = np.random.uniform(-size_km/2, size_km/2, n_stations)
    coords = np.column_stack((x, y))
    
    # Generate correlated noise (Cholesky decomposition)
    dists = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2))
    cov = np.exp(-dists / correlation_length_km)
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(n_stations))
    
    # True TEP delay (e.g., mean 100mm, variation 50mm)
    u = np.random.randn(n_stations)
    tep_delays = 100 + 50 * L @ u  # Monopole ~100mm, Dipole ~50mm
    
    return coords, tep_delays

def simulate_orbit_fit(tep_delays):
    """
    Simulate Dynamic Orbit Determination.
    The orbit fit absorbs the 'mean' (monopole) delay as a radial scaling.
    """
    # The orbit determination engine sees: Observed_Range = True_Range + TEP_Delay
    # It fits an orbit parameters (Orbit_Range) to minimize residuals.
    # Simplified: Fit a single 'bias' parameter b representing the orbit scale error absorption.
    # Minimize sum((TEP_Delay - b)^2)
    # The 'b' will effectively be the mean of TEP_Delay.
    
    absorbed_monopole = np.mean(tep_delays)
    
    # The residuals are what's left
    post_fit_residuals = tep_delays - absorbed_monopole
    
    return absorbed_monopole, post_fit_residuals

def analyze_correlations(coords, residuals):
    """
    Compute spatial correlation of residuals.
    """
    n = len(residuals)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            # Product of residuals (covariance proxy)
            # Normalized correlation would be better, but raw product shows the sign
            corr = residuals[i] * residuals[j] 
            pairs.append((dist, corr))
    
    pairs = np.array(pairs)
    return pairs

def bin_pairs(all_pairs, bins):
    """Mean residual-product per distance bin."""
    corrs = []
    for k in range(len(bins)-1):
        mask = (all_pairs[:,0] >= bins[k]) & (all_pairs[:,0] < bins[k+1])
        corrs.append(np.mean(all_pairs[mask, 1]) if np.sum(mask) > 0 else np.nan)
    return np.array(corrs)


def zero_crossing(bin_centers, corrs):
    """First distance at which the binned correlation crosses zero."""
    for k in range(len(corrs)-1):
        if corrs[k] > 0 and corrs[k+1] <= 0:
            return float(bin_centers[k])
    return None


def main():
    print("Running Anti-Echo Simulation...")
    apply_paper_style()
    
    np.random.seed(42)
    
    all_pairs_dyn = []
    all_pairs_kin = []
    
    # Monte Carlo simulation
    for _ in range(100):
        coords, tep_true = generate_tep_field(n_stations=50)
        
        # 1. Kinematic case (GNSS-like): receiver state solved epoch-by-epoch;
        # the orbit is an external fixed product, so the common mode is NOT
        # absorbed into a shared network constraint. The delay maps into the
        # per-station clock solutions, so the analyzed residuals retain the
        # field fluctuation about its ensemble (monopole) level and remain
        # positively correlated at all baselines.
        pairs_kin = analyze_correlations(coords, tep_true - 100.0)
        all_pairs_kin.append(pairs_kin)
        
        # 2. Dynamic case (SLR-like): the multi-arc orbit fit absorbs the
        # common-mode monopole into the fitted orbital scale, so post-fit
        # residuals carry the field's deviations from the absorbed mean and
        # anticorrelate once the baseline exceeds the turnover set by the
        # field correlation length and the network extent.
        monopole, residuals = simulate_orbit_fit(tep_true)
        
        pairs = analyze_correlations(coords, residuals)
        all_pairs_dyn.append(pairs)
        
    all_pairs_dyn = np.vstack(all_pairs_dyn)
    all_pairs_kin = np.vstack(all_pairs_kin)
    
    # Binning
    bins = np.linspace(0, 10000, 20)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    corrs = bin_pairs(all_pairs_dyn, bins)
    corrs_kin = bin_pairs(all_pairs_kin, bins)

    # Anchor-robustness scan: the corpus carries two measured lambda_T scales —
    # the GPS-PPP value 4,201 km and the MGEX combined-product value 1,862 km —
    # which differ because the fitted correlation length is product- and
    # metric-dependent (Paper 14). The SLR test is the turnover's location, so
    # the simulation is rerun across both anchors (and beyond) to check whether
    # the predicted turnover bin changes.
    anchor_lambdas = [1000, 1500, 1862, 3000, 4201, 8000]
    anchor_scan = {}
    for lam in anchor_lambdas:
        np.random.seed(42)
        scan_pairs = []
        for _ in range(100):
            coords_s, tep_s = generate_tep_field(n_stations=50, correlation_length_km=lam)
            _, res_s = simulate_orbit_fit(tep_s)
            scan_pairs.append(analyze_correlations(coords_s, res_s))
        scan_pairs = np.vstack(scan_pairs)
        scan_corrs = bin_pairs(scan_pairs, bins)
        anchor_scan[str(lam)] = {
            'dynamic_zero_crossing_km': zero_crossing(bin_centers, scan_corrs),
            'dynamic_correlations_normalized': (scan_corrs / np.nanmax(np.abs(scan_corrs))).tolist(),
        }

    # Normalize for plotting (each curve to its own max absolute value)
    corrs_norm = corrs / np.nanmax(np.abs(corrs))
    corrs_kin_norm = corrs_kin / np.nanmax(np.abs(corrs_kin))
    turnover_km = zero_crossing(bin_centers, corrs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, corrs_norm, 'o-', linewidth=2, color="#2D0140", label='Dynamic orbit fit (SLR)')
    plt.plot(bin_centers, corrs_kin_norm, 's--', linewidth=2, color="#495773", label='Kinematic solution (GNSS)')
    plt.axhline(0, color="#495773", linestyle='--', alpha=0.5)
    plt.axvline(4200, color="#8a7ca8", linestyle=':', alpha=0.7, label=r'$\lambda_T \approx 4{,}200$ km')
    plt.xlabel('Distance (km)')
    plt.ylabel('Correlation (Normalized)')
    plt.title('Anti-Echo: Conformal Common-Mode Field Through Two Estimators')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    output_file = PROJECT_ROOT / 'results' / 'figures' / 'sim_antiecho_proof.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved proof plot to {output_file}")
    
    # Save simulation results as JSON for manuscript traceability
    results = {
        'n_monte_carlo': 100,
        'n_stations': 50,
        'correlation_length_km': 4200,
        'size_km': 10000,
        'field_model': 'conformal Temporal-Topology common-mode delay field (no disformal term)',
        'bin_centers_km': bin_centers.tolist(),
        'dynamic_correlations_normalized': corrs_norm.tolist(),
        'dynamic_correlations_raw': corrs.tolist(),
        'kinematic_correlations_normalized': corrs_kin_norm.tolist(),
        'kinematic_correlations_raw': corrs_kin.tolist(),
        'dynamic_zero_crossing_km': turnover_km,
        'anchor_scan_lambda_km': anchor_lambdas,
        'anchor_scan': anchor_scan,
        'n_pairs_total': int(len(all_pairs_dyn)),
    }
    json_output = PROJECT_ROOT / 'results' / 'outputs' / 'step_3_0_sim_antiecho.json'
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved simulation results to {json_output}")

if __name__ == "__main__":
    main()
