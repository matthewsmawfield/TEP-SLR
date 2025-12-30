
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.plot_style import apply_paper_style

def generate_tep_field(n_stations=50, size_km=10000, correlation_length_km=3000):
    """
    Generate a spatially correlated TEP delay field (scalar field phi).
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

def main():
    print("Running Anti-Echo Simulation...")
    apply_paper_style()
    
    all_pairs = []
    
    # Monte Carlo simulation
    for _ in range(100):
        coords, tep_true = generate_tep_field(n_stations=50)
        
        # 1. Kinematic case (GNSS-like): Position solved epoch-by-epoch
        # Common mode is NOT absorbed into orbit (orbit is fixed external product)
        # Residuals ~ True TEP (plus measurement noise, ignored here)
        # But here we focus on the SLR dynamic case.
        
        # 2. Dynamic case (SLR-like): Orbit absorbs mean
        monopole, residuals = simulate_orbit_fit(tep_true)
        
        pairs = analyze_correlations(coords, residuals)
        all_pairs.append(pairs)
        
    all_pairs = np.vstack(all_pairs)
    
    # Binning
    bins = np.linspace(0, 10000, 20)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    corrs = []
    
    for k in range(len(bins)-1):
        mask = (all_pairs[:,0] >= bins[k]) & (all_pairs[:,0] < bins[k+1])
        if np.sum(mask) > 0:
            # Pearson correlation coefficient calculation for the bin?
            # Or just mean product?
            # To get a proper correlation coefficient [-1, 1], we need to normalize.
            # But the sign of the mean product tells us if it's correlated or anti-correlated.
            
            val = np.mean(all_pairs[mask, 1])
            corrs.append(val)
        else:
            corrs.append(np.nan)
            
    # Normalize for plotting (max absolute value to 1)
    corrs = np.array(corrs)
    corrs_norm = corrs / np.max(np.abs(corrs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, corrs_norm, 'o-', linewidth=2, color="#2D0140", label='Simulated SLR Residuals')
    plt.axhline(0, color="#495773", linestyle='--', alpha=0.5)
    plt.xlabel('Distance (km)')
    plt.ylabel('Correlation (Normalized)')
    plt.title('The Anti-Echo Effect: Simulation of Dynamic Orbit Fit')
    plt.grid(True, alpha=0.3)
    plt.legend()
    # plt.text(1000, -0.5, "Short-Range Anti-Correlation\n(Due to Monopole Absorption)", fontsize=12)
    
    output_file = 'results/figures/sim_antiecho_proof.png'
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved proof plot to {output_file}")

if __name__ == "__main__":
    main()
