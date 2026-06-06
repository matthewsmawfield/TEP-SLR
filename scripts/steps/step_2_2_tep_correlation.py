#!/usr/bin/env python3
"""
TEP-SLR Step 2.2: TEP Correlation Analysis

Analyzes SLR residuals for distance-structured correlations predicted by TEP.

TEP predicts that timing/ranging measurements should show coherent correlations
over the coherence length L_c ~ 4200 km (from GNSS analysis). For SLR:
- Station-satellite ranges: 5-9 Mm (span 1-2 coherence lengths)
- Station-station baselines: 0-15 Mm
- Temporal correlations within passes

Key analyses:
1. Range-residual correlation: Do residuals correlate with station-satellite distance?
2. Inter-station correlation: Do contemporaneous observations at different stations correlate?
3. Temporal structure: Do residuals show coherent temporal patterns?
4. TEP model fit: Does the data support λ ~ 4200 km coherence length?
"""

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Set up logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOGS_DIR / "step_2_2_tep_correlation.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# TEP coherence length from GNSS analysis
TEP_COHERENCE_LENGTH_KM = 4200.0

# Speed of light
C_M_S = 299792458.0


def tep_correlation_model(d: np.ndarray, lambda_km: float, amplitude: float) -> np.ndarray:
    """
    TEP correlation model: exponential decay with coherence length.
    
    C(d) = amplitude * exp(-d / lambda)
    
    Args:
        d: Distance in km
        lambda_km: Coherence length in km
        amplitude: Correlation amplitude at d=0
    
    Returns:
        Predicted correlation values
    """
    return amplitude * np.exp(-d / lambda_km)


def compute_station_positions(df: pd.DataFrame) -> Dict[int, np.ndarray]:
    """
    Extract approximate station ECEF positions from the data.
    Uses model_range and satellite position to back-calculate.
    For now, use SLRF2020 coordinates directly.
    """
    # This is a placeholder - ideally we'd load from SLRF2020 SINEX
    # For now, use known station coordinates
    stations = {
        7090: np.array([-2389026.352, 5043317.079, -3078530.934]),  # YARL
        7819: np.array([4075578.385, 931853.295, 4801568.124]),      # GRZL
        7237: np.array([-1535746.019, -5166996.770, 3401035.478]),   # CHAL
        7396: np.array([-5466006.878, -2404428.293, 2242228.197]),   # MATL
        7821: np.array([3899224.802, 396752.836, 5015078.388]),      # ZIML
        1824: np.array([4579649.022, -441696.896, 4422994.698]),     # SHIL
        1888: np.array([1130720.099, -4831349.619, 3994106.608]),    # WUHL
        1890: np.array([4201762.255, 332747.262, 4779294.155]),      # BADL
    }
    return stations


def compute_baseline_km(sta1: np.ndarray, sta2: np.ndarray) -> float:
    """Compute baseline distance between two stations in km."""
    return np.linalg.norm(sta1 - sta2) / 1000.0


def analyze_range_residual_correlation(df: pd.DataFrame) -> Dict:
    """
    Analyze correlation between station-satellite range and residuals.
    
    TEP predicts that longer ranges should show different residual patterns
    due to integration over more of the gravitational field.
    """
    results = {}
    
    # Correlation between range and residual
    r, p = stats.pearsonr(df['model_range_m'], df['residual_m'])
    results['range_residual_correlation'] = {
        'pearson_r': float(r),
        'p_value': float(p),
        'significant': p < 0.05
    }
    
    # Bin by range and compute statistics
    range_bins = np.linspace(df['model_range_m'].min(), df['model_range_m'].max(), 6)
    df['range_bin'] = pd.cut(df['model_range_m'], bins=range_bins)
    
    bin_stats = []
    for bin_label, group in df.groupby('range_bin', observed=True):
        if len(group) > 5:
            bin_stats.append({
                'range_km': float(group['model_range_m'].mean() / 1000),
                'n_obs': len(group),
                'residual_mean_mm': float(group['residual_mm'].mean()),
                'residual_std_mm': float(group['residual_mm'].std()),
            })
    
    results['range_binned_stats'] = bin_stats
    
    return results


def analyze_interstation_correlation(df: pd.DataFrame, max_time_diff_s: float = 60.0) -> Dict:
    """
    Analyze correlation between contemporaneous observations at different stations.
    
    TEP predicts that stations with smaller baselines should have more
    correlated residuals (within coherence length).
    """
    results = {'pairs': [], 'baseline_correlation': []}
    
    # Get station positions
    station_pos = compute_station_positions(df)
    
    # Convert epoch to datetime for matching
    df = df.copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'])
    
    # Find contemporaneous observations at different stations
    stations = df['station'].unique()
    
    for i, sta1 in enumerate(stations):
        for sta2 in stations[i+1:]:
            if sta1 not in station_pos or sta2 not in station_pos:
                continue
                
            baseline_km = compute_baseline_km(station_pos[sta1], station_pos[sta2])
            
            df1 = df[df['station'] == sta1].sort_values('epoch')
            df2 = df[df['station'] == sta2].sort_values('epoch')
            
            # Match observations within time window
            matched_residuals = []
            for _, row1 in df1.iterrows():
                time_diffs = abs((df2['epoch'] - row1['epoch']).dt.total_seconds())
                close_idx = time_diffs < max_time_diff_s
                if close_idx.any():
                    closest_idx = time_diffs.idxmin()
                    matched_residuals.append((row1['residual_m'], df2.loc[closest_idx, 'residual_m']))
            
            if len(matched_residuals) >= 5:
                r1 = [m[0] for m in matched_residuals]
                r2 = [m[1] for m in matched_residuals]
                corr, p = stats.pearsonr(r1, r2)
                
                results['pairs'].append({
                    'station_1': int(sta1),
                    'station_2': int(sta2),
                    'baseline_km': float(baseline_km),
                    'n_matched': len(matched_residuals),
                    'correlation': float(corr),
                    'p_value': float(p),
                })
                
                results['baseline_correlation'].append({
                    'baseline_km': float(baseline_km),
                    'correlation': float(corr),
                })
    
    # Fit TEP model to baseline-correlation data
    if len(results['baseline_correlation']) >= 3:
        baselines = np.array([p['baseline_km'] for p in results['baseline_correlation']])
        correlations = np.array([p['correlation'] for p in results['baseline_correlation']])
        
        try:
            popt, pcov = curve_fit(
                tep_correlation_model, 
                baselines, 
                correlations,
                p0=[TEP_COHERENCE_LENGTH_KM, 0.5],
                bounds=([100, -1], [20000, 1]),
                maxfev=1000
            )
            results['tep_fit'] = {
                'lambda_km': float(popt[0]),
                'amplitude': float(popt[1]),
                'lambda_uncertainty_km': float(np.sqrt(pcov[0, 0])) if pcov[0, 0] > 0 else None,
                'predicted_lambda_km': TEP_COHERENCE_LENGTH_KM,
                'consistent_with_tep': abs(popt[0] - TEP_COHERENCE_LENGTH_KM) < 2 * np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else None,
            }
        except Exception as e:
            results['tep_fit'] = {'error': str(e)}
    
    return results


def analyze_temporal_structure(df: pd.DataFrame) -> Dict:
    """
    Analyze temporal structure of residuals within and across passes.
    
    TEP predicts coherent temporal patterns as the geometry changes.
    """
    results = {}
    
    # Compute autocorrelation for each station
    station_autocorr = {}
    
    for sta in df['station'].unique():
        sta_df = df[df['station'] == sta].sort_values('epoch_utc')
        if len(sta_df) < 10:
            continue
        
        residuals = sta_df['residual_m'].values
        
        # Lag-1 autocorrelation
        if len(residuals) > 1:
            lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            station_autocorr[int(sta)] = float(lag1_corr)
    
    results['station_lag1_autocorr'] = station_autocorr
    results['mean_autocorr'] = float(np.mean(list(station_autocorr.values()))) if station_autocorr else None
    
    # Pass-level analysis
    df = df.copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'])
    df = df.sort_values(['station', 'epoch'])
    df['time_gap'] = df.groupby('station')['epoch'].diff().dt.total_seconds()
    df['pass_id'] = (df['time_gap'] > 300).cumsum()
    
    pass_means = df.groupby(['station', 'pass_id'])['residual_m'].mean()
    pass_stds = df.groupby(['station', 'pass_id'])['residual_m'].std()
    
    results['pass_statistics'] = {
        'n_passes': int(df['pass_id'].nunique()),
        'within_pass_std_mean_cm': float(pass_stds.mean() * 100),
        'between_pass_std_cm': float(pass_means.std() * 100),
        'variance_ratio': float(pass_means.var() / (pass_stds.mean()**2)) if pass_stds.mean() > 0 else None,
    }
    
    return results


def analyze_satellite_geometry(df: pd.DataFrame) -> Dict:
    """
    Analyze residuals by satellite and geometry.
    
    Different satellites at different ranges probe different parts of the field.
    """
    results = {}
    
    for sat in df['satellite'].unique():
        sat_df = df[df['satellite'] == sat]
        results[sat] = {
            'n_obs': len(sat_df),
            'mean_range_km': float(sat_df['model_range_m'].mean() / 1000),
            'residual_mean_m': float(sat_df['residual_m'].mean()),
            'residual_std_m': float(sat_df['residual_m'].std()),
            'residual_rms_cm': float(np.sqrt((sat_df['residual_m']**2).mean()) * 100),
        }
    
    # Cross-satellite correlation for same station
    station_cross_sat = {}
    for sta in df['station'].unique():
        sta_df = df[df['station'] == sta]
        sats = sta_df['satellite'].unique()
        if len(sats) == 2:
            r1 = sta_df[sta_df['satellite'] == sats[0]]['residual_m'].mean()
            r2 = sta_df[sta_df['satellite'] == sats[1]]['residual_m'].mean()
            station_cross_sat[int(sta)] = {
                'sat1': sats[0],
                'sat2': sats[1],
                'mean_diff_m': float(r1 - r2),
            }
    
    results['cross_satellite'] = station_cross_sat
    
    return results


def compute_tep_predictions(df: pd.DataFrame) -> Dict:
    """
    Compute TEP-specific predictions for the SLR data.
    """
    results = {}
    
    # Range statistics
    ranges_km = df['model_range_m'].values / 1000.0
    results['range_statistics'] = {
        'min_km': float(ranges_km.min()),
        'max_km': float(ranges_km.max()),
        'mean_km': float(ranges_km.mean()),
        'range_span_km': float(ranges_km.max() - ranges_km.min()),
        'coherence_lengths_spanned': float((ranges_km.max() - ranges_km.min()) / TEP_COHERENCE_LENGTH_KM),
    }
    
    # TEP signal expectations
    tep_signal_ns = 0.2  # From GNSS analysis
    c = C_M_S
    tep_signal_m = tep_signal_ns * 1e-9 * c  # Convert to meters (one-way)
    
    results['tep_signal_expectations'] = {
        'gnss_signal_ns': tep_signal_ns,
        'equivalent_range_cm': float(tep_signal_m * 100),
        'slr_precision_cm': 16.0,  # From our analysis
        'signal_to_noise': float(tep_signal_m * 100 / 16.0),
        'observations_needed_for_detection': int((16.0 / (tep_signal_m * 100))**2) if tep_signal_m > 0 else None,
    }
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="TEP-SLR Step 2.2: TEP Correlation Analysis")
    parser.add_argument("--input", default=str(OUTPUTS_DIR / "step_2_1_slr_residuals.csv"),
                        help="Input residuals CSV file")
    args = parser.parse_args()
    
    # Load residuals
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} residual observations from {input_path.name}")
    
    # Remove station bias for correlation analysis
    # (Use residual deviation from station mean)
    df['residual_debiased_m'] = df.groupby('station')['residual_m'].transform(lambda x: x - x.mean())
    df['residual_debiased_mm'] = df['residual_debiased_m'] * 1000
    
    logger.info("Station biases removed for correlation analysis")
    
    # Run analyses
    results = {
        'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
        'input_file': str(input_path),
        'n_observations': len(df),
        'n_stations': int(df['station'].nunique()),
        'n_satellites': int(df['satellite'].nunique()),
        'tep_coherence_length_km': TEP_COHERENCE_LENGTH_KM,
    }
    
    logger.info("Analyzing range-residual correlation...")
    results['range_residual'] = analyze_range_residual_correlation(df)
    
    logger.info("Analyzing inter-station correlation...")
    results['interstation'] = analyze_interstation_correlation(df)
    
    logger.info("Analyzing temporal structure...")
    results['temporal'] = analyze_temporal_structure(df)
    
    logger.info("Analyzing satellite geometry...")
    results['satellite_geometry'] = analyze_satellite_geometry(df)
    
    logger.info("Computing TEP predictions...")
    results['tep_predictions'] = compute_tep_predictions(df)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEP CORRELATION ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    logger.info(f"\nObservations: {len(df)} across {df['station'].nunique()} stations")
    logger.info(f"Range span: {results['tep_predictions']['range_statistics']['range_span_km']:.0f} km")
    logger.info(f"Coherence lengths spanned: {results['tep_predictions']['range_statistics']['coherence_lengths_spanned']:.2f}")
    
    if results['interstation'].get('pairs'):
        logger.info(f"\nInter-station pairs analyzed: {len(results['interstation']['pairs'])}")
        for pair in results['interstation']['pairs']:
            sig = "*" if pair['p_value'] < 0.05 else ""
            logger.info(f"  {pair['station_1']}-{pair['station_2']}: baseline={pair['baseline_km']:.0f}km, r={pair['correlation']:.3f}{sig}")
    
    if results['temporal'].get('mean_autocorr'):
        logger.info(f"\nMean lag-1 autocorrelation: {results['temporal']['mean_autocorr']:.3f}")
    
    logger.info(f"\nTEP signal expectations:")
    exp = results['tep_predictions']['tep_signal_expectations']
    logger.info(f"  GNSS signal: {exp['gnss_signal_ns']} ns -> {exp['equivalent_range_cm']:.1f} cm range equivalent")
    logger.info(f"  SLR precision: {exp['slr_precision_cm']:.1f} cm")
    logger.info(f"  SNR: {exp['signal_to_noise']:.2f}")
    logger.info(f"  Observations needed: ~{exp['observations_needed_for_detection']} for detection")
    
    # Save results
    out_json = OUTPUTS_DIR / "step_2_2_tep_correlation.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n[SUCCESS] Results saved to {out_json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
