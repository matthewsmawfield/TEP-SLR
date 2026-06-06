#!/usr/bin/env python3
"""
TEP-SLR Step 2.3: Magnitude-Weighted Phase Correlation Analysis

Implements the same MWPC methodology used in TEP-GNSS-RINEX for SLR residuals.

Methodology (identical to TEP-GNSS):
1. Cross-spectral density (CSD) via Welch's method
2. TEP frequency band: 10-500 µHz  
3. Magnitude-weighted circular phase averaging
4. Phase alignment = cos(weighted_phase)
5. Distance-binned coherence analysis

For SLR, we analyze:
- Inter-station residual correlations (like GNSS station pairs)
- Range-binned residual patterns within passes
- Temporal coherence structure
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
from scipy.signal import csd, welch, detrend, coherence
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
        logging.FileHandler(LOGS_DIR / "step_2_3_mwpc_analysis.log", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# TEP parameters (from GNSS analysis)
TEP_COHERENCE_LENGTH_KM = 4200.0
F1_HZ = 10e-6   # 10 µHz (TEP band lower limit)
F2_HZ = 500e-6  # 500 µHz (TEP band upper limit)
FS_HZ = 1/300   # 5-minute sampling (matching GNSS)

# Distance bins for analysis
DISTANCE_BINS_KM = [0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000]

# Station coordinates (SLRF2020)
STATION_COORDS = {
    7090: {'name': 'YARL', 'lat': -29.047, 'lon': 115.347, 'ecef': [-2389026.352, 5043317.079, -3078530.934]},
    7819: {'name': 'GRZL', 'lat': 47.068, 'lon': 15.493, 'ecef': [4075578.385, 931853.295, 4801568.124]},
    7237: {'name': 'CHAL', 'lat': 32.424, 'lon': -106.916, 'ecef': [-1535746.019, -5166996.770, 3401035.478]},
    7396: {'name': 'MATL', 'lat': 20.707, 'lon': -156.257, 'ecef': [-5466006.878, -2404428.293, 2242228.197]},
    7821: {'name': 'ZIML', 'lat': 52.279, 'lon': 10.450, 'ecef': [3899224.802, 396752.836, 5015078.388]},
    1824: {'name': 'SHIL', 'lat': 44.143, 'lon': 12.874, 'ecef': [4579649.022, -441696.896, 4422994.698]},
    1888: {'name': 'WUHL', 'lat': 39.018, 'lon': -76.827, 'ecef': [1130720.099, -4831349.619, 3994106.608]},
    1890: {'name': 'BADL', 'lat': 49.146, 'lon': 12.877, 'ecef': [4201762.255, 332747.262, 4779294.155]},
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def compute_mwpc(series1: np.ndarray, series2: np.ndarray, 
                 fs: float = FS_HZ, f1: float = F1_HZ, f2: float = F2_HZ) -> Dict:
    """
    Compute Magnitude-Weighted Phase Correlation (MWPC).
    
    IDENTICAL to TEP-GNSS-RINEX methodology:
    1. Linear detrend
    2. Cross-spectral density via Welch
    3. TEP frequency band selection
    4. Magnitude-weighted circular phase averaging
    5. Phase alignment = cos(weighted_phase)
    
    Args:
        series1, series2: Time series to correlate
        fs: Sampling frequency in Hz
        f1, f2: TEP frequency band limits in Hz
    
    Returns:
        Dict with coherence, phase_alignment, weighted_phase
    """
    n_points = len(series1)
    if n_points < 32:
        return {'coherence': np.nan, 'phase_alignment': np.nan, 'weighted_phase': np.nan}
    
    # STEP 1: Linear detrend
    series1_d = detrend(series1, type='linear')
    series2_d = detrend(series2, type='linear')
    
    # STEP 2: Compute CSD and auto-spectra via Welch
    nperseg = min(256, n_points // 2)
    if nperseg < 16:
        return {'coherence': np.nan, 'phase_alignment': np.nan, 'weighted_phase': np.nan}
    
    try:
        frequencies, Pxy = csd(series1_d, series2_d, fs=fs, nperseg=nperseg, detrend='constant')
        _, Pxx = welch(series1_d, fs=fs, nperseg=nperseg, detrend='constant')
        _, Pyy = welch(series2_d, fs=fs, nperseg=nperseg, detrend='constant')
    except Exception:
        return {'coherence': np.nan, 'phase_alignment': np.nan, 'weighted_phase': np.nan}
    
    # STEP 3: TEP frequency band selection
    band_mask = (frequencies > 0) & (frequencies >= f1) & (frequencies <= f2)
    if not np.any(band_mask):
        # If no frequencies in TEP band, use all positive frequencies
        band_mask = frequencies > 0
    
    if not np.any(band_mask):
        return {'coherence': np.nan, 'phase_alignment': np.nan, 'weighted_phase': np.nan}
    
    # STEP 4: Normalized coherence = |Pxy|² / (Pxx × Pyy)
    Pxy_band = Pxy[band_mask]
    Pxx_band = Pxx[band_mask]
    Pyy_band = Pyy[band_mask]
    
    denom = Pxx_band * Pyy_band
    valid_mask = denom > 0
    if not np.any(valid_mask):
        return {'coherence': np.nan, 'phase_alignment': np.nan, 'weighted_phase': np.nan}
    
    # Magnitude Squared Coherence (MSC)
    coh_squared = np.abs(Pxy_band[valid_mask])**2 / denom[valid_mask]
    magnitudes = np.sqrt(coh_squared)  # Coherence (0-1)
    
    # Phase from cross-spectrum
    phases = np.angle(Pxy_band[valid_mask])
    
    if len(magnitudes) == 0 or np.sum(magnitudes) == 0:
        return {'coherence': np.nan, 'phase_alignment': np.nan, 'weighted_phase': np.nan}
    
    # STEP 5: Magnitude-weighted circular phase averaging
    complex_phases = np.exp(1j * phases)
    weighted_complex = np.average(complex_phases, weights=magnitudes)
    weighted_phase = np.angle(weighted_complex)
    
    # Phase alignment = cos(weighted_phase)
    phase_alignment = np.cos(weighted_phase)
    
    # Mean coherence in band
    mean_coherence = float(np.mean(magnitudes))
    
    return {
        'coherence': mean_coherence,
        'phase_alignment': float(phase_alignment),
        'weighted_phase': float(weighted_phase),
    }


def exponential_decay(d: np.ndarray, amplitude: float, lambda_km: float, offset: float) -> np.ndarray:
    """TEP coherence model: C(d) = A * exp(-d/λ) + C."""
    return amplitude * np.exp(-d / lambda_km) + offset


def build_station_time_series(df: pd.DataFrame, station: int, 
                              resample_interval: str = '5min') -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a regular time series of residuals for a station.
    
    For MWPC, we need regularly-sampled time series. SLR observations
    are sparse, so we resample with interpolation.
    
    Returns:
        times: Unix timestamps
        residuals: Debiased residuals in meters
    """
    sta_df = df[df['station'] == station].copy()
    if len(sta_df) < 10:
        return np.array([]), np.array([])
    
    sta_df['epoch'] = pd.to_datetime(sta_df['epoch_utc'])
    sta_df = sta_df.sort_values('epoch')
    sta_df = sta_df.set_index('epoch')
    
    # Remove station bias
    sta_df['residual_debiased'] = sta_df['residual_m'] - sta_df['residual_m'].mean()
    
    # Resample to regular grid
    resampled = sta_df['residual_debiased'].resample(resample_interval).mean()
    resampled = resampled.interpolate(method='linear', limit=2)  # Fill small gaps
    resampled = resampled.dropna()
    
    if len(resampled) < 10:
        return np.array([]), np.array([])
    
    times = np.array([t.timestamp() for t in resampled.index])
    residuals = resampled.values
    
    return times, residuals


def analyze_pass_correlations(df: pd.DataFrame) -> Dict:
    """
    Analyze correlations between contemporaneous passes at different stations.
    
    This is more suitable for sparse SLR data than continuous time series MWPC.
    We compute correlations between residual patterns when multiple stations
    observe the same satellite pass.
    """
    logger.info("Analyzing pass-based correlations...")
    
    results = {
        'pass_pairs': [],
        'distance_binned': {},
    }
    
    df = df.copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'])
    
    # Group by approximate time windows (1-hour bins)
    df['time_bin'] = df['epoch'].dt.floor('1h')
    
    # For each time bin, find stations with observations
    for time_bin, group in df.groupby('time_bin'):
        stations_in_bin = group['station'].unique()
        
        if len(stations_in_bin) < 2:
            continue
        
        # Compare all station pairs in this time bin
        for i, sta1 in enumerate(stations_in_bin):
            for sta2 in stations_in_bin[i+1:]:
                if sta1 not in STATION_COORDS or sta2 not in STATION_COORDS:
                    continue
                
                coord1 = STATION_COORDS[sta1]
                coord2 = STATION_COORDS[sta2]
                baseline_km = haversine_distance_km(
                    coord1['lat'], coord1['lon'],
                    coord2['lat'], coord2['lon']
                )
                
                # Get residuals for each station in this bin
                r1 = group[group['station'] == sta1]['residual_m'].values
                r2 = group[group['station'] == sta2]['residual_m'].values
                
                # Remove station means (debias)
                r1_debiased = r1 - r1.mean()
                r2_debiased = r2 - r2.mean()
                
                # Compute mean residual for each station (single value per pass)
                mean1 = r1_debiased.mean()
                mean2 = r2_debiased.mean()
                
                results['pass_pairs'].append({
                    'time_bin': str(time_bin),
                    'station_1': int(sta1),
                    'station_2': int(sta2),
                    'baseline_km': float(baseline_km),
                    'n_obs_1': len(r1),
                    'n_obs_2': len(r2),
                    'mean_residual_1_mm': float(mean1 * 1000),
                    'mean_residual_2_mm': float(mean2 * 1000),
                    'product_mm2': float(mean1 * mean2 * 1e6),  # For correlation
                })
    
    # Aggregate by station pair
    if results['pass_pairs']:
        pair_df = pd.DataFrame(results['pass_pairs'])
        
        # Compute correlation for each unique station pair
        pair_correlations = []
        for (sta1, sta2), pair_data in pair_df.groupby(['station_1', 'station_2']):
            if len(pair_data) >= 3:
                # Pearson correlation of pass means
                r1_means = pair_data['mean_residual_1_mm'].values
                r2_means = pair_data['mean_residual_2_mm'].values
                
                if np.std(r1_means) > 0 and np.std(r2_means) > 0:
                    corr = np.corrcoef(r1_means, r2_means)[0, 1]
                    baseline = pair_data['baseline_km'].iloc[0]
                    
                    pair_correlations.append({
                        'station_1': int(sta1),
                        'station_2': int(sta2),
                        'baseline_km': float(baseline),
                        'n_passes': len(pair_data),
                        'correlation': float(corr),
                    })
        
        results['pair_correlations'] = pair_correlations
        
        # Bin by distance
        if pair_correlations:
            corr_df = pd.DataFrame(pair_correlations)
            for d_lo, d_hi in [(0, 5000), (5000, 10000), (10000, 15000)]:
                bin_data = corr_df[(corr_df['baseline_km'] >= d_lo) & (corr_df['baseline_km'] < d_hi)]
                if len(bin_data) > 0:
                    results['distance_binned'][f"{d_lo}-{d_hi}km"] = {
                        'n_pairs': len(bin_data),
                        'mean_correlation': float(bin_data['correlation'].mean()),
                        'std_correlation': float(bin_data['correlation'].std()) if len(bin_data) > 1 else 0,
                    }
    
    logger.info(f"  Found {len(results['pass_pairs'])} pass-pair observations")
    if 'pair_correlations' in results:
        logger.info(f"  Computed correlations for {len(results['pair_correlations'])} station pairs")
    
    return results


def analyze_interstation_mwpc(df: pd.DataFrame) -> Dict:
    """
    Compute MWPC for all station pairs.
    
    This mirrors the TEP-GNSS methodology:
    - Build time series for each station
    - Compute MWPC for all pairs
    - Bin by baseline distance
    - Fit exponential decay to extract coherence length
    """
    logger.info("Computing inter-station MWPC...")
    
    results = {
        'pairs': [],
        'distance_binned': {},
        'fit_results': {},
    }
    
    stations = df['station'].unique()
    station_series = {}
    
    # Build time series for each station
    for sta in stations:
        if sta not in STATION_COORDS:
            continue
        times, residuals = build_station_time_series(df, sta, resample_interval='5min')
        if len(residuals) >= 20:
            station_series[sta] = {'times': times, 'residuals': residuals}
            logger.info(f"  Station {sta}: {len(residuals)} resampled points")
    
    if len(station_series) < 2:
        logger.warning("Not enough stations with sufficient data for MWPC")
        return results
    
    # Compute MWPC for all pairs
    station_list = list(station_series.keys())
    
    for i, sta1 in enumerate(station_list):
        for sta2 in station_list[i+1:]:
            # Get coordinates
            coord1 = STATION_COORDS[sta1]
            coord2 = STATION_COORDS[sta2]
            baseline_km = haversine_distance_km(coord1['lat'], coord1['lon'], 
                                                 coord2['lat'], coord2['lon'])
            
            # Align time series (find overlap)
            t1, r1 = station_series[sta1]['times'], station_series[sta1]['residuals']
            t2, r2 = station_series[sta2]['times'], station_series[sta2]['residuals']
            
            # Find common time range
            t_start = max(t1.min(), t2.min())
            t_end = min(t1.max(), t2.max())
            
            if t_end <= t_start:
                continue
            
            # Interpolate to common grid
            common_times = np.arange(t_start, t_end, 300)  # 5-min grid
            if len(common_times) < 20:
                continue
            
            r1_interp = np.interp(common_times, t1, r1)
            r2_interp = np.interp(common_times, t2, r2)
            
            # Compute MWPC
            mwpc = compute_mwpc(r1_interp, r2_interp, fs=FS_HZ, f1=F1_HZ, f2=F2_HZ)
            
            if not np.isnan(mwpc['coherence']):
                results['pairs'].append({
                    'station_1': int(sta1),
                    'station_2': int(sta2),
                    'name_1': coord1['name'],
                    'name_2': coord2['name'],
                    'baseline_km': float(baseline_km),
                    'n_points': len(common_times),
                    **mwpc,
                })
    
    logger.info(f"  Computed MWPC for {len(results['pairs'])} station pairs")
    
    # Bin by distance
    if results['pairs']:
        pair_df = pd.DataFrame(results['pairs'])
        
        for i in range(len(DISTANCE_BINS_KM) - 1):
            d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i+1]
            bin_data = pair_df[(pair_df['baseline_km'] >= d_lo) & (pair_df['baseline_km'] < d_hi)]
            
            if len(bin_data) > 0:
                results['distance_binned'][f"{d_lo}-{d_hi}km"] = {
                    'n_pairs': len(bin_data),
                    'mean_distance_km': float(bin_data['baseline_km'].mean()),
                    'mean_coherence': float(bin_data['coherence'].mean()),
                    'std_coherence': float(bin_data['coherence'].std()),
                    'mean_phase_alignment': float(bin_data['phase_alignment'].mean()),
                    'std_phase_alignment': float(bin_data['phase_alignment'].std()),
                }
        
        # Fit exponential decay
        if len(pair_df) >= 3:
            try:
                distances = pair_df['baseline_km'].values
                coherences = pair_df['coherence'].values
                phase_alignments = pair_df['phase_alignment'].values
                
                valid_coh = ~np.isnan(coherences)
                valid_pa = ~np.isnan(phase_alignments)
                
                if np.sum(valid_coh) >= 3:
                    popt_coh, pcov_coh = curve_fit(
                        exponential_decay,
                        distances[valid_coh],
                        coherences[valid_coh],
                        p0=[0.5, TEP_COHERENCE_LENGTH_KM, 0.0],
                        bounds=([0, 100, -1], [1, 20000, 1]),
                        maxfev=2000
                    )
                    results['fit_results']['coherence'] = {
                        'amplitude': float(popt_coh[0]),
                        'lambda_km': float(popt_coh[1]),
                        'offset': float(popt_coh[2]),
                        'lambda_uncertainty_km': float(np.sqrt(pcov_coh[1, 1])) if pcov_coh[1, 1] > 0 else None,
                    }
                    
                if np.sum(valid_pa) >= 3:
                    popt_pa, pcov_pa = curve_fit(
                        exponential_decay,
                        distances[valid_pa],
                        phase_alignments[valid_pa],
                        p0=[0.5, TEP_COHERENCE_LENGTH_KM, 0.0],
                        bounds=([-1, 100, -1], [1, 20000, 1]),
                        maxfev=2000
                    )
                    results['fit_results']['phase_alignment'] = {
                        'amplitude': float(popt_pa[0]),
                        'lambda_km': float(popt_pa[1]),
                        'offset': float(popt_pa[2]),
                        'lambda_uncertainty_km': float(np.sqrt(pcov_pa[1, 1])) if pcov_pa[1, 1] > 0 else None,
                    }
                    
            except Exception as e:
                logger.warning(f"  Fit failed: {e}")
    
    return results


def analyze_range_coherence(df: pd.DataFrame) -> Dict:
    """
    Analyze coherence as a function of station-satellite range.
    
    TEP predicts that ranging measurements should show distance-structured
    correlations. We bin observations by range and compute coherence metrics.
    """
    logger.info("Analyzing range-dependent coherence...")
    
    results = {
        'range_bins': [],
        'correlation_with_range': {},
    }
    
    # Remove station biases
    df = df.copy()
    df['residual_debiased'] = df.groupby('station')['residual_m'].transform(lambda x: x - x.mean())
    
    # Bin by range
    range_bins = np.linspace(df['model_range_m'].min(), df['model_range_m'].max(), 8)
    df['range_bin'] = pd.cut(df['model_range_m'], bins=range_bins)
    
    for bin_label, group in df.groupby('range_bin', observed=True):
        if len(group) < 10:
            continue
        
        mid_range_km = group['model_range_m'].mean() / 1000
        
        # Compute statistics
        residuals = group['residual_debiased'].values
        
        # Autocorrelation (lag-1)
        if len(residuals) > 1:
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        else:
            autocorr = np.nan
        
        results['range_bins'].append({
            'range_km': float(mid_range_km),
            'n_obs': len(group),
            'mean_residual_mm': float(group['residual_debiased'].mean() * 1000),
            'std_residual_mm': float(group['residual_debiased'].std() * 1000),
            'autocorrelation': float(autocorr) if not np.isnan(autocorr) else None,
        })
    
    # Overall range-residual correlation
    r, p = stats.pearsonr(df['model_range_m'], df['residual_debiased'])
    results['correlation_with_range'] = {
        'pearson_r': float(r),
        'p_value': float(p),
        'significant': p < 0.05,
    }
    
    return results


def analyze_temporal_coherence(df: pd.DataFrame) -> Dict:
    """
    Analyze temporal coherence structure of residuals.
    
    Computes autocorrelation function and spectral properties
    to identify TEP-relevant temporal patterns.
    """
    logger.info("Analyzing temporal coherence...")
    
    results = {
        'station_acf': {},
        'overall_spectral': {},
    }
    
    # For each station, compute autocorrelation function
    for sta in df['station'].unique():
        sta_df = df[df['station'] == sta].copy()
        if len(sta_df) < 20:
            continue
        
        sta_df['epoch'] = pd.to_datetime(sta_df['epoch_utc'])
        sta_df = sta_df.sort_values('epoch')
        
        residuals = sta_df['residual_m'].values - sta_df['residual_m'].mean()
        
        # Compute ACF for lags 0-10
        acf_values = []
        for lag in range(min(10, len(residuals) // 2)):
            if lag == 0:
                acf_values.append(1.0)
            else:
                acf = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                acf_values.append(float(acf) if not np.isnan(acf) else 0.0)
        
        results['station_acf'][int(sta)] = acf_values
    
    # Overall spectral analysis (combine all stations)
    all_residuals = []
    for sta in df['station'].unique():
        times, residuals = build_station_time_series(df, sta, resample_interval='5min')
        if len(residuals) > 0:
            all_residuals.extend(residuals.tolist())
    
    if len(all_residuals) > 50:
        all_residuals = np.array(all_residuals)
        all_residuals = detrend(all_residuals, type='linear')
        
        try:
            freqs, psd = welch(all_residuals, fs=FS_HZ, nperseg=min(256, len(all_residuals)//2))
            
            # TEP band power
            tep_mask = (freqs >= F1_HZ) & (freqs <= F2_HZ)
            if np.any(tep_mask):
                tep_power = np.mean(psd[tep_mask])
                total_power = np.mean(psd[freqs > 0])
                results['overall_spectral'] = {
                    'tep_band_power': float(tep_power),
                    'total_power': float(total_power),
                    'tep_band_fraction': float(tep_power / total_power) if total_power > 0 else None,
                }
        except Exception:
            pass
    
    return results


def compare_with_gnss(results: Dict) -> Dict:
    """
    Compare SLR MWPC results with TEP-GNSS predictions.
    """
    comparison = {
        'gnss_predictions': {
            'coherence_length_km': TEP_COHERENCE_LENGTH_KM,
            'signal_amplitude_ns': 0.2,
            'tep_band_hz': [F1_HZ, F2_HZ],
        },
        'slr_results': {},
        'consistency': {},
    }
    
    # Extract SLR fitted coherence length
    if 'interstation_mwpc' in results:
        fit = results['interstation_mwpc'].get('fit_results', {})
        
        if 'coherence' in fit:
            slr_lambda = fit['coherence']['lambda_km']
            slr_lambda_unc = fit['coherence'].get('lambda_uncertainty_km')
            
            comparison['slr_results']['coherence_lambda_km'] = slr_lambda
            comparison['slr_results']['coherence_lambda_uncertainty_km'] = slr_lambda_unc
            
            # Check consistency with GNSS
            if slr_lambda_unc:
                sigma_diff = abs(slr_lambda - TEP_COHERENCE_LENGTH_KM) / slr_lambda_unc
                comparison['consistency']['coherence_sigma_difference'] = float(sigma_diff)
                comparison['consistency']['coherence_consistent_2sigma'] = sigma_diff < 2
        
        if 'phase_alignment' in fit:
            slr_lambda = fit['phase_alignment']['lambda_km']
            slr_lambda_unc = fit['phase_alignment'].get('lambda_uncertainty_km')
            
            comparison['slr_results']['phase_alignment_lambda_km'] = slr_lambda
            comparison['slr_results']['phase_alignment_lambda_uncertainty_km'] = slr_lambda_unc
            
            if slr_lambda_unc:
                sigma_diff = abs(slr_lambda - TEP_COHERENCE_LENGTH_KM) / slr_lambda_unc
                comparison['consistency']['phase_alignment_sigma_difference'] = float(sigma_diff)
                comparison['consistency']['phase_alignment_consistent_2sigma'] = sigma_diff < 2
    
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="TEP-SLR Step 2.3: MWPC Analysis")
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
    logger.info(f"Stations: {sorted(df['station'].unique())}")
    logger.info(f"Satellites: {sorted(df['satellite'].unique())}")
    
    # Run MWPC analyses
    results = {
        'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
        'input_file': str(input_path),
        'methodology': 'Magnitude-Weighted Phase Correlation (TEP-GNSS-RINEX)',
        'tep_parameters': {
            'coherence_length_km': TEP_COHERENCE_LENGTH_KM,
            'frequency_band_hz': [F1_HZ, F2_HZ],
            'sampling_hz': FS_HZ,
        },
        'data_summary': {
            'n_observations': len(df),
            'n_stations': int(df['station'].nunique()),
            'n_satellites': int(df['satellite'].nunique()),
            'date_range': [df['epoch_utc'].min(), df['epoch_utc'].max()],
        },
    }
    
    # 1. Pass-based correlations (works better with sparse SLR data)
    results['pass_correlations'] = analyze_pass_correlations(df)
    
    # 2. Inter-station MWPC (requires temporal overlap)
    results['interstation_mwpc'] = analyze_interstation_mwpc(df)
    
    # 3. Range coherence
    results['range_coherence'] = analyze_range_coherence(df)
    
    # 3. Temporal coherence
    results['temporal_coherence'] = analyze_temporal_coherence(df)
    
    # 4. Comparison with GNSS
    results['gnss_comparison'] = compare_with_gnss(results)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("MAGNITUDE-WEIGHTED PHASE CORRELATION ANALYSIS")
    logger.info("=" * 70)
    
    logger.info(f"\nMethodology: {results['methodology']}")
    logger.info(f"TEP coherence length (GNSS): {TEP_COHERENCE_LENGTH_KM} km")
    logger.info(f"TEP frequency band: {F1_HZ*1e6:.0f}-{F2_HZ*1e6:.0f} µHz")
    
    if results['interstation_mwpc']['pairs']:
        logger.info(f"\nInter-station pairs analyzed: {len(results['interstation_mwpc']['pairs'])}")
        for pair in results['interstation_mwpc']['pairs']:
            logger.info(f"  {pair['name_1']}-{pair['name_2']}: "
                       f"baseline={pair['baseline_km']:.0f}km, "
                       f"coh={pair['coherence']:.3f}, "
                       f"phase_align={pair['phase_alignment']:.3f}")
    
    fit = results['interstation_mwpc'].get('fit_results', {})
    if 'coherence' in fit:
        logger.info(f"\nFitted coherence length: {fit['coherence']['lambda_km']:.0f} km")
        logger.info(f"  (GNSS prediction: {TEP_COHERENCE_LENGTH_KM} km)")
        if fit['coherence'].get('lambda_uncertainty_km'):
            logger.info(f"  Uncertainty: ±{fit['coherence']['lambda_uncertainty_km']:.0f} km")
    
    if 'phase_alignment' in fit:
        logger.info(f"\nFitted phase alignment λ: {fit['phase_alignment']['lambda_km']:.0f} km")
    
    # Comparison summary
    comp = results['gnss_comparison']
    if comp.get('consistency'):
        logger.info("\nConsistency with TEP-GNSS:")
        for key, val in comp['consistency'].items():
            logger.info(f"  {key}: {val}")
    
    # Save results
    out_json = OUTPUTS_DIR / "step_2_3_mwpc_analysis.json"
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n[SUCCESS] Results saved to {out_json}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
