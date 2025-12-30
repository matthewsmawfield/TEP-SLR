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
import gzip
import re
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

# Setup path for imports
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.logger import TEPLogger, set_step_logger

# Set up logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger("step_2_3", log_file_path=LOGS_DIR / "step_2_3_mwpc_analysis.log")
set_step_logger(logger)

# TEP parameters (from GNSS analysis)
TEP_COHERENCE_LENGTH_KM = 4200.0
F1_HZ = 10e-6   # 10 µHz (TEP band lower limit)
F2_HZ = 500e-6  # 500 µHz (TEP band upper limit)
FS_HZ = 1/300   # 5-minute sampling (matching GNSS)
BROADBAND_MIN_HZ = 1e-3
DEFAULT_RESIDUAL_THRESHOLD_M = 0.5
DEFAULT_RESIDUAL_THRESHOLDS_M = [0.3, 0.5, 1.0]
DEFAULT_BOOTSTRAP_N = 1000

# Distance bins for analysis
DISTANCE_BINS_KM = [0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000]

PASS_TIME_BIN = '15min'  # Widened from 5min to increase overlap
MWPC_RESAMPLE_INTERVAL = '5min'
MIN_OBS_PER_STATION_BIN = 1  # Lowered from 2 to include more pass-pairs
MIN_POINTS_MWPC = 24  # Lowered from 32
MIN_PAIRS_PER_DISTANCE_BIN = 3  # Lowered from 5
MAX_LOGGED_PAIRS = 60

MIN_SHARED_BINS_INTERSTATION = 6  # Lowered from 12 to include more pairs

# Daily aggregation parameters (new)
DAILY_AGG_MIN_OBS_PER_DAY = 3
DAILY_AGG_MIN_DAYS_PER_STATION = 10
DAILY_AGG_MIN_SHARED_DAYS = 5
TOP_STATIONS_COUNT = 20  # Focus on highest-activity stations
IRREGULAR_N_FREQ = 64
IRREGULAR_SEGMENT_MIN_POINTS = 8
IRREGULAR_SEGMENT_STEP_POINTS = 4
IRREGULAR_SEGMENT_MAX_GAP_SECONDS = 450.0

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - _WGS84_E2))

    for _ in range(7):
        sin_lat = math.sin(lat)
        n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / math.cos(lat) - n
        lat_new = math.atan2(z, p * (1.0 - _WGS84_E2 * (n / (n + h))))
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = math.sin(lat)
    n = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - n
    return lat, lon, h


def _sinex_epoch_to_datetime(epoch: str) -> Optional[datetime]:
    m = re.match(r"^(\d{2}):(\d{3}):(\d{5})$", epoch.strip())
    if not m:
        return None
    yy = int(m.group(1))
    doy = int(m.group(2))
    sec = int(m.group(3))
    year = 2000 + yy if yy <= 79 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, seconds=sec)


def load_station_coords_from_slrf2020(sinex_path: Path) -> Dict[int, Dict[str, object]]:
    opener = gzip.open if sinex_path.suffix == ".gz" else open
    in_est = False
    stations: Dict[str, Dict[str, Tuple[float, datetime]]] = {}

    with opener(sinex_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("+SOLUTION/ESTIMATE"):
                in_est = True
                continue
            if line.startswith("-SOLUTION/ESTIMATE"):
                in_est = False
                continue
            if not in_est:
                continue
            if not line.strip() or line.startswith("*"):
                continue

            parts = line.split()
            if len(parts) < 9:
                continue

            param = parts[1].upper()
            site = parts[2].strip()
            epoch = parts[5]
            value = parts[8]

            if param not in {"STAX", "STAY", "STAZ"}:
                continue

            try:
                v = float(value)
            except Exception:
                continue

            dt = _sinex_epoch_to_datetime(epoch)
            if dt is None:
                continue

            if site not in stations:
                stations[site] = {}

            prev = stations[site].get(param)
            if prev is None or dt > prev[1]:
                stations[site][param] = (float(v), dt)

    out: Dict[int, Dict[str, object]] = {}
    for site, params in stations.items():
        if not all(k in params for k in ["STAX", "STAY", "STAZ"]):
            continue

        x = params["STAX"][0]
        y = params["STAY"][0]
        z = params["STAZ"][0]
        lat_rad, lon_rad, _h_m = ecef_to_lla(x, y, z)

        try:
            code_int = int(site)
        except Exception:
            continue

        out[code_int] = {
            "name": str(code_int),
            "lat": float(math.degrees(lat_rad)),
            "lon": float(math.degrees(lon_rad)),
            "ecef": [float(x), float(y), float(z)],
        }

    return out


def find_local_slrf2020_sinex(project_root: Path) -> Optional[Path]:
    resource_dir = project_root / "data" / "slr" / "products" / "resource"
    if not resource_dir.exists():
        return None
    candidates = sorted(list(resource_dir.glob("*slrf2020*.snx")) + list(resource_dir.glob("*slrf2020*.snx.gz")))
    if candidates:
        return candidates[-1]
    candidates = sorted(list(resource_dir.glob("*.snx")) + list(resource_dir.glob("*.snx.gz")))
    return candidates[-1] if candidates else None


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
    
    sta_df['epoch'] = pd.to_datetime(sta_df['epoch_utc'], format='mixed')
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


def build_station_resampled_series(df: pd.DataFrame, station: int, resample_interval: str = '5min') -> pd.Series:
    sta_df = df[df['station'] == station].copy()
    if len(sta_df) < 10:
        return pd.Series(dtype=float)

    sta_df['epoch'] = pd.to_datetime(sta_df['epoch_utc'], format='mixed')
    sta_df = sta_df.sort_values('epoch')
    sta_df = sta_df.set_index('epoch')

    sta_df['residual_debiased'] = sta_df['residual_m'] - sta_df['residual_m'].mean()

    resampled = sta_df['residual_debiased'].resample(resample_interval).mean()
    resampled = resampled.interpolate(method='linear', limit=2)
    return resampled


def _longest_contiguous_overlap_segment(df_pair: pd.DataFrame, step: pd.Timedelta) -> Optional[pd.DataFrame]:
    if df_pair.empty:
        return None

    mask = df_pair['r1'].notna() & df_pair['r2'].notna()
    if not mask.any():
        return None

    idx = df_pair.index
    best_start = None
    best_end = None
    cur_start = None
    prev_t = None

    for t, ok in zip(idx, mask.values):
        if ok and (prev_t is None or (t - prev_t) == step):
            if cur_start is None:
                cur_start = t
        else:
            if cur_start is not None:
                cur_end = prev_t
                if best_start is None or (cur_end - cur_start) > (best_end - best_start):
                    best_start, best_end = cur_start, cur_end
                cur_start = None
        prev_t = t

    if cur_start is not None:
        cur_end = prev_t
        if best_start is None or (cur_end - cur_start) > (best_end - best_start):
            best_start, best_end = cur_start, cur_end

    if best_start is None or best_end is None:
        return None

    seg = df_pair.loc[best_start:best_end]
    seg = seg.dropna()
    return seg


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return float('nan')
    sx = float(np.std(x))
    sy = float(np.std(y))
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0.0 or sy == 0.0:
        return float('nan')
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    cov = float(np.mean((x - xm) * (y - ym)))
    return float(cov / (sx * sy))


def _weighted_pearson_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.size != y.size or x.size != w.size or x.size < 3:
        return float('nan')
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.sum(mask) < 3:
        return float('nan')
    x = x[mask]
    y = y[mask]
    w = w[mask]
    sw = float(np.sum(w))
    if sw <= 0:
        return float('nan')
    mx = float(np.sum(w * x) / sw)
    my = float(np.sum(w * y) / sw)
    vx = float(np.sum(w * (x - mx) ** 2) / sw)
    vy = float(np.sum(w * (y - my) ** 2) / sw)
    if vx <= 0 or vy <= 0:
        return float('nan')
    cov = float(np.sum(w * (x - mx) * (y - my)) / sw)
    return float(cov / math.sqrt(vx * vy))


def _parse_float_list_csv(s: str) -> List[float]:
    out: List[float] = []
    for part in str(s).split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    return out


def _bootstrap_mean(values: np.ndarray, n_boot: int, seed: int = 0) -> Dict[str, object]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            'n': 0,
            'mean': None,
            'ci95': [None, None],
        }

    if int(n_boot) <= 0:
        return {
            'n': int(v.size),
            'mean': float(np.mean(v)),
            'ci95': [None, None],
        }

    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, int(v.size), size=(int(n_boot), int(v.size)))
    means = np.mean(v[idx], axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        'n': int(v.size),
        'mean': float(np.mean(v)),
        'ci95': [float(lo), float(hi)],
    }


def _lag1_autocorr(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float('nan')
    x = v[:-1]
    y = v[1:]
    return float(np.corrcoef(x, y)[0, 1])


def _lag1_autocorr_max_gap(values: np.ndarray, times_s: np.ndarray, max_gap_s: float) -> Tuple[float, int]:
    v = np.asarray(values, dtype=float)
    t = np.asarray(times_s, dtype=float)
    m = np.isfinite(v) & np.isfinite(t)
    v = v[m]
    t = t[m]
    if v.size < 3:
        return float('nan'), 0
    order = np.argsort(t)
    v = v[order]
    t = t[order]

    dt = np.diff(t)
    ok = dt <= float(max_gap_s)
    if not np.any(ok):
        return float('nan'), 0
    x = v[:-1][ok]
    y = v[1:][ok]
    if x.size < 3:
        return float('nan'), int(x.size)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0 or (not np.isfinite(sx)) or (not np.isfinite(sy)):
        return float('nan'), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def _station_lag1_autocorr_by_range(
    df: pd.DataFrame,
    range_min_m: float,
    range_max_m: float,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    sel = df[(df['model_range_m'] >= float(range_min_m)) & (df['model_range_m'] < float(range_max_m))]
    if sel.empty:
        return out

    sel = sel.copy()
    sel['epoch'] = pd.to_datetime(sel['epoch_utc'], format='mixed')
    for sta, grp in sel.groupby('station'):
        g = grp.sort_values('epoch')
        r = g['residual_debiased'].to_numpy(dtype=float)
        ac = _lag1_autocorr(r)
        if not np.isnan(ac):
            out[int(sta)] = float(ac)
    return out


def analyze_pass_correlations(
    df: pd.DataFrame,
    station_coords: Dict[int, Dict[str, object]],
    n_permutations: int = 200,
    fwer_metric: str = 'weighted',
    include_pass_pairs: bool = True,
) -> Dict:
    """
    Analyze correlations between contemporaneous passes at different stations.
    
    This is more suitable for sparse SLR data than continuous time series MWPC.
    We compute correlations between residual patterns when multiple stations
    observe the same satellite pass.
    """
    logger.info("Analyzing pass-based correlations...")
    
    results: Dict[str, object] = {
        'pass_pairs': [] if include_pass_pairs else None,
        'distance_binned': {},
    }
    
    # Pre-calculate global station biases to preserve pass-specific variations
    station_biases = df.groupby('station')['residual_m'].mean()
    
    df = df.copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'], format='mixed')
    
    df['time_bin'] = df['epoch'].dt.floor(PASS_TIME_BIN)
    
    # For each time bin, find stations with observations
    for (satellite, time_bin), group in df.groupby(['satellite', 'time_bin']):
        stations_in_bin = group['station'].unique()
        
        if len(stations_in_bin) < 2:
            continue
        
        # Compare all station pairs in this time bin
        for i, sta1 in enumerate(stations_in_bin):
            for sta2 in stations_in_bin[i+1:]:
                sta1_i = int(sta1)
                sta2_i = int(sta2)
                if sta1_i not in station_coords or sta2_i not in station_coords:
                    continue

                coord1 = station_coords[sta1_i]
                coord2 = station_coords[sta2_i]
                baseline_km = haversine_distance_km(
                    coord1['lat'], coord1['lon'],
                    coord2['lat'], coord2['lon']
                )
                
                # Get residuals for each station in this bin
                r1 = group[group['station'] == sta1]['residual_m'].values
                r2 = group[group['station'] == sta2]['residual_m'].values

                if len(r1) < MIN_OBS_PER_STATION_BIN or len(r2) < MIN_OBS_PER_STATION_BIN:
                    continue
                
                # Debias using GLOBAL station means (preserving the pass anomaly)
                # If we subtract the local mean, we get 0.0 for every pass.
                bias1 = station_biases.get(sta1_i, 0.0)
                bias2 = station_biases.get(sta2_i, 0.0)
                
                mean1 = r1.mean() - bias1
                mean2 = r2.mean() - bias2
                
                pass_pair_entry = {
                    'time_bin': str(time_bin),
                    'satellite': str(satellite),
                    'station_1': int(sta1),
                    'station_2': int(sta2),
                    'baseline_km': float(baseline_km),
                    'n_obs_1': len(r1),
                    'n_obs_2': len(r2),
                    'mean_residual_1_mm': float(mean1 * 1000),
                    'mean_residual_2_mm': float(mean2 * 1000),
                    'product_mm2': float(mean1 * mean2 * 1e6),  # For correlation
                }
                if include_pass_pairs:
                    assert isinstance(results['pass_pairs'], list)
                    results['pass_pairs'].append(pass_pair_entry)
                else:
                    results.setdefault('_pass_pairs_internal', []).append(pass_pair_entry)
    
    pass_pairs_internal = results.get('pass_pairs')
    if not include_pass_pairs:
        pass_pairs_internal = results.pop('_pass_pairs_internal', [])

    # Aggregate by station pair
    if pass_pairs_internal:
        pair_df = pd.DataFrame(pass_pairs_internal)

        def _pair_corrs_for_df(df_in: pd.DataFrame) -> List[Dict[str, object]]:
            pair_correlations: List[Dict[str, object]] = []
            for (sta1, sta2), pair_data in df_in.groupby(['station_1', 'station_2']):
                if len(pair_data) < 3:
                    continue
                r1_means = pair_data['mean_residual_1_mm'].values
                r2_means = pair_data['mean_residual_2_mm'].values
                w = np.minimum(pair_data['n_obs_1'].to_numpy(dtype=float), pair_data['n_obs_2'].to_numpy(dtype=float))
                corr = _pearson_corr(r1_means, r2_means)
                corr_w = _weighted_pearson_corr(r1_means, r2_means, w)
                if not np.isnan(corr) or not np.isnan(corr_w):
                    baseline = float(pair_data['baseline_km'].iloc[0])
                    pair_correlations.append({
                        'station_1': int(sta1),
                        'station_2': int(sta2),
                        'baseline_km': float(baseline),
                        'n_passes': int(len(pair_data)),
                        'correlation': float(corr) if not np.isnan(corr) else None,
                        'correlation_weighted': float(corr_w) if not np.isnan(corr_w) else None,
                    })
            return pair_correlations

        results['pair_correlations'] = _pair_corrs_for_df(pair_df)

        by_sat: Dict[str, List[Dict[str, object]]] = {}
        for sat, sub in pair_df.groupby('satellite'):
            by_sat[str(sat)] = _pair_corrs_for_df(sub)
        results['pair_correlations_by_satellite'] = by_sat

        def _bin_by_distance(pair_corrs: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
            out: Dict[str, Dict[str, object]] = {}
            if not pair_corrs:
                return out
            corr_df = pd.DataFrame(pair_corrs)
            for i in range(len(DISTANCE_BINS_KM) - 1):
                d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
                bin_data = corr_df[(corr_df['baseline_km'] >= d_lo) & (corr_df['baseline_km'] < d_hi)]
                if len(bin_data) > 0:
                    v = bin_data['correlation'].astype(float)
                    vw = bin_data['correlation_weighted'].astype(float)
                    out[f"{d_lo}-{d_hi}km"] = {
                        'n_pairs': int(len(bin_data)),
                        'mean_distance_km': float(bin_data['baseline_km'].mean()),
                        'mean_correlation': float(v.mean()) if v.notna().any() else None,
                        'std_correlation': float(v.std()) if int(v.notna().sum()) > 1 else 0.0,
                        'mean_correlation_weighted': float(vw.mean()) if vw.notna().any() else None,
                        'std_correlation_weighted': float(vw.std()) if int(vw.notna().sum()) > 1 else 0.0,
                    }
            return out

        results['distance_binned'] = _bin_by_distance(results['pair_correlations'])
        results['distance_binned_by_satellite'] = {
            sat: _bin_by_distance(pc)
            for sat, pc in results.get('pair_correlations_by_satellite', {}).items()
        }

        def _fwer_null_for_df(df_in: pd.DataFrame) -> Optional[Dict[str, object]]:
            pair_series: List[Dict[str, object]] = []
            for (sta1, sta2), sub in df_in.groupby(['station_1', 'station_2']):
                if len(sub) < 3:
                    continue
                r1 = sub['mean_residual_1_mm'].to_numpy(dtype=float)
                r2 = sub['mean_residual_2_mm'].to_numpy(dtype=float)
                w = np.minimum(sub['n_obs_1'].to_numpy(dtype=float), sub['n_obs_2'].to_numpy(dtype=float))
                if str(fwer_metric).lower() == 'unweighted':
                    corr_obs = _pearson_corr(r1, r2)
                else:
                    corr_obs = _weighted_pearson_corr(r1, r2, w)
                if np.isnan(corr_obs):
                    continue
                baseline = float(sub['baseline_km'].iloc[0])
                pair_series.append({'baseline_km': baseline, 'r1': r1, 'r2': r2, 'w': w})

            if not pair_series:
                return None

            obs_bin_means: Dict[str, float] = {}
            for i in range(len(DISTANCE_BINS_KM) - 1):
                d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
                vals: List[float] = []
                for ps in pair_series:
                    if d_lo <= ps['baseline_km'] < d_hi:
                        c = _weighted_pearson_corr(ps['r1'], ps['r2'], ps['w'])
                        if not np.isnan(c):
                            vals.append(float(c))
                if len(vals) >= MIN_PAIRS_PER_DISTANCE_BIN:
                    obs_bin_means[f"{d_lo}-{d_hi}km"] = float(np.mean(vals))

            if not obs_bin_means:
                return None

            obs_stat = float(np.nanmin(np.array(list(obs_bin_means.values()), dtype=float)))
            rng = np.random.default_rng(0)

            null_stats: List[float] = []
            null_bin_means: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
            for _ in range(int(n_permutations)):
                perm_bin_vals: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
                for ps in pair_series:
                    r1 = np.asarray(ps['r1'], dtype=float)
                    r2 = np.asarray(ps['r2'], dtype=float)
                    w = np.asarray(ps['w'], dtype=float)
                    n = int(r2.size)
                    if n < 3:
                        continue
                    shift = int(rng.integers(1, n))
                    r2s = np.roll(r2, shift)
                    if str(fwer_metric).lower() == 'unweighted':
                        c = _pearson_corr(r1, r2s)
                    else:
                        c = _weighted_pearson_corr(r1, r2s, w)
                    if np.isnan(c):
                        continue
                    baseline = float(ps['baseline_km'])
                    for i in range(len(DISTANCE_BINS_KM) - 1):
                        d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
                        k = f"{d_lo}-{d_hi}km"
                        if k not in perm_bin_vals:
                            continue
                        if d_lo <= baseline < d_hi:
                            perm_bin_vals[k].append(float(c))
                            break

                perm_means_list: List[float] = []
                for k, vals in perm_bin_vals.items():
                    if vals:
                        m = float(np.mean(vals))
                        null_bin_means[k].append(m)
                        perm_means_list.append(m)
                    else:
                        null_bin_means[k].append(np.nan)

                if perm_means_list:
                    null_stats.append(float(np.nanmin(np.array(perm_means_list, dtype=float))))

            if not null_stats:
                return None

            null_arr = np.array(null_stats, dtype=float)
            p_fwer = float((np.sum(null_arr <= obs_stat) + 1) / (len(null_arr) + 1))

            per_bin_p: Dict[str, Optional[float]] = {}
            for k, obs_mean in obs_bin_means.items():
                arr = np.asarray([v for v in null_bin_means.get(k, []) if not np.isnan(v)], dtype=float)
                if arr.size == 0:
                    per_bin_p[k] = None
                else:
                    per_bin_p[k] = float((np.sum(arr <= float(obs_mean)) + 1) / (len(arr) + 1))

            return {
                'method': 'min_mean_over_bins_fwer_correlation',
                'distance_bins_km': [float(x) for x in DISTANCE_BINS_KM],
                'min_pairs_per_bin': int(MIN_PAIRS_PER_DISTANCE_BIN),
                'n_permutations': int(len(null_arr)),
                'correlation_metric': (
                    'pearson' if str(fwer_metric).lower() == 'unweighted' else 'weighted_pearson(min(n_obs_1,n_obs_2))'
                ),
                'observed_bin_means': {k: float(v) for k, v in obs_bin_means.items()},
                'observed_statistic': float(obs_stat),
                'p_value_one_sided_le': float(p_fwer),
                'p_values_one_sided_le_by_bin': per_bin_p,
            }

        null_by_sat: Dict[str, Dict[str, object]] = {}
        pvals_for_fisher: List[float] = []
        for sat, sub in pair_df.groupby('satellite'):
            nt = _fwer_null_for_df(sub)
            if nt is not None:
                null_by_sat[str(sat)] = nt
                pv = nt.get('p_value_one_sided_le')
                if pv is not None:
                    pvals_for_fisher.append(float(pv))

        results['null_tests_by_satellite'] = null_by_sat
        if pvals_for_fisher:
            stat = float(-2.0 * np.sum(np.log(np.array(pvals_for_fisher, dtype=float))))
            dof = 2 * len(pvals_for_fisher)
            p_combined = float(1.0 - stats.chi2.cdf(stat, dof))
            results['null_tests_fwer'] = {
                'method': 'fisher',
                'n_inputs': int(len(pvals_for_fisher)),
                'chi2_stat': float(stat),
                'dof': int(dof),
                'p_value_combined': float(p_combined),
                'p_values_inputs': [float(p) for p in pvals_for_fisher],
            }
    
    logger.info(f"  Found {len(results['pass_pairs']) if isinstance(results.get('pass_pairs'), list) else 0} pass-pair observations")
    if 'pair_correlations' in results:
        logger.info(f"  Computed correlations for {len(results['pair_correlations'])} station pairs")
    
    return results


def _extract_years(df: pd.DataFrame) -> List[int]:
    if df.empty:
        return []
    try:
        t = pd.to_datetime(df['epoch_utc'], format='mixed', utc=True)
    except Exception:
        t = pd.to_datetime(df['epoch_utc'], utc=True, errors='coerce')
    years = sorted({int(y) for y in t.dt.year.dropna().unique().tolist()})
    return years


def _compute_irregular_phase_coherence(
    t_seconds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    f1_hz: float = F1_HZ,
    f2_hz: float = F2_HZ,
    n_freq: int = IRREGULAR_N_FREQ,
) -> Dict[str, float]:
    t = np.asarray(t_seconds, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < MIN_SHARED_BINS_INTERSTATION:
        return {'coherence': float('nan'), 'phase_alignment': float('nan'), 'weighted_phase': float('nan')}

    t = t - float(t.min())

    # Detrend with respect to time (irregular)
    try:
        px = np.polyfit(t, x, deg=1)
        py = np.polyfit(t, y, deg=1)
        x = x - (px[0] * t + px[1])
        y = y - (py[0] * t + py[1])
    except Exception:
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))

    # Welch-style segmentation over contiguous time to avoid coherence degeneracy.
    # A single segment yields |Pxy| = sqrt(Pxx Pyy) identically for our single-DFT estimator.
    # Averaging spectra over multiple segments breaks this identity and yields 0<=coh<=1.
    seg_len = int(IRREGULAR_SEGMENT_MIN_POINTS)
    seg_step = max(1, int(IRREGULAR_SEGMENT_STEP_POINTS))

    # First split into contiguous blocks by large gaps, then within each block do fixed-length windowing.
    seg_slices: List[slice] = []
    block_start = 0
    for i in range(1, int(t.size)):
        if float(t[i] - t[i - 1]) > float(IRREGULAR_SEGMENT_MAX_GAP_SECONDS):
            block = slice(block_start, i)
            n_block = int(block.stop - block.start)
            if n_block >= seg_len:
                for j in range(0, n_block - seg_len + 1, seg_step):
                    seg_slices.append(slice(block.start + j, block.start + j + seg_len))
            block_start = i
    block = slice(block_start, int(t.size))
    n_block = int(block.stop - block.start)
    if n_block >= seg_len:
        for j in range(0, n_block - seg_len + 1, seg_step):
            seg_slices.append(slice(block.start + j, block.start + j + seg_len))

    freqs = np.linspace(float(f1_hz), float(f2_hz), int(n_freq), dtype=float)
    omega = 2.0 * math.pi * freqs

    if len(seg_slices) < 2:
        # Broadband (frequency-averaged) coherence fallback for sparse overlaps.
        # Uses aggregation across frequencies to avoid the single-frequency degeneracy.
        tt = t - float(t.min())
        E = np.exp(-1j * omega[:, None] * tt[None, :])
        X = E @ x
        Y = E @ y
        Pxy_f = X * np.conjugate(Y)
        Pxx_f = (np.abs(X) ** 2).real
        Pyy_f = (np.abs(Y) ** 2).real
        denom_f = np.sqrt(Pxx_f * Pyy_f)
        m = denom_f > 0
        if not np.any(m):
            return {
                'coherence': float('nan'),
                'phase_alignment': float('nan'),
                'weighted_phase': float('nan'),
                'n_segments': 1.0,
            }

        Pxy_sum = np.sum(Pxy_f[m])
        Pxx_sum = float(np.sum(Pxx_f[m]))
        Pyy_sum = float(np.sum(Pyy_f[m]))
        denom = math.sqrt(Pxx_sum * Pyy_sum) if (Pxx_sum > 0 and Pyy_sum > 0) else 0.0
        if denom <= 0:
            return {
                'coherence': float('nan'),
                'phase_alignment': float('nan'),
                'weighted_phase': float('nan'),
                'n_segments': 1.0,
            }

        coh = float(np.clip(np.abs(Pxy_sum) / denom, 0.0, 1.0))
        weighted_phase = float(np.angle(Pxy_sum))
        phase_alignment = float(math.cos(weighted_phase))

        return {
            'coherence': coh,
            'phase_alignment': phase_alignment,
            'weighted_phase': weighted_phase,
            'n_segments': 1.0,
        }

    Pxy_acc = np.zeros(int(n_freq), dtype=complex)
    Pxx_acc = np.zeros(int(n_freq), dtype=float)
    Pyy_acc = np.zeros(int(n_freq), dtype=float)

    n_used = 0
    for s in seg_slices:
        tt = t[s]
        xx = x[s]
        yy = y[s]
        if tt.size < int(IRREGULAR_SEGMENT_MIN_POINTS):
            continue
        tt = tt - float(tt.min())
        E = np.exp(-1j * omega[:, None] * tt[None, :])
        X = E @ xx
        Y = E @ yy
        Pxy_acc += X * np.conjugate(Y)
        Pxx_acc += (np.abs(X) ** 2).real
        Pyy_acc += (np.abs(Y) ** 2).real
        n_used += 1

    if n_used < 2:
        return {
            'coherence': float('nan'),
            'phase_alignment': float('nan'),
            'weighted_phase': float('nan'),
            'n_segments': float(n_used),
        }

    Pxy = Pxy_acc / float(n_used)
    Pxx = Pxx_acc / float(n_used)
    Pyy = Pyy_acc / float(n_used)
    denom = np.sqrt(Pxx * Pyy)

    mask = denom > 0
    if not np.any(mask):
        return {
            'coherence': float('nan'),
            'phase_alignment': float('nan'),
            'weighted_phase': float('nan'),
            'n_segments': float(n_used),
        }

    coh = np.zeros_like(denom, dtype=float)
    coh[mask] = np.abs(Pxy[mask]) / denom[mask]
    coh = np.clip(coh, 0.0, 1.0)

    weights = coh[mask]
    if weights.size == 0 or float(np.sum(weights)) <= 0:
        return {
            'coherence': float('nan'),
            'phase_alignment': float('nan'),
            'weighted_phase': float('nan'),
            'n_segments': float(n_used),
        }

    phases = np.angle(Pxy[mask])
    weighted_complex = np.average(np.exp(1j * phases), weights=weights)
    weighted_phase = float(np.angle(weighted_complex))
    phase_alignment = float(math.cos(weighted_phase))
    mean_coherence = float(np.mean(coh[mask]))

    return {
        'coherence': float(mean_coherence),
        'phase_alignment': float(phase_alignment),
        'weighted_phase': float(weighted_phase),
        'n_segments': float(n_used),
    }


def analyze_interstation_irregular_phase(
    df: pd.DataFrame,
    station_coords: Dict[int, Dict[str, object]],
    n_permutations: int = 200,
) -> Dict:
    logger.info("Computing inter-station irregular-sampling phase coherence (exploratory)...")

    results: Dict[str, object] = {
        'pairs': [],
        'distance_binned': {},
        'fit_results': {},
        'exploratory': True,
        'min_shared_bins': int(MIN_SHARED_BINS_INTERSTATION),
        'n_freq': int(IRREGULAR_N_FREQ),
        'frequency_band_hz': [float(F1_HZ), float(F2_HZ)],
    }

    if df.empty:
        return results

    df = df.copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'], format='mixed')
    df['time_bin'] = df['epoch'].dt.floor(PASS_TIME_BIN)

    bias = df.groupby('station')['residual_m'].mean()
    df['anom_mm'] = (df['residual_m'] - df['station'].map(bias).fillna(0.0)) * 1000.0

    # Station bin means
    station_bins: Dict[int, pd.Series] = {}
    for sta in sorted(df['station'].unique().tolist()):
        sta_i = int(sta)
        if sta_i not in station_coords:
            continue
        sub = df[df['station'] == sta_i]
        if sub.empty:
            continue
        s = sub.groupby('time_bin')['anom_mm'].mean().sort_index()
        if int(s.size) >= MIN_SHARED_BINS_INTERSTATION:
            station_bins[sta_i] = s

    stas = list(station_bins.keys())
    if len(stas) < 2:
        return results

    pairs: List[Dict[str, object]] = []
    pair_data_for_null: List[Dict[str, object]] = []
    for i, sta1 in enumerate(stas):
        for sta2 in stas[i + 1:]:
            s1 = station_bins[sta1]
            s2 = station_bins[sta2]
            joined = pd.concat([s1.rename('x'), s2.rename('y')], axis=1, join='inner').dropna()
            if int(len(joined)) < MIN_SHARED_BINS_INTERSTATION:
                continue
            t_sec = (joined.index.astype('int64') / 1e9).to_numpy(dtype=float)
            x = joined['x'].to_numpy(dtype=float)
            y = joined['y'].to_numpy(dtype=float)

            coord1 = station_coords[sta1]
            coord2 = station_coords[sta2]
            baseline_km = haversine_distance_km(coord1['lat'], coord1['lon'], coord2['lat'], coord2['lon'])

            pc = _compute_irregular_phase_coherence(t_sec, x, y)
            if np.isnan(pc['coherence']):
                continue

            pairs.append({
                'station_1': int(sta1),
                'station_2': int(sta2),
                'name_1': coord1['name'],
                'name_2': coord2['name'],
                'baseline_km': float(baseline_km),
                'n_points': int(len(joined)),
                **pc,
            })

            pair_data_for_null.append(
                {
                    'baseline_km': float(baseline_km),
                    't_sec': np.asarray(t_sec, dtype=float),
                    'x': np.asarray(x, dtype=float),
                    'y': np.asarray(y, dtype=float),
                }
            )

    results['pairs'] = pairs

    if pair_data_for_null:
        pair_df = pd.DataFrame(pairs)
        obs_bin_means: Dict[str, float] = {}
        for i in range(len(DISTANCE_BINS_KM) - 1):
            d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
            sel = pair_df[(pair_df['baseline_km'] >= d_lo) & (pair_df['baseline_km'] < d_hi)]
            if len(sel) < MIN_PAIRS_PER_DISTANCE_BIN:
                continue
            obs_bin_means[f"{d_lo}-{d_hi}km"] = float(sel['phase_alignment'].mean())

        if obs_bin_means:
            obs_stat = float(np.nanmin(np.array(list(obs_bin_means.values()), dtype=float)))
            rng = np.random.default_rng(0)

            null_stats: List[float] = []
            null_bin_means: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
            for _ in range(int(n_permutations)):
                perm_means: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
                for pd_pair in pair_data_for_null:
                    t_sec_a = np.asarray(pd_pair['t_sec'], dtype=float)
                    x_a = np.asarray(pd_pair['x'], dtype=float)
                    y_a = np.asarray(pd_pair['y'], dtype=float)
                    n = int(y_a.size)
                    if n < 8:
                        continue
                    shift = int(rng.integers(1, n))
                    y_s = np.roll(y_a, shift)
                    pc_s = _compute_irregular_phase_coherence(t_sec_a, x_a, y_s)
                    pa_s = pc_s.get('phase_alignment')
                    if pa_s is None or np.isnan(pa_s):
                        continue

                    baseline_km = float(pd_pair['baseline_km'])
                    for i in range(len(DISTANCE_BINS_KM) - 1):
                        d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
                        k = f"{d_lo}-{d_hi}km"
                        if k not in perm_means:
                            continue
                        if d_lo <= baseline_km < d_hi:
                            perm_means[k].append(float(pa_s))
                            break

                perm_bin_mean_vals: List[float] = []
                for k, vals in perm_means.items():
                    if vals:
                        m = float(np.mean(vals))
                        null_bin_means[k].append(m)
                        perm_bin_mean_vals.append(m)
                    else:
                        null_bin_means[k].append(np.nan)

                if perm_bin_mean_vals:
                    null_stats.append(float(np.nanmin(np.array(perm_bin_mean_vals, dtype=float))))

            if null_stats:
                null_arr = np.array(null_stats, dtype=float)
                p_fwer = float((np.sum(null_arr <= obs_stat) + 1) / (len(null_arr) + 1))

                per_bin_p: Dict[str, Optional[float]] = {}
                for k, obs_mean in obs_bin_means.items():
                    arr = np.asarray([v for v in null_bin_means.get(k, []) if not np.isnan(v)], dtype=float)
                    if arr.size == 0:
                        per_bin_p[k] = None
                    else:
                        per_bin_p[k] = float((np.sum(arr <= float(obs_mean)) + 1) / (len(arr) + 1))

                results['null_tests'] = {
                    'phase_alignment_mean': {
                        'method': 'min_mean_over_bins_fwer_irregular_phase_alignment',
                        'distance_bins_km': [float(x) for x in DISTANCE_BINS_KM],
                        'min_pairs_per_bin': int(MIN_PAIRS_PER_DISTANCE_BIN),
                        'n_permutations': int(len(null_arr)),
                        'observed_bin_means': {k: float(v) for k, v in obs_bin_means.items()},
                        'observed_statistic': float(obs_stat),
                        'p_value_one_sided_le': float(p_fwer),
                        'p_values_one_sided_le_by_bin': per_bin_p,
                    }
                }

    summarized = _summarize_interstation_pairs(pairs)
    results['distance_binned'] = summarized.get('distance_binned', {})
    results['fit_results'] = summarized.get('fit_results', {})
    return results


def _summarize_interstation_pairs(
    pairs: List[Dict[str, object]],
    null_test_p_values: Optional[List[float]] = None,
) -> Dict:
    results: Dict[str, object] = {
        'pairs': pairs,
        'distance_binned': {},
        'fit_results': {},
    }

    if null_test_p_values:
        pvals = [float(p) for p in null_test_p_values if p is not None and 0 < float(p) <= 1]
        if pvals:
            stat = float(-2.0 * np.sum(np.log(np.array(pvals, dtype=float))))
            dof = 2 * len(pvals)
            p_combined = float(1.0 - stats.chi2.cdf(stat, dof))
            results['null_tests'] = {
                'phase_alignment_mean': {
                    'method': 'fisher',
                    'n_inputs': int(len(pvals)),
                    'chi2_stat': float(stat),
                    'dof': int(dof),
                    'p_value_combined': float(p_combined),
                    'p_values_inputs': [float(p) for p in pvals],
                }
            }

    if not pairs:
        return results

    pair_df = pd.DataFrame(pairs)

    for i in range(len(DISTANCE_BINS_KM) - 1):
        d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
        bin_data = pair_df[(pair_df['baseline_km'] >= d_lo) & (pair_df['baseline_km'] < d_hi)]

        if len(bin_data) > 0:
            results['distance_binned'][f"{d_lo}-{d_hi}km"] = {
                'n_pairs': int(len(bin_data)),
                'mean_distance_km': float(bin_data['baseline_km'].mean()),
                'mean_coherence': float(bin_data['coherence'].mean()),
                'std_coherence': float(bin_data['coherence'].std()) if len(bin_data) > 1 else 0.0,
                'mean_phase_alignment': float(bin_data['phase_alignment'].mean()),
                'std_phase_alignment': float(bin_data['phase_alignment'].std()) if len(bin_data) > 1 else 0.0,
            }

    if len(pair_df) < 3:
        return results

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
                maxfev=2000,
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
                maxfev=2000,
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


def analyze_daily_aggregation_correlations(
    df: pd.DataFrame,
    station_coords: Dict[int, Dict[str, object]],
    n_permutations: int = 2000,
    top_n_stations: int = TOP_STATIONS_COUNT,
) -> Dict:
    """
    Analyze inter-station correlations using daily-aggregated residuals.
    
    This approach dramatically increases overlap compared to pass-bin methods:
    - Aggregate residuals to daily means per station
    - Compute correlations between station pairs across shared days
    - Bin by baseline distance and fit exponential decay
    
    This is the highest-impact improvement for detecting distance-structured
    correlations in sparse SLR data.
    """
    logger.info("Analyzing daily-aggregation inter-station correlations...")
    
    results: Dict[str, object] = {
        'pairs': [],
        'distance_binned': {},
        'fit_results': {},
        'top_stations_used': int(top_n_stations),
        'method': 'daily_aggregation',
    }
    
    if df.empty:
        return results
    
    df = df.copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'], format='mixed')
    df['date'] = df['epoch'].dt.date
    
    # Remove station biases
    station_biases = df.groupby('station')['residual_m'].mean()
    df['residual_debiased'] = df['residual_m'] - df['station'].map(station_biases)
    
    # Identify top stations by observation count
    station_counts = df.groupby('station').size().sort_values(ascending=False)
    top_stations = station_counts.head(int(top_n_stations)).index.tolist()
    top_stations = [int(s) for s in top_stations if int(s) in station_coords]
    
    logger.info(f"  Using top {len(top_stations)} stations by observation count")
    
    # Build daily mean series for each top station
    daily_series: Dict[int, pd.Series] = {}
    for sta in top_stations:
        sta_df = df[df['station'] == sta]
        daily = sta_df.groupby('date').agg({
            'residual_debiased': ['mean', 'count']
        })
        daily.columns = ['mean', 'count']
        # Filter days with minimum observations
        daily = daily[daily['count'] >= DAILY_AGG_MIN_OBS_PER_DAY]
        
        if len(daily) >= DAILY_AGG_MIN_DAYS_PER_STATION:
            daily_series[sta] = daily['mean']
            logger.info(f"    Station {sta}: {len(daily)} days with ≥{DAILY_AGG_MIN_OBS_PER_DAY} obs/day")
    
    if len(daily_series) < 2:
        logger.warning("  Not enough stations with sufficient daily data")
        return results
    
    # Compute correlations for all station pairs
    pair_data_for_null: List[Dict[str, object]] = []
    station_list = list(daily_series.keys())
    
    for i, sta1 in enumerate(station_list):
        for sta2 in station_list[i+1:]:
            if sta1 not in station_coords or sta2 not in station_coords:
                continue
            
            coord1 = station_coords[sta1]
            coord2 = station_coords[sta2]
            baseline_km = haversine_distance_km(
                coord1['lat'], coord1['lon'],
                coord2['lat'], coord2['lon']
            )
            
            # Find shared days
            s1 = daily_series[sta1]
            s2 = daily_series[sta2]
            shared_idx = s1.index.intersection(s2.index)
            
            if len(shared_idx) < DAILY_AGG_MIN_SHARED_DAYS:
                continue
            
            r1 = s1.loc[shared_idx].values
            r2 = s2.loc[shared_idx].values
            
            # Compute Pearson correlation
            corr = _pearson_corr(r1, r2)
            if np.isnan(corr):
                continue
            
            # Also compute lagged correlations (TEP prediction: lag structure)
            lag_corrs = {}
            for lag in [-3, -2, -1, 0, 1, 2, 3]:
                if lag == 0:
                    lag_corrs[lag] = float(corr)
                else:
                    s1_shifted = s1.shift(lag)
                    shared_lag = s1_shifted.index.intersection(s2.index)
                    shared_lag = [d for d in shared_lag if d in s1_shifted.dropna().index]
                    if len(shared_lag) >= DAILY_AGG_MIN_SHARED_DAYS:
                        r1_lag = s1_shifted.loc[shared_lag].values
                        r2_lag = s2.loc[shared_lag].values
                        lag_corrs[lag] = float(_pearson_corr(r1_lag, r2_lag))
            
            pair_entry = {
                'station_1': int(sta1),
                'station_2': int(sta2),
                'name_1': coord1.get('name', str(sta1)),
                'name_2': coord2.get('name', str(sta2)),
                'baseline_km': float(baseline_km),
                'n_shared_days': int(len(shared_idx)),
                'correlation': float(corr),
                'lag_correlations': lag_corrs,
            }
            results['pairs'].append(pair_entry)
            
            pair_data_for_null.append({
                'baseline_km': float(baseline_km),
                'r1': np.asarray(r1, dtype=float),
                'r2': np.asarray(r2, dtype=float),
            })
    
    logger.info(f"  Computed correlations for {len(results['pairs'])} station pairs")
    
    # Family-wise permutation test
    if pair_data_for_null:
        pair_df = pd.DataFrame(results['pairs'])
        obs_bin_means: Dict[str, float] = {}
        
        for i in range(len(DISTANCE_BINS_KM) - 1):
            d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
            sel = pair_df[(pair_df['baseline_km'] >= d_lo) & (pair_df['baseline_km'] < d_hi)]
            if len(sel) >= MIN_PAIRS_PER_DISTANCE_BIN:
                obs_bin_means[f"{d_lo}-{d_hi}km"] = float(sel['correlation'].mean())
        
        if obs_bin_means:
            # Look for most negative bin (TEP predicts anti-correlation at certain distances)
            obs_stat = float(np.nanmin(np.array(list(obs_bin_means.values()), dtype=float)))
            rng = np.random.default_rng(42)
            
            null_stats: List[float] = []
            null_bin_means: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
            
            for _ in range(int(n_permutations)):
                perm_bin_vals: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
                
                for pd_pair in pair_data_for_null:
                    r1 = np.asarray(pd_pair['r1'], dtype=float)
                    r2 = np.asarray(pd_pair['r2'], dtype=float)
                    n = int(r2.size)
                    if n < DAILY_AGG_MIN_SHARED_DAYS:
                        continue
                    
                    # Circular shift to destroy synchrony
                    shift = int(rng.integers(1, n))
                    r2s = np.roll(r2, shift)
                    c = _pearson_corr(r1, r2s)
                    if np.isnan(c):
                        continue
                    
                    baseline = float(pd_pair['baseline_km'])
                    for j in range(len(DISTANCE_BINS_KM) - 1):
                        d_lo, d_hi = DISTANCE_BINS_KM[j], DISTANCE_BINS_KM[j + 1]
                        k = f"{d_lo}-{d_hi}km"
                        if k not in perm_bin_vals:
                            continue
                        if d_lo <= baseline < d_hi:
                            perm_bin_vals[k].append(float(c))
                            break
                
                perm_means_list: List[float] = []
                for k, vals in perm_bin_vals.items():
                    if vals:
                        m = float(np.mean(vals))
                        null_bin_means[k].append(m)
                        perm_means_list.append(m)
                    else:
                        null_bin_means[k].append(np.nan)
                
                if perm_means_list:
                    null_stats.append(float(np.nanmin(np.array(perm_means_list, dtype=float))))
            
            if null_stats:
                null_arr = np.array(null_stats, dtype=float)
                p_fwer = float((np.sum(null_arr <= obs_stat) + 1) / (len(null_arr) + 1))
                
                per_bin_p: Dict[str, Optional[float]] = {}
                for k, obs_mean in obs_bin_means.items():
                    arr = np.asarray([v for v in null_bin_means.get(k, []) if not np.isnan(v)], dtype=float)
                    if arr.size == 0:
                        per_bin_p[k] = None
                    else:
                        per_bin_p[k] = float((np.sum(arr <= float(obs_mean)) + 1) / (len(arr) + 1))
                
                results['null_tests'] = {
                    'method': 'daily_aggregation_fwer',
                    'distance_bins_km': [float(x) for x in DISTANCE_BINS_KM],
                    'min_pairs_per_bin': int(MIN_PAIRS_PER_DISTANCE_BIN),
                    'n_permutations': int(len(null_arr)),
                    'observed_bin_means': {k: float(v) for k, v in obs_bin_means.items()},
                    'observed_statistic': float(obs_stat),
                    'p_value_one_sided_le': float(p_fwer),
                    'p_values_one_sided_le_by_bin': per_bin_p,
                }
    
    # Bin by distance
    if results['pairs']:
        pair_df = pd.DataFrame(results['pairs'])
        
        for i in range(len(DISTANCE_BINS_KM) - 1):
            d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
            bin_data = pair_df[(pair_df['baseline_km'] >= d_lo) & (pair_df['baseline_km'] < d_hi)]
            
            if len(bin_data) > 0:
                results['distance_binned'][f"{d_lo}-{d_hi}km"] = {
                    'n_pairs': int(len(bin_data)),
                    'mean_distance_km': float(bin_data['baseline_km'].mean()),
                    'mean_correlation': float(bin_data['correlation'].mean()),
                    'std_correlation': float(bin_data['correlation'].std()) if len(bin_data) > 1 else 0.0,
                    'median_correlation': float(bin_data['correlation'].median()),
                }
        
        # Fit exponential decay
        if len(pair_df) >= 3:
            try:
                distances = pair_df['baseline_km'].values
                correlations = pair_df['correlation'].values
                valid = ~np.isnan(correlations)
                
                if np.sum(valid) >= 3:
                    popt, pcov = curve_fit(
                        exponential_decay,
                        distances[valid],
                        correlations[valid],
                        p0=[0.3, TEP_COHERENCE_LENGTH_KM, 0.0],
                        bounds=([-1, 100, -1], [1, 20000, 1]),
                        maxfev=2000,
                    )
                    results['fit_results'] = {
                        'amplitude': float(popt[0]),
                        'lambda_km': float(popt[1]),
                        'offset': float(popt[2]),
                        'lambda_uncertainty_km': float(np.sqrt(pcov[1, 1])) if pcov[1, 1] > 0 else None,
                    }
                    logger.info(f"  Fitted λ = {popt[1]:.0f} km (GNSS prediction: {TEP_COHERENCE_LENGTH_KM} km)")
            except Exception as e:
                logger.warning(f"  Fit failed: {e}")
    
    return results


def analyze_lagged_cross_correlations(
    df: pd.DataFrame,
    station_coords: Dict[int, Dict[str, object]],
    max_lag_days: int = 7,
    top_n_stations: int = TOP_STATIONS_COUNT,
) -> Dict:
    """
    Analyze lagged cross-correlations between station pairs.
    
    TEP predicts that correlations should peak at lag ≈ 0 for nearby stations
    and shift systematically with distance as the scalar field gradient sweeps across.
    """
    logger.info("Analyzing lagged cross-correlations...")
    
    results: Dict[str, object] = {
        'pairs': [],
        'lag_structure_by_distance': {},
        'method': 'lagged_xcorr',
        'max_lag_days': int(max_lag_days),
    }
    
    if df.empty:
        return results
    
    df = df.copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'], format='mixed')
    df['date'] = df['epoch'].dt.date
    
    # Remove station biases
    station_biases = df.groupby('station')['residual_m'].mean()
    df['residual_debiased'] = df['residual_m'] - df['station'].map(station_biases)
    
    # Identify top stations
    station_counts = df.groupby('station').size().sort_values(ascending=False)
    top_stations = station_counts.head(int(top_n_stations)).index.tolist()
    top_stations = [int(s) for s in top_stations if int(s) in station_coords]
    
    # Build daily mean series
    daily_series: Dict[int, pd.Series] = {}
    for sta in top_stations:
        sta_df = df[df['station'] == sta]
        daily = sta_df.groupby('date')['residual_debiased'].mean()
        if len(daily) >= DAILY_AGG_MIN_DAYS_PER_STATION:
            daily_series[sta] = daily
    
    if len(daily_series) < 2:
        return results
    
    station_list = list(daily_series.keys())
    
    for i, sta1 in enumerate(station_list):
        for sta2 in station_list[i+1:]:
            if sta1 not in station_coords or sta2 not in station_coords:
                continue
            
            coord1 = station_coords[sta1]
            coord2 = station_coords[sta2]
            baseline_km = haversine_distance_km(
                coord1['lat'], coord1['lon'],
                coord2['lat'], coord2['lon']
            )
            
            s1 = daily_series[sta1]
            s2 = daily_series[sta2]
            
            # Compute lagged correlations
            lag_corrs: Dict[int, float] = {}
            peak_lag = 0
            peak_corr = -2.0
            
            for lag in range(-max_lag_days, max_lag_days + 1):
                s1_shifted = s1.shift(lag)
                shared = s1_shifted.dropna().index.intersection(s2.index)
                
                if len(shared) >= DAILY_AGG_MIN_SHARED_DAYS:
                    r1 = s1_shifted.loc[shared].values
                    r2 = s2.loc[shared].values
                    corr = _pearson_corr(r1, r2)
                    if not np.isnan(corr):
                        lag_corrs[lag] = float(corr)
                        if corr > peak_corr:
                            peak_corr = corr
                            peak_lag = lag
            
            if lag_corrs:
                results['pairs'].append({
                    'station_1': int(sta1),
                    'station_2': int(sta2),
                    'baseline_km': float(baseline_km),
                    'lag_correlations': lag_corrs,
                    'peak_lag_days': int(peak_lag),
                    'peak_correlation': float(peak_corr),
                    'correlation_at_lag0': lag_corrs.get(0),
                })
    
    # Analyze lag structure by distance bin
    if results['pairs']:
        pair_df = pd.DataFrame(results['pairs'])
        
        for i in range(len(DISTANCE_BINS_KM) - 1):
            d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
            bin_data = pair_df[(pair_df['baseline_km'] >= d_lo) & (pair_df['baseline_km'] < d_hi)]
            
            if len(bin_data) >= MIN_PAIRS_PER_DISTANCE_BIN:
                # Average lag correlations across pairs in this bin
                avg_lag_corrs: Dict[int, List[float]] = {}
                for _, row in bin_data.iterrows():
                    for lag, corr in row['lag_correlations'].items():
                        if lag not in avg_lag_corrs:
                            avg_lag_corrs[lag] = []
                        avg_lag_corrs[lag].append(corr)
                
                mean_lag_corrs = {lag: float(np.mean(vals)) for lag, vals in avg_lag_corrs.items()}
                
                results['lag_structure_by_distance'][f"{d_lo}-{d_hi}km"] = {
                    'n_pairs': int(len(bin_data)),
                    'mean_distance_km': float(bin_data['baseline_km'].mean()),
                    'mean_peak_lag_days': float(bin_data['peak_lag_days'].mean()),
                    'mean_lag_correlations': mean_lag_corrs,
                }
    
    logger.info(f"  Analyzed lagged correlations for {len(results['pairs'])} pairs")
    return results


def analyze_interstation_mwpc(df: pd.DataFrame, station_coords: Dict[int, Dict[str, object]], n_permutations: int = 200) -> Dict:
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
    station_series: Dict[int, pd.Series] = {}
    
    for sta in stations:
        sta_i = int(sta)
        if sta_i not in station_coords:
            continue

        s = build_station_resampled_series(df, sta_i, resample_interval=MWPC_RESAMPLE_INTERVAL)
        if int(s.notna().sum()) >= MIN_POINTS_MWPC:
            station_series[sta_i] = s
            logger.info(f"  Station {sta_i}: {int(s.notna().sum())} resampled points")
    
    if len(station_series) < 2:
        logger.warning("Not enough stations with sufficient data for MWPC")
        return results
    
    station_list = list(station_series.keys())

    pair_data_for_null: List[Dict[str, object]] = []
    
    for i, sta1 in enumerate(station_list):
        for sta2 in station_list[i+1:]:
            # Get coordinates
            coord1 = station_coords[sta1]
            coord2 = station_coords[sta2]
            baseline_km = haversine_distance_km(coord1['lat'], coord1['lon'], 
                                                 coord2['lat'], coord2['lon'])
            
            s1 = station_series.get(int(sta1))
            s2 = station_series.get(int(sta2))
            if s1 is None or s2 is None:
                continue

            joined = pd.concat([s1.rename('r1'), s2.rename('r2')], axis=1, join='outer')
            joined = joined.sort_index()
            seg = _longest_contiguous_overlap_segment(joined, pd.to_timedelta(MWPC_RESAMPLE_INTERVAL))
            if seg is None:
                continue
            if len(seg) < MIN_POINTS_MWPC:
                continue

            r1_seg = seg['r1'].to_numpy(dtype=float)
            r2_seg = seg['r2'].to_numpy(dtype=float)
            mwpc = compute_mwpc(r1_seg, r2_seg, fs=FS_HZ, f1=F1_HZ, f2=F2_HZ)

            if not np.isnan(mwpc['coherence']):
                results['pairs'].append({
                    'station_1': int(sta1),
                    'station_2': int(sta2),
                    'name_1': coord1['name'],
                    'name_2': coord2['name'],
                    'baseline_km': float(baseline_km),
                    'n_points': int(len(seg)),
                    **mwpc,
                })

                pair_data_for_null.append(
                    {
                        'baseline_km': float(baseline_km),
                        'r1': r1_seg,
                        'r2': r2_seg,
                    }
                )
    
    logger.info(f"  Computed MWPC for {len(results['pairs'])} station pairs")

    if pair_data_for_null:
        pair_df = pd.DataFrame(pair_data_for_null)
        obs_bin_means: Dict[str, float] = {}
        for i in range(len(DISTANCE_BINS_KM) - 1):
            d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
            sel = pair_df[(pair_df['baseline_km'] >= d_lo) & (pair_df['baseline_km'] < d_hi)]
            if len(sel) < MIN_PAIRS_PER_DISTANCE_BIN:
                continue
            vals: List[float] = []
            for row in sel.itertuples(index=False):
                mw = compute_mwpc(np.asarray(row.r1, dtype=float), np.asarray(row.r2, dtype=float), fs=FS_HZ, f1=F1_HZ, f2=F2_HZ)
                if not np.isnan(mw['phase_alignment']):
                    vals.append(float(mw['phase_alignment']))
            if vals:
                obs_bin_means[f"{d_lo}-{d_hi}km"] = float(np.mean(vals))

        if obs_bin_means:
            obs_stat = float(np.nanmin(np.array(list(obs_bin_means.values()), dtype=float)))
            rng = np.random.default_rng(0)

            null_stats: List[float] = []
            null_bin_means: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
            for _ in range(int(n_permutations)):
                perm_means: Dict[str, List[float]] = {k: [] for k in obs_bin_means.keys()}
                for pd_pair in pair_data_for_null:
                    r1a = np.asarray(pd_pair['r1'], dtype=float)
                    r2a = np.asarray(pd_pair['r2'], dtype=float)
                    n = len(r2a)
                    if n < 8:
                        continue
                    shift = int(rng.integers(1, n))
                    r2s = np.roll(r2a, shift)
                    mw = compute_mwpc(r1a, r2s, fs=FS_HZ, f1=F1_HZ, f2=F2_HZ)
                    if np.isnan(mw['phase_alignment']):
                        continue
                    baseline_km = float(pd_pair['baseline_km'])
                    for i in range(len(DISTANCE_BINS_KM) - 1):
                        d_lo, d_hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
                        k = f"{d_lo}-{d_hi}km"
                        if k not in perm_means:
                            continue
                        if d_lo <= baseline_km < d_hi:
                            perm_means[k].append(float(mw['phase_alignment']))
                            break

                perm_bin_mean_vals: List[float] = []
                for k, vals in perm_means.items():
                    if vals:
                        m = float(np.mean(vals))
                        null_bin_means[k].append(m)
                        perm_bin_mean_vals.append(m)
                    else:
                        null_bin_means[k].append(np.nan)

                if perm_bin_mean_vals:
                    null_stats.append(float(np.nanmin(np.array(perm_bin_mean_vals, dtype=float))))

            if null_stats:
                null_arr = np.array(null_stats, dtype=float)
                p_fwer = float((np.sum(null_arr <= obs_stat) + 1) / (len(null_arr) + 1))

                per_bin_p: Dict[str, Optional[float]] = {}
                for k, obs_mean in obs_bin_means.items():
                    arr = np.asarray([v for v in null_bin_means.get(k, []) if not np.isnan(v)], dtype=float)
                    if arr.size == 0:
                        per_bin_p[k] = None
                    else:
                        per_bin_p[k] = float((np.sum(arr <= float(obs_mean)) + 1) / (len(arr) + 1))

                results['null_tests'] = {
                    'phase_alignment_mean': {
                        'method': 'min_mean_over_bins_fwer',
                        'distance_bins_km': [float(x) for x in DISTANCE_BINS_KM],
                        'min_pairs_per_bin': int(MIN_PAIRS_PER_DISTANCE_BIN),
                        'n_permutations': int(len(null_arr)),
                        'observed_bin_means': {k: float(v) for k, v in obs_bin_means.items()},
                        'observed_statistic': float(obs_stat),
                        'p_value_one_sided_le': float(p_fwer),
                        'p_values_one_sided_le_by_bin': per_bin_p,
                    }
                }
    
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


def analyze_range_coherence(df: pd.DataFrame, bootstrap_n: int = DEFAULT_BOOTSTRAP_N) -> Dict:
    """
    Analyze coherence as a function of station-satellite range.
    
    TEP predicts that ranging measurements should show distance-structured
    correlations. We bin observations by range and compute coherence metrics.
    """
    logger.info("Analyzing range-dependent coherence...")
    
    results: Dict[str, object] = {
        'range_bins': [],
        'correlation_with_range': {},
        'ratio_summary': {},
    }
    
    # Remove station biases
    df = df.copy()
    df['residual_debiased'] = df.groupby('station')['residual_m'].transform(lambda x: x - x.mean())
    
    # Bin by range
    range_bins = np.linspace(float(df['model_range_m'].min()), float(df['model_range_m'].max()), 8)
    df['range_bin'] = pd.cut(df['model_range_m'], bins=range_bins)

    max_gap_s = 450.0
    step_s = 300.0

    for bin_label, group in df.groupby('range_bin', observed=True):
        if len(group) < 25:
            continue

        group = group.copy()
        group['epoch'] = pd.to_datetime(group['epoch_utc'], format='mixed')
        group['time_bin'] = group['epoch'].dt.floor('5min')
        mid_range_km = float(group['model_range_m'].mean() / 1000.0)

        station_autocorrs: List[float] = []
        station_pairs: List[int] = []
        for sta, grp in group.groupby('station'):
            # Use 5-minute bin means to avoid trivial within-pass autocorrelation.
            g = grp.groupby('time_bin')['residual_debiased'].mean().sort_index()
            if int(g.size) < 4:
                continue
            r = g.to_numpy(dtype=float)
            t = (g.index.astype('int64') / 1e9).to_numpy(dtype=float)
            ac, n_pairs = _lag1_autocorr_max_gap(r, t, max_gap_s=max_gap_s)
            if not np.isnan(ac) and int(n_pairs) >= 3:
                station_autocorrs.append(float(ac))
                station_pairs.append(int(n_pairs))

        se = float(np.std(np.array(station_autocorrs, dtype=float), ddof=1) / math.sqrt(len(station_autocorrs))) if len(station_autocorrs) > 1 else 0.0
        results['range_bins'].append({
            'range_km': float(mid_range_km),
            'n_obs': int(len(group)),
            'n_stations': int(len(station_autocorrs)),
            'mean_lag_pairs_per_station': float(np.mean(np.array(station_pairs, dtype=float))) if station_pairs else 0.0,
            'mean_residual_mm': float(group['residual_debiased'].mean() * 1000),
            'std_residual_mm': float(group['residual_debiased'].std(ddof=1) * 1000) if int(len(group)) > 1 else 0.0,
            'autocorrelation_mean': float(np.mean(np.array(station_autocorrs, dtype=float))) if station_autocorrs else None,
            'autocorrelation_stderr': float(se),
        })

    high_max_m = 6500.0e3
    low_min_m = 8000.0e3
    high_by_sta: Dict[int, float] = {}
    low_by_sta: Dict[int, float] = {}

    df_h = df[(df['model_range_m'] >= float(df['model_range_m'].min())) & (df['model_range_m'] < float(high_max_m))].copy()
    df_l = df[(df['model_range_m'] >= float(low_min_m)) & (df['model_range_m'] < float(df['model_range_m'].max()) + 1.0)].copy()
    df_h['epoch'] = pd.to_datetime(df_h['epoch_utc'], format='mixed')
    df_l['epoch'] = pd.to_datetime(df_l['epoch_utc'], format='mixed')
    df_h['time_bin'] = df_h['epoch'].dt.floor('5min')
    df_l['time_bin'] = df_l['epoch'].dt.floor('5min')

    for sta, grp in df_h.groupby('station'):
        g = grp.groupby('time_bin')['residual_debiased'].mean().sort_index()
        if int(g.size) < 4:
            continue
        r = g.to_numpy(dtype=float)
        t = (g.index.astype('int64') / 1e9).to_numpy(dtype=float)
        ac, n_pairs = _lag1_autocorr_max_gap(r, t, max_gap_s=max_gap_s)
        if not np.isnan(ac) and int(n_pairs) >= 3:
            high_by_sta[int(sta)] = float(ac)

    for sta, grp in df_l.groupby('station'):
        g = grp.groupby('time_bin')['residual_debiased'].mean().sort_index()
        if int(g.size) < 4:
            continue
        r = g.to_numpy(dtype=float)
        t = (g.index.astype('int64') / 1e9).to_numpy(dtype=float)
        ac, n_pairs = _lag1_autocorr_max_gap(r, t, max_gap_s=max_gap_s)
        if not np.isnan(ac) and int(n_pairs) >= 3:
            low_by_sta[int(sta)] = float(ac)

    shared_stas = sorted(set(high_by_sta.keys()).intersection(set(low_by_sta.keys())))
    high_vals = np.array([high_by_sta[s] for s in shared_stas], dtype=float)
    low_vals = np.array([low_by_sta[s] for s in shared_stas], dtype=float)
    ratio = float(np.mean(low_vals) / np.mean(high_vals)) if (high_vals.size > 0 and np.mean(high_vals) != 0.0) else float('nan')

    delta = float(np.mean(low_vals) - np.mean(high_vals)) if (high_vals.size > 0 and low_vals.size > 0) else float('nan')

    boot_high = _bootstrap_mean(high_vals, int(bootstrap_n), seed=1)
    boot_low = _bootstrap_mean(low_vals, int(bootstrap_n), seed=2)

    if int(bootstrap_n) > 0 and high_vals.size > 0 and low_vals.size > 0:
        rng = np.random.default_rng(3)
        idx = rng.integers(0, int(high_vals.size), size=(int(bootstrap_n), int(high_vals.size)))
        rh = np.mean(high_vals[idx], axis=1)
        rl = np.mean(low_vals[idx], axis=1)
        rr = rl / rh
        rr = rr[np.isfinite(rr)]
        if rr.size:
            lo, hi = np.quantile(rr, [0.025, 0.975])
            ratio_ci = [float(lo), float(hi)]
        else:
            ratio_ci = [None, None]

        dd = rl - rh
        dd = dd[np.isfinite(dd)]
        if dd.size:
            lo, hi = np.quantile(dd, [0.025, 0.975])
            delta_ci = [float(lo), float(hi)]
        else:
            delta_ci = [None, None]
    else:
        ratio_ci = [None, None]
        delta_ci = [None, None]

    results['ratio_summary'] = {
        'high_range_max_km': float(high_max_m / 1000.0),
        'low_range_min_km': float(low_min_m / 1000.0),
        'n_stations_shared': int(len(shared_stas)),
        'high_autocorr': boot_high,
        'low_autocorr': boot_low,
        'ratio_low_over_high': {
            'value': float(ratio) if np.isfinite(ratio) else None,
            'ci95': ratio_ci,
            'bootstrap_n': int(bootstrap_n),
        },
        'delta_low_minus_high': {
            'value': float(delta) if np.isfinite(delta) else None,
            'ci95': delta_ci,
            'bootstrap_n': int(bootstrap_n),
        },
    }
    
    # Overall range-residual correlation
    r, p = stats.pearsonr(df['model_range_m'], df['residual_debiased'])
    results['correlation_with_range'] = {
        'pearson_r': float(r),
        'p_value': float(p),
        'significant': p < 0.05,
    }
    
    return results


def _station_spectral_ratios(df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for sta in df['station'].unique():
        times, residuals = build_station_time_series(df, sta, resample_interval='5min')
        if residuals.size < 64:
            continue
        residuals = detrend(residuals, type='linear')
        try:
            freqs, psd = welch(residuals, fs=FS_HZ, nperseg=min(256, len(residuals)//2))
        except Exception:
            continue
        tep_mask = (freqs >= F1_HZ) & (freqs <= F2_HZ)
        total_mask = freqs > 0
        bb_mask = freqs >= BROADBAND_MIN_HZ
        if not np.any(tep_mask) or not np.any(total_mask):
            continue
        tep_power = float(np.mean(psd[tep_mask]))
        total_power = float(np.mean(psd[total_mask]))
        bb_power = float(np.mean(psd[bb_mask])) if np.any(bb_mask) else float('nan')

        ratio_total = float(tep_power / total_power) if total_power > 0 else float('nan')
        ratio_bb = float(tep_power / bb_power) if (np.isfinite(bb_power) and bb_power > 0) else float('nan')

        if np.isfinite(ratio_total):
            out[int(sta)] = {
                'tep_power': float(tep_power),
                'total_power': float(total_power),
                'broadband_power': float(bb_power) if np.isfinite(bb_power) else None,
                'tep_over_total': float(ratio_total),
                'tep_over_broadband': float(ratio_bb) if np.isfinite(ratio_bb) else None,
            }
    return out


def analyze_temporal_coherence(df: pd.DataFrame, bootstrap_n: int = DEFAULT_BOOTSTRAP_N) -> Dict:
    """
    Analyze temporal coherence structure of residuals.
    
    Computes autocorrelation function and spectral properties
    to identify TEP-relevant temporal patterns.
    """
    logger.info("Analyzing temporal coherence...")
    
    results: Dict[str, object] = {
        'station_acf': {},
        'overall_spectral': {},
        'spectral_concentration': {},
    }
    
    # For each station, compute autocorrelation function
    for sta in df['station'].unique():
        sta_df = df[df['station'] == sta].copy()
        if len(sta_df) < 20:
            continue
        
        sta_df['epoch'] = pd.to_datetime(sta_df['epoch_utc'], format='mixed')
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
    
    station_spec = _station_spectral_ratios(df)
    ratios_total = np.array([v.get('tep_over_total') for v in station_spec.values() if v is not None], dtype=float)
    ratios_total = ratios_total[np.isfinite(ratios_total)]
    boot_total = _bootstrap_mean(ratios_total, int(bootstrap_n), seed=4)

    ratios_bb = np.array([v.get('tep_over_broadband') for v in station_spec.values() if v is not None], dtype=float)
    ratios_bb = ratios_bb[np.isfinite(ratios_bb)]
    boot_bb = _bootstrap_mean(ratios_bb, int(bootstrap_n), seed=5)

    results['spectral_concentration'] = {
        'frequency_band_hz': [float(F1_HZ), float(F2_HZ)],
        'broadband_min_hz': float(BROADBAND_MIN_HZ),
        'station_ratios': {str(k): v for k, v in station_spec.items()},
        'summary': {
            'tep_over_total': {
                'n_stations': int(ratios_total.size),
                'mean': boot_total.get('mean'),
                'ci95': boot_total.get('ci95'),
                'median': float(np.median(ratios_total)) if ratios_total.size else None,
                'min': float(np.min(ratios_total)) if ratios_total.size else None,
                'max': float(np.max(ratios_total)) if ratios_total.size else None,
                'bootstrap_n': int(bootstrap_n),
            },
            'tep_over_broadband': {
                'n_stations': int(ratios_bb.size),
                'mean': boot_bb.get('mean'),
                'ci95': boot_bb.get('ci95'),
                'median': float(np.median(ratios_bb)) if ratios_bb.size else None,
                'min': float(np.min(ratios_bb)) if ratios_bb.size else None,
                'max': float(np.max(ratios_bb)) if ratios_bb.size else None,
                'bootstrap_n': int(bootstrap_n),
            },
        },
    }
    
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
    global PASS_TIME_BIN
    global MIN_OBS_PER_STATION_BIN
    global MIN_SHARED_BINS_INTERSTATION

    parser = argparse.ArgumentParser(description="TEP-SLR Step 2.3: MWPC Analysis")
    parser.add_argument("--input", default=str(OUTPUTS_DIR / "step_2_1_slr_residuals.csv"),
                        help="Input residuals CSV file")
    parser.add_argument("--slrf2020", default="",
                        help="Path to local SLRF2020 SINEX (.snx or .snx.gz). If omitted, auto-detect under data/slr/products/resource.")
    parser.add_argument("--n-permutations", type=int, default=2000,
                        help="Number of circular-shift permutations for the family-wise pass-correlation null test across distance bins")
    parser.add_argument("--residual-threshold-m", type=float, default=DEFAULT_RESIDUAL_THRESHOLD_M,
                        help="Primary absolute residual threshold (m) used for the main analysis")
    parser.add_argument("--residual-thresholds-m", default=",".join(str(x) for x in DEFAULT_RESIDUAL_THRESHOLDS_M),
                        help="Comma-separated list of residual thresholds (m) used for robustness sweeps")
    parser.add_argument("--bootstrap-n", type=int, default=DEFAULT_BOOTSTRAP_N,
                        help="Number of station-bootstrap resamples used for uncertainty estimates")
    parser.add_argument("--include-pass-pairs", action="store_true",
                        help="Include raw pass-pair entries in JSON output (very large). Default: omit.")
    parser.add_argument("--pass-time-bin", default=PASS_TIME_BIN,
                        help="Time bin width for contemporaneous pass aggregation (e.g., 5min, 10min)")
    parser.add_argument("--min-obs-per-station-bin", type=int, default=MIN_OBS_PER_STATION_BIN,
                        help="Minimum observations per station within a time bin to form a pass-pair value")
    parser.add_argument("--min-shared-bins-interstation", type=int, default=MIN_SHARED_BINS_INTERSTATION,
                        help="Minimum shared time bins for exploratory irregular-phase interstation analysis")
    parser.add_argument("--pass-fwer-metric", choices=["weighted", "unweighted"], default="weighted",
                        help="Correlation metric used for the pass-based FWER null test")
    args = parser.parse_args()
    PASS_TIME_BIN = str(args.pass_time_bin)
    MIN_OBS_PER_STATION_BIN = int(args.min_obs_per_station_bin)
    MIN_SHARED_BINS_INTERSTATION = int(args.min_shared_bins_interstation)
    
    # Load residuals
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    df_all = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df_all)} residual observations from {input_path.name}")

    slrf_path = Path(args.slrf2020) if args.slrf2020 else find_local_slrf2020_sinex(PROJECT_ROOT)
    if slrf_path is None or not slrf_path.exists():
        logger.error("SLRF2020 SINEX not found. Run step_2_1_slr_residuals.py at least once to download station coordinates, or pass --slrf2020.")
        return 2

    station_coords = load_station_coords_from_slrf2020(slrf_path)
    if not station_coords:
        logger.error(f"Failed to load station coordinates from SLRF2020: {slrf_path}")
        return 2

    logger.info(f"Loaded station coordinates: {len(station_coords)} stations from {slrf_path.name}")

    primary_thr = float(args.residual_threshold_m)
    sweep_thrs = sorted(set([float(primary_thr)] + _parse_float_list_csv(str(args.residual_thresholds_m))))

    initial_count = int(len(df_all))
    df = df_all[df_all['residual_m'].abs() < float(primary_thr)].copy()
    logger.info(f"Filtered {initial_count - len(df)} outliers (|residual| > {primary_thr} m). Remaining: {len(df)}")
    
    logger.info(f"Stations: {sorted(df['station'].unique())}")
    logger.info(f"Satellites: {sorted(df['satellite'].unique())}")

    stations_in_data = sorted({int(s) for s in df['station'].unique().tolist()})
    stations_with_coords = [s for s in stations_in_data if s in station_coords]
    logger.info(f"Stations with coords: {len(stations_with_coords)}/{len(stations_in_data)}")
    
    # Run MWPC analyses
    results: Dict[str, object] = {
        'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
        'input_file': str(input_path),
        'methodology': 'Magnitude-Weighted Phase Correlation (TEP-GNSS-RINEX)',
        'tep_parameters': {
            'coherence_length_km': TEP_COHERENCE_LENGTH_KM,
            'frequency_band_hz': [F1_HZ, F2_HZ],
            'sampling_hz': FS_HZ,
        },
        'run_parameters': {
            'residual_threshold_m': float(primary_thr),
            'residual_thresholds_sweep_m': [float(x) for x in sweep_thrs],
            'bootstrap_n': int(args.bootstrap_n),
            'broadband_min_hz': float(BROADBAND_MIN_HZ),
            'n_permutations': int(args.n_permutations),
        },
        'data_summary': {
            'n_observations': len(df),
            'n_stations': int(df['station'].nunique()),
            'n_stations_with_coords': int(len(stations_with_coords)),
            'n_satellites': int(df['satellite'].nunique()),
            'date_range': [df['epoch_utc'].min(), df['epoch_utc'].max()],
        },
    }
    
    # 1. Pass-based correlations (works better with sparse SLR data)
    results['pass_correlations'] = analyze_pass_correlations(
        df,
        station_coords,
        n_permutations=int(args.n_permutations),
        fwer_metric=str(args.pass_fwer_metric),
        include_pass_pairs=bool(args.include_pass_pairs),
    )

    # 2. Inter-station irregular-sampling phase coherence (exploratory; no interpolation)
    by_sat_phase: Dict[str, Dict] = {}
    pooled_phase_pairs: List[Dict[str, object]] = []
    null_pvals_phase: List[float] = []
    for sat in sorted(df['satellite'].unique().tolist()):
        sat_df = df[df['satellite'] == sat].copy()
        sat_res = analyze_interstation_irregular_phase(
            sat_df,
            station_coords,
            n_permutations=int(args.n_permutations),
        )
        by_sat_phase[str(sat)] = sat_res
        for p in sat_res.get('pairs', []):
            pp = dict(p)
            pp['satellite'] = str(sat)
            pooled_phase_pairs.append(pp)

        nt = sat_res.get('null_tests', {}).get('phase_alignment_mean', {})
        pv = nt.get('p_value_one_sided_le')
        if pv is not None:
            null_pvals_phase.append(float(pv))
    results['interstation_irregular_phase_by_satellite'] = by_sat_phase
    results['interstation_irregular_phase'] = _summarize_interstation_pairs(
        pooled_phase_pairs,
        null_test_p_values=null_pvals_phase,
    )

    # 3. Inter-station MWPC (requires temporal overlap)
    # IMPORTANT: compute per-satellite to avoid mixing independent orbit geometries.
    by_sat: Dict[str, Dict] = {}
    pooled_pairs: List[Dict[str, object]] = []
    null_pvals: List[float] = []
    for sat in sorted(df['satellite'].unique().tolist()):
        sat_df = df[df['satellite'] == sat].copy()
        sat_res = analyze_interstation_mwpc(sat_df, station_coords, n_permutations=int(args.n_permutations))
        by_sat[str(sat)] = sat_res

        for p in sat_res.get('pairs', []):
            pp = dict(p)
            pp['satellite'] = str(sat)
            pooled_pairs.append(pp)

        nt = sat_res.get('null_tests', {}).get('phase_alignment_mean', {})
        pv = nt.get('p_value_one_sided_le')
        if pv is not None:
            null_pvals.append(float(pv))

    results['interstation_mwpc_by_satellite'] = by_sat
    results['interstation_mwpc'] = _summarize_interstation_pairs(
        pooled_pairs,
        null_test_p_values=null_pvals,
    )
    
    # 4. Range coherence
    results['range_coherence'] = analyze_range_coherence(df, bootstrap_n=int(args.bootstrap_n))
    
    # 5. Temporal coherence
    results['temporal_coherence'] = analyze_temporal_coherence(df, bootstrap_n=int(args.bootstrap_n))
    
    # 6. Daily-aggregation inter-station correlations (NEW - highest impact for sparse data)
    results['daily_aggregation_correlations'] = analyze_daily_aggregation_correlations(
        df,
        station_coords,
        n_permutations=int(args.n_permutations),
        top_n_stations=TOP_STATIONS_COUNT,
    )
    
    # 7. Lagged cross-correlations (NEW - tests TEP lag structure prediction)
    results['lagged_cross_correlations'] = analyze_lagged_cross_correlations(
        df,
        station_coords,
        max_lag_days=7,
        top_n_stations=TOP_STATIONS_COUNT,
    )

    # 5b. Robustness summaries (compact; manuscript-facing)
    robustness: Dict[str, object] = {}

    # (i) Threshold sweep (compute only summary metrics to keep JSON small)
    sweep: List[Dict[str, object]] = []
    for thr in sweep_thrs:
        df_thr = df_all[df_all['residual_m'].abs() < float(thr)].copy()
        if df_thr.empty:
            continue
        rc = analyze_range_coherence(df_thr, bootstrap_n=int(args.bootstrap_n))
        tc = analyze_temporal_coherence(df_thr, bootstrap_n=int(args.bootstrap_n))
        sweep.append({
            'residual_threshold_m': float(thr),
            'n_observations': int(len(df_thr)),
            'n_stations': int(df_thr['station'].nunique()),
            'range_coherence_ratio_summary': rc.get('ratio_summary', {}),
            'spectral_concentration_summary': (tc.get('spectral_concentration', {}) or {}).get('summary', {}),
        })
    robustness['threshold_sweep'] = sweep

    # (ii) Year-block stability (per-year point estimates; bootstrap across years)
    years = _extract_years(df)
    year_rows: List[Dict[str, object]] = []
    ratio_by_year: List[float] = []
    delta_by_year: List[float] = []
    spec_by_year: List[float] = []
    for y in years:
        try:
            t = pd.to_datetime(df['epoch_utc'], format='mixed', utc=True)
        except Exception:
            t = pd.to_datetime(df['epoch_utc'], utc=True, errors='coerce')
        sel = df.loc[t.dt.year == int(y)].copy()
        if sel.empty:
            continue
        rc_y = analyze_range_coherence(sel, bootstrap_n=0)
        ratio_y = (rc_y.get('ratio_summary', {}) or {}).get('ratio_low_over_high', {})
        ratio_val = ratio_y.get('value')
        delta_y = (rc_y.get('ratio_summary', {}) or {}).get('delta_low_minus_high', {})
        delta_val = delta_y.get('value')

        tc_y = analyze_temporal_coherence(sel, bootstrap_n=0)
        spec_sum = (tc_y.get('spectral_concentration', {}) or {}).get('summary', {})
        spec_val = spec_sum.get('mean')

        row = {
            'year': int(y),
            'n_observations': int(len(sel)),
            'n_stations': int(sel['station'].nunique()),
            'ratio_low_over_high': float(ratio_val) if ratio_val is not None else None,
            'delta_low_minus_high': float(delta_val) if delta_val is not None else None,
            'tep_over_broadband_mean': float(spec_val) if spec_val is not None else None,
        }
        year_rows.append(row)
        if ratio_val is not None and np.isfinite(float(ratio_val)):
            ratio_by_year.append(float(ratio_val))
        if delta_val is not None and np.isfinite(float(delta_val)):
            delta_by_year.append(float(delta_val))
        if spec_val is not None and np.isfinite(float(spec_val)):
            spec_by_year.append(float(spec_val))

    robustness['year_blocks'] = {
        'years': year_rows,
        'ratio_low_over_high': _bootstrap_mean(np.array(ratio_by_year, dtype=float), int(args.bootstrap_n), seed=11),
        'delta_low_minus_high': _bootstrap_mean(np.array(delta_by_year, dtype=float), int(args.bootstrap_n), seed=13),
        'tep_over_broadband_mean': _bootstrap_mean(np.array(spec_by_year, dtype=float), int(args.bootstrap_n), seed=12),
    }

    results['robustness'] = robustness

    # 6. Comparison with GNSS
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

        logged = 0
        for pair in results['interstation_mwpc']['pairs']:
            if logged >= MAX_LOGGED_PAIRS:
                break
            logger.info(f"  {pair['name_1']}-{pair['name_2']}: "
                       f"baseline={pair['baseline_km']:.0f}km, "
                       f"coh={pair['coherence']:.3f}, "
                       f"phase_align={pair['phase_alignment']:.3f}")
            logged += 1

        if len(results['interstation_mwpc']['pairs']) > MAX_LOGGED_PAIRS:
            logger.info(f"  ... (omitted {len(results['interstation_mwpc']['pairs']) - MAX_LOGGED_PAIRS} pairs)")
    
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
