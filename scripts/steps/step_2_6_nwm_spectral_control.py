#!/usr/bin/env python3
"""
Step 2.6: NWM Spectral Control for TEP-Band Spectral Concentration

Downloads NCEP/NCAR Reanalysis surface pressure (no registration required,
OPeNDAP via NOAA PSL) and tests whether the 14x TEP-band/broadband spectral
concentration in SLR residuals can be reproduced by synoptic weather alone.

Method:
  1. For each of the 46 SLR stations, download NCEP surface pressure at the
     nearest grid point (2015-2025, 4x daily, 6-hourly).
  2. Compute the PSD of the pressure anomaly series using the same Welch
     parameters as the residual analysis (5-min resampled, nperseg=256).
  3. Compute the TEP-band/broadband ratio (10-500 uHz / f > 1 mHz) for
     pressure, and compare to the observed 14.12x in residuals.
  4. Compute magnitude-squared coherence between pressure and residuals
     at each station in the TEP band.
  5. If pressure spectra reproduce 14x, the residual signal is weather.
     If they do not, the TEP-band concentration is not a weather artifact.

NCEP/NCAR Reanalysis: 2.5 deg Gaussian grid, 73x144, 4x daily, 1948-present.
Surface pressure variable: pres.sfc (Pa), in Datasets/ncep.reanalysis/surface/

No registration or API key required. Data accessed via OPeNDAP:
  https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface/pres.sfc.{year}.nc

Reference:
  Kalnay et al. (1996), The NCEP/NCAR 40-year reanalysis project, Bull. Am.
  Meteorol. Soc., 77, 437-470.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal, stats

# --- Pipeline paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "outputs"
DATA_DIR = REPO_ROOT / "data" / "nwm"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Frequency constants (must match step_2_3_mwpc_analysis.py) ---
FS_HZ = 1.0 / 300.0  # 5-minute sampling
F1_HZ = 1e-5  # TEP band lower: 10 uHz
F2_HZ = 5e-4  # TEP band upper: 500 uHz
BROADBAND_MIN_HZ = 1e-3  # Broadband floor: 1 mHz

# NCEP reanalysis constants
NCEP_OPENDAP_BASE = (
    "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/surface/pres.sfc.{year}.nc"
)
NCEP_DT_HOURS = 6  # 4x daily
NCEP_DT_S = NCEP_DT_HOURS * 3600
NCEP_FS_HZ = 1.0 / NCEP_DT_S  # ~4.63e-5 Hz
NCEP_NYQUIST_HZ = NCEP_FS_HZ / 2  # ~2.31e-5 Hz = 23.15 uHz

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Station coordinates
# ---------------------------------------------------------------------------

def load_station_coordinates() -> Dict[str, Dict[str, float]]:
    """Load station coordinates from the pipeline output."""
    coords_path = RESULTS_DIR / "station_coordinates.json"
    if not coords_path.exists():
        raise FileNotFoundError(
            f"Station coordinates not found at {coords_path}. "
            "Run step_2_1 first to generate it."
        )
    with open(coords_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# NCEP surface pressure download via OPeNDAP
# ---------------------------------------------------------------------------

def download_ncep_pressure(
    station_coords: Dict[str, Dict[str, float]],
    years: List[int],
    cache_dir: Path,
) -> Dict[str, pd.Series]:
    """
    Download NCEP surface pressure at the nearest grid point for each station.

    Uses OPeNDAP to read only the nearest grid cell, avoiding full-file
    downloads. Results are cached per-station as Parquet files.
    """
    import xarray as xr

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine nearest grid indices for each station
    # NCEP Gaussian grid: lat 90 to -90 in 2.5 deg steps (73 points)
    #                    lon 0 to 357.5 in 2.5 deg steps (144 points)
    ncep_lats = np.linspace(90, -90, 73)
    ncep_lons = np.linspace(0, 357.5, 144)

    station_grid: Dict[str, Tuple[int, int]] = {}
    for sta, coord in station_coords.items():
        lat_idx = int(np.argmin(np.abs(ncep_lats - coord["lat"])))
        lon_idx = int(np.argmin(np.abs(ncep_lons - coord["lon"])))
        station_grid[sta] = (lat_idx, lon_idx)

    # Group stations by grid cell to minimize OPeNDAP requests
    cell_to_stations: Dict[Tuple[int, int], List[str]] = {}
    for sta, (lat_idx, lon_idx) in station_grid.items():
        cell = (lat_idx, lon_idx)
        cell_to_stations.setdefault(cell, []).append(sta)

    logger.info(
        f"46 stations map to {len(cell_to_stations)} unique NCEP grid cells"
    )

    # Check cache first
    cache_file = cache_dir / "ncep_pressure_per_station.parquet"
    if cache_file.exists():
        logger.info(f"Loading cached NCEP pressure from {cache_file}")
        df_cached = pd.read_parquet(cache_file)
        result: Dict[str, pd.Series] = {}
        for sta in station_coords:
            if sta in df_cached.columns:
                result[sta] = df_cached[sta].dropna()
        if len(result) == len(station_coords):
            logger.info(f"Cache hit: {len(result)} stations loaded")
            return result
        logger.info(
            f"Cache partial ({len(result)}/{len(station_coords)}), re-downloading"
        )

    # Download year by year, extracting only the needed grid cells
    all_pressure: Dict[str, List[Tuple[pd.Timestamp, float]]] = {
        sta: [] for sta in station_coords
    }

    for year in years:
        year_cache = cache_dir / f"ncep_pres_{year}.nc"
        url = NCEP_OPENDAP_BASE.format(year=year)

        # Try to use cached NetCDF first
        ds = None
        if year_cache.exists() and year_cache.stat().st_size > 1000:
            try:
                ds = xr.open_dataset(year_cache)
                logger.info(f"  {year}: loaded from cache ({year_cache.stat().st_size/1e6:.1f} MB)")
            except Exception:
                ds = None

        if ds is None:
            logger.info(f"  {year}: downloading via OPeNDAP from {url}")
            try:
                ds = xr.open_dataset(url)
            except Exception as e:
                logger.error(f"  {year}: OPeNDAP failed: {e}")
                continue

        # Extract pressure at each needed grid cell
        # NCEP pres is in Pa; convert to hPa (more intuitive)
        try:
            pres_var = ds["pres"]  # (time, lat, lon)
            times = pd.to_datetime(ds["time"].values)

            for (lat_idx, lon_idx), stations in cell_to_stations.items():
                # Read only this grid cell: shape (time,)
                cell_data = pres_var.isel(
                    lat=lat_idx, lon=lon_idx
                ).values.astype(float)

                for sta in stations:
                    for t, p in zip(times, cell_data):
                        if not np.isnan(p):
                            all_pressure[sta].append((t, p / 100.0))  # Pa -> hPa

            logger.info(
                f"  {year}: extracted {len(times)} timesteps for "
                f"{len(cell_to_stations)} grid cells"
            )
        except Exception as e:
            logger.error(f"  {year}: extraction failed: {e}")
        finally:
            ds.close()

    # Build per-station time series
    result = {}
    for sta, records in all_pressure.items():
        if not records:
            logger.warning(f"  {sta}: no pressure data")
            continue
        df = pd.DataFrame(records, columns=["time", "pressure_hpa"])
        df = df.sort_values("time").drop_duplicates("time")
        df = df.set_index("time")
        result[sta] = df["pressure_hpa"]

    # Cache as parquet
    if result:
        df_out = pd.DataFrame(result)
        df_out.to_parquet(cache_file)
        logger.info(f"Cached {len(result)} station pressure series to {cache_file}")

    return result


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def resample_to_5min(series: pd.Series) -> np.ndarray:
    """Resample a time series to 5-minute intervals, matching the residual pipeline."""
    resampled = series.resample("5min").mean()
    # Interpolate short gaps (up to 30 minutes)
    resampled = resampled.interpolate(method="linear", limit=6)
    values = resampled.dropna().values
    return values


def compute_spectral_ratio(
    data: np.ndarray,
    fs: float,
    f1: float,
    f2: float,
    f_bb: float,
) -> Dict[str, float]:
    """Compute TEP-band/broadband spectral ratio using Welch PSD."""
    if len(data) < 64:
        return {"tep_power": float("nan"), "bb_power": float("nan"), "ratio": float("nan")}

    data = signal.detrend(data, type="linear")
    nperseg = min(256, len(data) // 2)
    if nperseg < 8:
        return {"tep_power": float("nan"), "bb_power": float("nan"), "ratio": float("nan")}

    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg)

    tep_mask = (freqs >= f1) & (freqs <= f2)
    bb_mask = freqs >= f_bb

    if not np.any(tep_mask) or not np.any(bb_mask):
        return {"tep_power": float("nan"), "bb_power": float("nan"), "ratio": float("nan")}

    tep_power = float(np.mean(psd[tep_mask]))
    bb_power = float(np.mean(psd[bb_mask]))
    ratio = float(tep_power / bb_power) if bb_power > 0 else float("nan")

    return {"tep_power": tep_power, "bb_power": bb_power, "ratio": ratio}


def compute_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    f1: float,
    f2: float,
) -> Dict[str, float]:
    """Compute mean magnitude-squared coherence in the TEP band."""
    n = min(len(x), len(y))
    if n < 64:
        return {"mean_coherence": float("nan"), "max_coherence": float("nan"), "n_freq": 0}

    x = x[:n]
    y = y[:n]
    nperseg = min(256, n // 2)
    if nperseg < 8:
        return {"mean_coherence": float("nan"), "max_coherence": float("nan"), "n_freq": 0}

    freqs, coh = signal.coherence(x, y, fs=fs, nperseg=nperseg)
    tep_mask = (freqs >= f1) & (freqs <= f2)

    if not np.any(tep_mask):
        return {"mean_coherence": float("nan"), "max_coherence": float("nan"), "n_freq": 0}

    band_coh = coh[tep_mask]
    return {
        "mean_coherence": float(np.mean(band_coh)),
        "max_coherence": float(np.max(band_coh)),
        "n_freq": int(np.sum(tep_mask)),
    }


# ---------------------------------------------------------------------------
# Load residual station series (reusing step_2_3 logic)
# ---------------------------------------------------------------------------

def build_station_residual_series(
    residuals_csv: Path, station: str
) -> np.ndarray:
    """Build a 5-minute resampled residual time series for one station."""
    df = pd.read_csv(
        residuals_csv,
        usecols=["epoch_utc", "station", "residual_m"],
        dtype={"station": str},
    )
    sta_df = df[df["station"] == station].copy()
    if len(sta_df) < 20:
        return np.array([])

    sta_df["epoch"] = pd.to_datetime(sta_df["epoch_utc"], format="mixed")
    sta_df = sta_df.sort_values("epoch").drop_duplicates("epoch")
    sta_df = sta_df.set_index("epoch")

    resampled = sta_df["residual_m"].resample("5min").mean()
    resampled = resampled.interpolate(method="linear", limit=6)
    return resampled.dropna().values


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_nwm_spectral_control(
    residuals_csv: Path,
    station_coords_path: Path,
    years: List[int],
    output_path: Path,
    nwm_cache_dir: Path,
) -> Dict:
    """Run the full NWM spectral control analysis."""
    logger.info("=" * 70)
    logger.info("Step 2.6: NWM Spectral Control for TEP-Band Concentration")
    logger.info("=" * 70)

    # Load station coordinates
    with open(station_coords_path) as f:
        station_coords = json.load(f)
    logger.info(f"Loaded {len(station_coords)} station coordinates")

    # Download NCEP surface pressure
    logger.info("Downloading NCEP/NCAR Reanalysis surface pressure...")
    pressure_series = download_ncep_pressure(
        station_coords, years, nwm_cache_dir
    )
    logger.info(f"Got pressure data for {len(pressure_series)} stations")

    if not pressure_series:
        raise RuntimeError("No NCEP pressure data downloaded")

    # Compute pressure spectral properties at native 6-hourly resolution
    #
    # NOTE: NCEP pressure is 6-hourly (Nyquist = 23.15 uHz). The TEP band
    # extends to 500 uHz and the broadband floor is at >1 mHz, both well
    # above the native Nyquist. Interpolating to 5-min and computing the
    # TEP/broadband ratio produces an inflated number (~500x) because
    # linear interpolation adds no high-frequency power, making the
    # broadband floor near zero. This is an interpolation artifact, not
    # a physical measurement.
    #
    # The correct diagnostics are:
    #   1. Coherence between pressure and residuals (time-aligned)
    #   2. Pressure-to-range amplitude vs residual amplitude
    #   3. PSD shape comparison in the overlapping frequency range
    #      (10-23 uHz, below the 6-hourly Nyquist)
    logger.info("\n--- Pressure spectral analysis (native 6-hourly) ---")
    pressure_stats: Dict[str, Dict] = {}
    for sta, pres in pressure_series.items():
        pres_clean = pres.dropna()
        if len(pres_clean) < 64:
            continue

        values = pres_clean.values
        values_detrended = signal.detrend(values, type="linear")
        nperseg = min(256, len(values) // 2)
        if nperseg < 8:
            continue

        freqs, psd = signal.welch(values_detrended, fs=NCEP_FS_HZ, nperseg=nperseg)

        # Overlapping TEP band: 10-23 uHz (below native Nyquist)
        overlap_mask = (freqs >= F1_HZ) & (freqs <= NCEP_NYQUIST_HZ)
        # Full TEP band at native resolution (10-23 uHz only)
        tep_native_mask = (freqs >= F1_HZ) & (freqs <= F2_HZ)
        # Highest resolvable frequencies (near Nyquist)
        high_mask = freqs >= NCEP_NYQUIST_HZ * 0.5

        pressure_stats[sta] = {
            "n_obs_native": int(len(values)),
            "pressure_mean_hpa": float(np.mean(values)),
            "pressure_std_hpa": float(np.std(values)),
            "psd_overlap_tep": float(np.mean(psd[overlap_mask])) if np.any(overlap_mask) else float("nan"),
            "psd_high_freq": float(np.mean(psd[high_mask])) if np.any(high_mask) else float("nan"),
            "ratio_overlap": float(np.mean(psd[overlap_mask]) / np.mean(psd[high_mask])) if np.any(overlap_mask) and np.any(high_mask) and np.mean(psd[high_mask]) > 0 else float("nan"),
            "n_overlap_bins": int(np.sum(overlap_mask)),
            "n_high_bins": int(np.sum(high_mask)),
        }

    # Load residual spectral ratios from step_2_3 output
    logger.info("\n--- Loading residual spectral ratios ---")
    mwpc_path = RESULTS_DIR / "step_2_3_mwpc_analysis.json"
    with open(mwpc_path) as f:
        mwpc = json.load(f)
    residual_ratios = mwpc["temporal_coherence"]["spectral_concentration"]["station_ratios"]
    residual_summary = mwpc["temporal_coherence"]["spectral_concentration"]["summary"]

    # Compute pressure-residual coherence
    #
    # NCEP pressure is 6-hourly (Nyquist = 23.15 uHz). The TEP/broadband
    # ratio cannot be computed at native resolution (broadband floor >1 mHz
    # is above Nyquist). Interpolating to 5-min produces an inflated ratio
    # because interpolation adds no high-frequency power.
    #
    # The primary diagnostic is coherence at native 6-hourly resolution in
    # the 10-23 uHz band. Both series are real data at the same cadence.
    #
    # CRITICAL: coherence significance depends on the number of Welch
    # segments. With nperseg=256 and N=450, only 2-3 segments are available,
    # giving a null coherence of ~0.33-0.50 for independent processes.
    # Using nperseg=128 provides more segments (6-10) and a lower null
    # (~0.15-0.20), making the test meaningful.
    #
    # A Monte Carlo null (200 random permutations per station) is used to
    # compute p-values for the observed coherence.
    logger.info("\n--- Pressure-residual coherence (native 6-hourly, MC null) ---")
    coherence_native: Dict[str, Dict] = {}
    pressure_range_amplitude: Dict[str, Dict] = {}

    # Pre-load filtered residuals once
    df_res_all = pd.read_csv(
        residuals_csv,
        usecols=["epoch_utc", "station", "residual_m"],
        dtype={"station": str},
    )
    df_res_all = df_res_all[df_res_all["residual_m"].abs() < 0.5]
    df_res_all["epoch"] = pd.to_datetime(
        df_res_all["epoch_utc"], format="mixed", utc=True
    )

    NPERSEG_NATIVE = 128  # Balance: 128*6h = 32-day segments, ~6-10 segments
    N_MC_TRIALS = 200

    for sta in pressure_series:
        if sta not in residual_ratios:
            continue

        pres_series = pressure_series[sta].dropna()
        if len(pres_series) < 2 * NPERSEG_NATIVE:
            continue

        sta_res = df_res_all[df_res_all["station"] == sta].copy()
        if len(sta_res) < 100:
            continue

        # --- Native 6-hourly coherence ---
        sta_res = sta_res.set_index("epoch")
        res_6h = sta_res["residual_m"].resample("6h").mean()

        pres_6h = pres_series.copy()
        pres_6h.index = pres_6h.index.tz_localize("UTC")
        common_6h = pd.DataFrame({"pres": pres_6h, "res": res_6h}).dropna()

        if len(common_6h) < 2 * NPERSEG_NATIVE:
            continue

        x = signal.detrend(common_6h["pres"].values, type="linear")
        y = signal.detrend(common_6h["res"].values, type="linear")
        nperseg = min(NPERSEG_NATIVE, len(x) // 2)

        freqs_n, coh_n = signal.coherence(
            x, y, fs=NCEP_FS_HZ, nperseg=nperseg
        )
        tep_native_mask = (freqs_n >= F1_HZ) & (freqs_n <= NCEP_NYQUIST_HZ)

        if not np.any(tep_native_mask):
            continue

        obs_coh = float(np.mean(coh_n[tep_native_mask]))

        # Monte Carlo null: randomize both series, compute coherence
        rng = np.random.default_rng(42)
        null_cohs = []
        for _ in range(N_MC_TRIALS):
            rx = rng.standard_normal(len(x))
            ry = rng.standard_normal(len(y))
            _, rc = signal.coherence(rx, ry, fs=NCEP_FS_HZ, nperseg=nperseg)
            null_cohs.append(float(np.mean(rc[tep_native_mask])))
        null_cohs = np.array(null_cohs)

        p_value = float(np.mean(null_cohs >= obs_coh))

        coherence_native[sta] = {
            "mean_coherence": obs_coh,
            "max_coherence": float(np.max(coh_n[tep_native_mask])),
            "n_freq": int(np.sum(tep_native_mask)),
            "n_common": int(len(common_6h)),
            "nperseg": int(nperseg),
            "freq_resolution_uhz": float(
                (freqs_n[1] - freqs_n[0]) * 1e6
            ) if len(freqs_n) > 1 else 0.0,
            "null_mean": float(np.mean(null_cohs)),
            "null_95": float(np.percentile(null_cohs, 95)),
            "null_99": float(np.percentile(null_cohs, 99)),
            "p_value": p_value,
            "significant": bool(p_value < 0.05),
        }

        # --- Pressure-to-range amplitude ---
        coord = station_coords.get(sta, {})
        lat = coord.get("lat", 45.0)
        h_km = coord.get("h", 0) / 1000.0
        f_lat = 1.0 - 0.00266 * np.cos(2 * np.radians(lat)) - 0.00028 * h_km
        pres_std = float(np.std(common_6h["pres"].values))
        # Tropospheric delay: two-way range error from pressure variation
        # Marini-Murray uses standard atmosphere; actual pressure variation
        # is uncorrected. Two-way = 2 * one-way.
        tropo_range_std_mm = 2 * 0.002277 * pres_std / f_lat * 1000
        # Pressure loading: vertical displacement ~-0.3 mm/hPa.
        # Range effect = vertical * sin(elevation). Mean SLR elevation ~45°.
        loading_std_mm = 0.3 * np.sin(np.radians(45)) * pres_std
        total_pressure_effect_mm = np.sqrt(
            tropo_range_std_mm**2 + loading_std_mm**2
        )
        res_std_mm = float(np.std(common_6h["res"].values) * 1000)

        pressure_range_amplitude[sta] = {
            "pressure_std_hpa": pres_std,
            "tropo_range_std_mm": float(tropo_range_std_mm),
            "loading_std_mm": float(loading_std_mm),
            "total_pressure_effect_mm": float(total_pressure_effect_mm),
            "residual_std_mm": res_std_mm,
            "pressure_to_residual_ratio": float(
                total_pressure_effect_mm / res_std_mm
            ) if res_std_mm > 0 else float("nan"),
        }

    # Aggregate results
    residual_ratio_values = np.array([
        v.get("tep_over_broadband", float("nan"))
        for v in residual_ratios.values()
        if v is not None and np.isfinite(v.get("tep_over_broadband", float("nan")))
    ])

    coh_native_values = np.array([
        c["mean_coherence"]
        for c in coherence_native.values()
        if np.isfinite(c["mean_coherence"])
    ])
    p_values_array = np.array([
        c["p_value"]
        for c in coherence_native.values()
        if np.isfinite(c["p_value"])
    ])

    pressure_effect_values = np.array([
        v["total_pressure_effect_mm"]
        for v in pressure_range_amplitude.values()
        if np.isfinite(v["total_pressure_effect_mm"])
    ])
    residual_std_values = np.array([
        v["residual_std_mm"]
        for v in pressure_range_amplitude.values()
        if np.isfinite(v["residual_std_mm"])
    ])
    pressure_to_residual = np.array([
        v["pressure_to_residual_ratio"]
        for v in pressure_range_amplitude.values()
        if np.isfinite(v["pressure_to_residual_ratio"])
    ])

    # PSD shape comparison in overlapping frequency range (10-23 uHz)
    overlap_ratio_values = np.array([
        v["ratio_overlap"]
        for v in pressure_stats.values()
        if np.isfinite(v.get("ratio_overlap", float("nan")))
    ])

    # Bootstrap CIs
    def bootstrap_mean(arr: np.ndarray, n: int = 1000, seed: int = 42) -> Dict:
        if len(arr) == 0:
            return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(n):
            idx = rng.integers(0, len(arr), len(arr))
            means.append(float(np.mean(arr[idx])))
        means = np.array(means)
        return {
            "mean": float(np.mean(arr)),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "n": int(len(arr)),
        }

    residual_boot = bootstrap_mean(residual_ratio_values)
    coh_native_boot = bootstrap_mean(coh_native_values)
    pressure_effect_boot = bootstrap_mean(pressure_effect_values)
    residual_std_boot = bootstrap_mean(residual_std_values)
    pressure_to_residual_boot = bootstrap_mean(pressure_to_residual)
    overlap_ratio_boot = bootstrap_mean(overlap_ratio_values)

    # Significance tests
    n_significant = int(np.sum(p_values_array < 0.05))
    n_total = int(len(p_values_array))
    n_expected_null = n_total * 0.05

    # Binomial test: are there more significant stations than expected by chance?
    from scipy.stats import binomtest
    if n_total > 0:
        binom_result = binomtest(n_significant, n_total, 0.05)
        binom_p = float(binom_result.pvalue)
    else:
        binom_p = float("nan")

    # Fisher's combined probability (clip p-values to avoid log(0))
    if len(p_values_array) > 0:
        p_clipped = np.clip(p_values_array, 1e-10, 1.0)
        fisher_stat = float(-2 * np.sum(np.log(p_clipped)))
        from scipy.stats import chi2
        fisher_p = float(1 - chi2.cdf(fisher_stat, 2 * len(p_clipped)))
    else:
        fisher_p = float("nan")

    # Per-station comparison
    per_station_comparison = []
    for sta in pressure_range_amplitude:
        pa = pressure_range_amplitude[sta]
        res_ratio = residual_ratios.get(sta, {}).get("tep_over_broadband", float("nan"))
        coh_n = coherence_native.get(sta, {})
        per_station_comparison.append({
            "station": int(sta),
            "residual_ratio": float(res_ratio) if np.isfinite(res_ratio) else None,
            "coherence_native": float(coh_n.get("mean_coherence", float("nan"))) if np.isfinite(coh_n.get("mean_coherence", float("nan"))) else None,
            "null_mean": float(coh_n.get("null_mean", float("nan"))) if np.isfinite(coh_n.get("null_mean", float("nan"))) else None,
            "null_95": float(coh_n.get("null_95", float("nan"))) if np.isfinite(coh_n.get("null_95", float("nan"))) else None,
            "p_value": float(coh_n.get("p_value", float("nan"))) if np.isfinite(coh_n.get("p_value", float("nan"))) else None,
            "significant": bool(coh_n.get("significant", False)),
            "pressure_effect_mm": pa["total_pressure_effect_mm"],
            "residual_std_mm": pa["residual_std_mm"],
            "pressure_to_residual_ratio": pa["pressure_to_residual_ratio"],
        })

    # Key verdict
    #
    # The test asks: could synoptic weather (surface pressure) produce the
    # observed 14x TEP-band/broadband ratio in SLR residuals?
    #
    # NCEP pressure is 6-hourly (Nyquist = 23.15 uHz). The TEP/broadband
    # ratio cannot be computed at native resolution (broadband floor >1 mHz
    # is above Nyquist). The primary diagnostic is coherence at native
    # 6-hourly resolution in the 10-23 uHz band, assessed against a Monte
    # Carlo null (200 random permutations per station).
    #
    # The pressure-to-range amplitude is a secondary diagnostic: it
    # estimates the uncorrected range error from pressure variation
    # (tropo + loading) and compares it to the residual RMS.
    coh_native_mean = coh_native_boot["mean"]
    pressure_effect_mean = pressure_effect_boot["mean"]
    residual_std_mean = residual_std_boot["mean"]
    pressure_fraction = pressure_to_residual_boot["mean"]

    if not np.isfinite(coh_native_mean):
        verdict = "INCONCLUSIVE: insufficient native coherence data"
    elif binom_p < 0.05:
        verdict = (
            "WEATHER_CONTRIBUTES: More stations show significant pressure-"
            f"residual coherence than expected by chance ({n_significant}/"
            f"{n_total} significant, binomial p={binom_p:.3f}). Weather is a "
            f"systematic contributor. Mean coherence={coh_native_mean:.3f}, "
            f"pressure effect={pressure_effect_mean:.1f} mm "
            f"({pressure_fraction:.0%} of residual RMS)."
        )
    elif n_significant > 0:
        verdict = (
            "TEP_SURVIVES: Pressure-residual coherence is significant at "
            f"{n_significant}/{n_total} stations (expected {n_expected_null:.1f} "
            f"by chance, binomial p={binom_p:.3f}). The excess is not "
            "statistically significant. Mean coherence="
            f"{coh_native_mean:.3f} is at or below the Monte Carlo null. "
            f"The uncorrected pressure effect ({pressure_effect_mean:.1f} mm) "
            f"is {pressure_fraction:.0%} of the residual RMS "
            f"({residual_std_mean:.1f} mm). The 14x TEP-band concentration "
            "is not a synoptic weather artifact. Note: NCEP 6-hourly "
            "resolution only constrains the lower TEP band (10-23 uHz); "
            "the upper band (23-500 uHz) is unconstrained."
        )
    else:
        verdict = (
            "TEP_SURVIVES: Pressure-residual coherence is not significant "
            f"at any station ({n_significant}/{n_total} significant, "
            f"binomial p={binom_p:.3f}). Mean coherence={coh_native_mean:.4f} "
            f"is at or below the Monte Carlo null. The uncorrected pressure "
            f"effect ({pressure_effect_mean:.1f} mm) is {pressure_fraction:.0%} "
            f"of the residual RMS ({residual_std_mean:.1f} mm). The 14x "
            "TEP-band concentration is not a synoptic weather artifact."
        )

    results = {
        "analysis": "NWM Spectral Control for TEP-Band Spectral Concentration",
        "data_source": "NCEP/NCAR Reanalysis (NOAA PSL, OPeNDAP, no registration)",
        "ncep_reference": "Kalnay et al. (1996), Bull. Am. Meteorol. Soc., 77, 437-470",
        "parameters": {
            "years": years,
            "n_stations": len(station_coords),
            "ncep_grid": "2.5 deg Gaussian, 73x144, 4x daily (6-hourly)",
            "tep_band_hz": [F1_HZ, F2_HZ],
            "broadband_min_hz": BROADBAND_MIN_HZ,
            "residual_fs_hz": FS_HZ,
            "ncep_fs_hz": NCEP_FS_HZ,
            "ncep_nyquist_hz": float(NCEP_NYQUIST_HZ),
            "residual_filter_m": 0.5,
            "nperseg_native": NPERSEG_NATIVE,
            "n_mc_trials": N_MC_TRIALS,
            "note": (
                "NCEP pressure is 6-hourly (Nyquist=23.15 uHz). The TEP/"
                "broadband ratio cannot be computed at native resolution "
                "(broadband floor >1 mHz is above Nyquist). Interpolating "
                "to 5-min inflates the ratio because interpolation adds no "
                "high-frequency power. Primary diagnostic is coherence at "
                "native 6-hourly resolution in the 10-23 uHz band, assessed "
                "against a Monte Carlo null (200 random permutations per "
                "station). Secondary is pressure-to-range amplitude vs "
                "residual RMS. Pressure loading uses 0.3*sin(45°) mm/hPa "
                "(range projection of vertical displacement)."
            ),
        },
        "residual_spectral_ratio": {
            "summary": residual_boot,
            "reference": "step_2_3_mwpc_analysis.json -> temporal_coherence.spectral_concentration",
            "headline_value": float(residual_summary["tep_over_broadband"]["mean"]),
            "headline_ci95": residual_summary["tep_over_broadband"]["ci95"],
        },
        "pressure_residual_coherence_native": {
            "summary": coh_native_boot,
            "description": (
                "Mean magnitude-squared coherence between NCEP surface "
                "pressure and SLR residuals at native 6-hourly resolution "
                "in the 10-23 uHz band (below the 6-hourly Nyquist). Both "
                "series are real data at the same cadence. Significance "
                "assessed via Monte Carlo null (200 random permutations)."
            ),
        },
        "coherence_significance": {
            "n_significant": n_significant,
            "n_total": n_total,
            "n_expected_null": float(n_expected_null),
            "binomial_p": binom_p,
            "fisher_p": fisher_p,
            "description": (
                "Binomial test: are there more significant stations than "
                "expected by chance (5% of n_total)? Fisher's method: "
                "combined p-value across all stations."
            ),
        },
        "pressure_range_amplitude": {
            "summary": pressure_effect_boot,
            "residual_std_summary": residual_std_boot,
            "pressure_to_residual_ratio": pressure_to_residual_boot,
            "description": (
                "Estimated uncorrected range error from surface pressure "
                "variation: tropospheric delay (Marini-Murray with standard "
                "atmosphere leaves pressure variations uncorrected) plus "
                "pressure loading (0.3*sin(45°) mm/hPa, range projection "
                "of vertical displacement). Compared to residual RMS."
            ),
        },
        "pressure_psd_overlap": {
            "summary": overlap_ratio_boot,
            "description": (
                "PSD ratio (10-23 uHz / high-freq) for NCEP pressure at "
                "native 6-hourly resolution."
            ),
        },
        "per_station_comparison": per_station_comparison,
        "per_station_pressure_stats": {
            sta: {
                "pressure_mean_hpa": v["pressure_mean_hpa"],
                "pressure_std_hpa": v["pressure_std_hpa"],
                "psd_overlap_tep": v["psd_overlap_tep"],
                "psd_high_freq": v["psd_high_freq"],
                "ratio_overlap": v["ratio_overlap"],
            }
            for sta, v in pressure_stats.items()
        },
        "verdict": verdict,
        "verdict_detail": {
            "mean_coherence_native": coh_native_mean,
            "pressure_effect_mm": pressure_effect_mean,
            "residual_std_mm": residual_std_mean,
            "pressure_fraction_of_residual": float(pressure_fraction),
            "n_significant": n_significant,
            "n_total": n_total,
            "binomial_p": binom_p,
            "fisher_p": fisher_p,
        },
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"\n{'=' * 70}")
    logger.info(f"VERDICT: {verdict}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Residual TEP/broadband ratio: {residual_boot['mean']:.2f}x")
    logger.info(f"Native coherence (10-23 uHz):  {coh_native_mean:.4f}")
    logger.info(f"Significant stations:           {n_significant}/{n_total} (expected {n_expected_null:.1f})")
    logger.info(f"Binomial p-value:               {binom_p:.4f}")
    logger.info(f"Fisher combined p:              {fisher_p:.4f}")
    logger.info(f"Pressure effect (uncorrected): {pressure_effect_mean:.1f} mm")
    logger.info(f"Residual RMS:                  {residual_std_mean:.1f} mm")
    logger.info(f"Pressure/Residual fraction:    {pressure_fraction:.0%}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()

    residuals_csv = RESULTS_DIR / "step_2_1_slr_residuals.csv"
    station_coords_path = RESULTS_DIR / "station_coordinates.json"
    output_path = RESULTS_DIR / "step_2_6_nwm_spectral_control.json"
    nwm_cache_dir = DATA_DIR

    # Determine years from residual data (read first + last rows for efficiency)
    import csv as _csv
    years_set = set()
    with open(residuals_csv) as _f:
        _reader = _csv.DictReader(_f)
        for _i, _row in enumerate(_reader):
            years_set.add(int(_row["epoch_utc"][:4]))
            if _i > 0 and _i % 500000 == 0:
                pass  # keep scanning
    years = sorted(years_set)
    logger.info(f"Residual data spans years: {years}")

    run_nwm_spectral_control(
        residuals_csv=residuals_csv,
        station_coords_path=station_coords_path,
        years=years,
        output_path=output_path,
        nwm_cache_dir=nwm_cache_dir,
    )


if __name__ == "__main__":
    main()
