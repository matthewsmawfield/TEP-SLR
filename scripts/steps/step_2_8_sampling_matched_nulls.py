#!/usr/bin/env python3
"""
TEP-SLR Step 2.8: Sampling-Matched Coloured-Noise Nulls & Amplitude Budget

Responds to the methodological gap in the Step 2.3 spectral diagnostic: the
published TEP-band concentration (mean PSD in 10-500 uHz / mean PSD above
1 mHz) is evaluated on a stitched series — the 5-minute resampled record
after linear interpolation (limit=2) and gap removal. The stitched vector is
treated by Welch's method as uniformly sampled even though the observing
duty cycle is well under one percent, so the estimator mixes the data's
spectrum with the sampling kernel.

This step supplies the missing controls:

1. SAMPLING-MATCHED STITCH NULLS. Surrogate processes (white, AR(1) with the
   station's measured lag-1 autocorrelation, canonical flicker 1/f, matched
   power-law, and random-walk) are generated as continuous processes on the
   station's full 5-minute grid, sampled at the actual observation bins,
   passed through the identical interpolate/dropna/detrend/Welch chain, and
   their TEP-band/broadband ratios are compared with the observed ratio.
   This prices the resampling channel exactly and tests whether the observed
   concentration exceeds what each noise model yields through the same
   sampling.

2. REAL-EPOCH LOMB-SCARGLE ESTIMATOR. The same band/broadband ratio is
   recomputed on the genuine observation epochs (5-minute bin means, no
   interpolation, no stitching) using Lomb-Scargle power, which is valid
   for irregular sampling. The identical surrogate families evaluated at
   the same real epochs provide matched nulls, so the spectral content of
   the residuals is assessed free of the stitching kernel.

3. STRUCTURE FUNCTION. The pairwise structure function SF(tau) =
   0.5 <(y_j - y_i)^2> over lag bins spanning the band-equivalent
   timescales (20 min - 28 h) measures the coherent wander at TEP-band
   timescales directly on real epochs, and a white-noise reference
   isolates the red excess.

4. AMPLITUDE MAP (delta-A -> residual). Under the timer-rate channel the
   two-way flight time is conformally invariant while the station event
   timer runs at the local matter rate A(t), so a conformal excursion
   delta-A maps to a range offset delta-R = R * delta-A. The measured
   residual wander is converted into the required in-band conformal
   excursion and compared with the Earth's surface conformal depth
   u_earth = GM_earth/(c^2 R_earth) ~ 6.95e-10, giving a falsifiable
   amplitude budget.

Output: results/outputs/step_2_8_sampling_matched_nulls.json
"""

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch, detrend, lombscargle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.logger import TEPLogger, set_step_logger

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger("step_2_8", log_file_path=LOGS_DIR / "step_2_8_sampling_matched_nulls.log")
set_step_logger(logger)

# Shared band definitions (identical to step_2_3_mwpc_analysis.py)
F1_HZ = 10e-6
F2_HZ = 500e-6
FS_HZ = 1 / 300.0
BROADBAND_MIN_HZ = 1e-3
RESIDUAL_THRESHOLD_M = 0.5

# Lomb-Scargle frequency grid: covers the band plus the broadband floor up
# to the 5-minute-bin Nyquist (1.667 mHz).
LS_FREQS = np.geomspace(5e-6, 1.667e-3, 240)
LS_ANG = 2 * np.pi * LS_FREQS
LS_BAND = (LS_FREQS >= F1_HZ) & (LS_FREQS <= F2_HZ)
LS_BB = LS_FREQS >= BROADBAND_MIN_HZ

# Structure-function lag grid spanning the band-equivalent timescales
SF_LAGS_H = [0.33, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0]

# Physical constants for the amplitude map
C_LIGHT = 299792458.0
GM_EARTH = 3.986004418e14
R_EARTH = 6.371e6
U_EARTH = GM_EARTH / (C_LIGHT ** 2 * R_EARTH)   # ~6.95e-10 surface conformal depth

POWER_LAW_GRID = [0.0, 0.5, 1.0, 1.5]


def build_station_grid(sta_df: pd.DataFrame):
    """Resample a station's debiased residuals to the 5-minute grid.

    Returns the observation-bin index, the bin-mean values, the kept
    (post-interpolation) index set, and the stitched series — the exact
    construction used by step_2_3's spectral diagnostic.
    """
    s = sta_df.sort_values('epoch').set_index('epoch')
    deb = s['residual_m'] - s['residual_m'].mean()
    res = deb.resample('5min').mean()
    obs_idx = np.flatnonzero(res.notna().values)
    if obs_idx.size < 10:
        return None
    interp = res.interpolate(method='linear', limit=2)
    kept_idx = np.flatnonzero(interp.notna().values)
    stitched = interp.dropna().values
    bin_vals = res.dropna().values
    rng_m = float(sta_df['model_range_m'].mean()) if 'model_range_m' in sta_df else float('nan')
    return {
        'obs_idx': obs_idx,
        'n_grid': int(len(res)),
        'kept_idx': kept_idx,
        'stitched': stitched,
        't_sec': obs_idx.astype(float) * 300.0,
        'y_real': bin_vals - bin_vals.mean(),
        'range_m': rng_m,
    }


def stitched_ratio(y: np.ndarray) -> float:
    """Observed diagnostic: detrend + Welch + TEP-band/broadband ratio."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yd = detrend(y, type='linear')
        f, p = welch(yd, fs=FS_HZ, nperseg=min(256, len(yd) // 2))
    bm = (f >= F1_HZ) & (f <= F2_HZ)
    bb = f >= BROADBAND_MIN_HZ
    if not np.any(bm) or not np.any(bb) or p[bb].mean() <= 0:
        return float('nan')
    return float(p[bm].mean() / p[bb].mean())


def ls_ratio(t: np.ndarray, y: np.ndarray) -> float:
    """Real-epoch Lomb-Scargle band/broadband ratio."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = lombscargle(t, y, LS_ANG, normalize=True, precenter=True)
    bb_mean = p[LS_BB].mean()
    if bb_mean <= 0:
        return float('nan')
    return float(p[LS_BAND].mean() / bb_mean)


def structure_function(t: np.ndarray, y: np.ndarray):
    """SF(tau) = 0.5 <(y_j - y_i)^2> for pairs inside +/-20% lag windows."""
    out = {}
    n = len(t)
    for lh in SF_LAGS_H:
        lo, hi = lh * 3600.0 * 0.8, lh * 3600.0 * 1.2
        diffs = []
        for i in range(n):
            j0 = np.searchsorted(t, t[i] + lo)
            j1 = np.searchsorted(t, t[i] + hi, side='right')
            if j1 > j0:
                diffs.append((y[j0:j1] - y[i]) ** 2)
        if diffs:
            d = np.concatenate(diffs)
            out[f'{lh:g}h'] = {'sf': float(0.5 * d.mean()), 'n_pairs': int(d.size)}
        else:
            out[f'{lh:g}h'] = {'sf': None, 'n_pairs': 0}
    return out


def _pl_series(n: int, alpha: float, rng: np.random.Generator, amp_cache: dict):
    """Timmer-Koenig Gaussian series with PSD ~ f^{-alpha} on an n-grid."""
    if alpha not in amp_cache:
        f = np.fft.rfftfreq(n)
        a = np.zeros(f.size)
        a[1:] = f[1:] ** (-alpha / 2.0)
        amp_cache[alpha] = a
    a = amp_cache[alpha]
    spec = a * (rng.standard_normal(a.size) + 1j * rng.standard_normal(a.size))
    return np.fft.irfft(spec, n=n)


def _ar1_series(n: int, phi: float, rng: np.random.Generator, amp_cache: dict):
    """Gaussian series with the AR(1) spectral shape on an n-grid."""
    key = f'ar1_{phi:.4f}'
    if key not in amp_cache:
        f = np.fft.rfftfreq(n)
        a = np.zeros(f.size)
        a[1:] = 1.0 / np.sqrt(1 - 2 * phi * np.cos(2 * np.pi * f[1:]) + phi ** 2)
        amp_cache[key] = a
    a = amp_cache[key]
    spec = a * (rng.standard_normal(a.size) + 1j * rng.standard_normal(a.size))
    return np.fft.irfft(spec, n=n)


def _rw_series(n: int, rng: np.random.Generator):
    return np.cumsum(rng.standard_normal(n))


def analyze_station(task: dict) -> dict:
    sta = task['station']
    obs_idx = task['obs_idx']
    kept_idx = task['kept_idx']
    stitched = task['stitched']
    t = task['t_sec']
    y_real = task['y_real']
    n_grid = task['n_grid']
    phi = task['phi']
    n_surr = task['n_surr']
    seed = task['seed']

    rng = np.random.default_rng(seed)
    amp_cache: dict = {}

    res = {'station': int(sta), 'n_bins_obs': int(obs_idx.size),
           'n_grid': int(n_grid), 'stitched_n': int(len(stitched)),
           'duty_fraction': float(obs_idx.size / max(n_grid, 1))}

    # --- observed diagnostics ---
    obs_stitch = stitched_ratio(stitched)
    obs_ls = ls_ratio(t, y_real)
    res['obs_ratio_stitched'] = obs_stitch
    res['obs_ratio_ls'] = obs_ls

    # stitched-spectrum slope (log-log OLS on the Welch bins)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yd = detrend(stitched, type='linear')
        f_w, p_w = welch(yd, fs=FS_HZ, nperseg=min(256, len(yd) // 2))
    pos = f_w > 0
    if pos.sum() > 4:
        res['alpha_fit_stitched'] = float(np.polyfit(np.log10(f_w[pos]),
                                                   np.log10(p_w[pos]), 1)[0])
    else:
        res['alpha_fit_stitched'] = None

    # observed structure function on real epochs
    res['structure_function'] = structure_function(t, y_real)
    res['residual_rms_mm'] = float(y_real.std() * 1e3)
    res['range_m'] = task['range_m']

    # --- surrogate families ---
    def draw(alpha_key):
        if alpha_key == 'ar1':
            return _ar1_series(n_grid, phi, rng, amp_cache)
        if alpha_key == 'rw':
            return _rw_series(n_grid, rng)
        return _pl_series(n_grid, float(alpha_key), rng, amp_cache)

    alpha_matched = res['alpha_fit_stitched']
    fam_defs = {
        'white': '0.0', 'pl05': '0.5', 'flicker': '1.0',
        'pl15': '1.5', 'rw': 'rw', 'ar1': 'ar1',
    }
    if alpha_matched is not None:
        # alpha_fit_stitched is the log-log *slope* (negative); the PSD
        # exponent in S ~ f^{-alpha} is its negation
        fam_defs['pl_matched'] = str(round(-alpha_matched, 3))

    acc = {ch: {fam: [] for fam in fam_defs} for ch in ('stitch', 'ls')}
    for _ in range(n_surr):
        for fam, akey in fam_defs.items():
            x = draw(akey)
            # stitch channel: same interpolate/dropna/detrend/Welch pipeline
            xs = np.interp(kept_idx, obs_idx, x[obs_idx])
            r1 = stitched_ratio(xs)
            if np.isfinite(r1):
                acc['stitch'][fam].append(r1)
            # real-epoch channel: Lomb-Scargle at the actual observation bins
            xe = x[obs_idx]
            xe = xe - xe.mean()
            r2 = ls_ratio(t, xe)
            if np.isfinite(r2):
                acc['ls'][fam].append(r2)

    for ch in ('stitch', 'ls'):
        obs = obs_stitch if ch == 'stitch' else obs_ls
        for fam in fam_defs:
            r_arr = np.asarray(acc[ch][fam])
            if r_arr.size < 10 or not np.isfinite(obs):
                res[f'{ch}_null_{fam}'] = None
                continue
            res[f'{ch}_null_{fam}'] = {
                'null_mean': float(r_arr.mean()),
                'null_std': float(r_arr.std(ddof=1)),
                'null_ci95': [float(v) for v in np.percentile(r_arr, [2.5, 97.5])],
                'obs': float(obs),
                'p_ge_obs': float(np.mean(r_arr >= obs)),
                'obs_percentile': float(np.mean(r_arr <= obs)),
                'excess': float(obs / r_arr.mean()) if r_arr.mean() > 0 else None,
            }

    # white-noise SF reference: analytic SF = sigma_y^2 (lag-independent for
    # independent samples), plus a small numerical set for sampling spread
    sf_ref: dict = {}
    sig2 = float(y_real.std() ** 2)
    n_ref = 5
    ref_acc = {f'{lh:g}h': [] for lh in SF_LAGS_H}
    for _ in range(n_ref):
        xw = rng.standard_normal(len(t)) * y_real.std()
        sfw = structure_function(t, xw)
        for k, v in sfw.items():
            if v['sf'] is not None:
                ref_acc[k].append(v['sf'])
    for k, arr in ref_acc.items():
        sf_ref[k] = {'analytic': sig2,
                     'mean': float(np.mean(arr)) if arr else None,
                     'std': float(np.std(arr)) if arr else None}
    res['structure_function_white_ref'] = sf_ref

    return res


def _aggregate(per_station: list, key: str, n_surr: int) -> dict:
    """Aggregate one null family across stations (same summary as step_2_3)."""
    rows = []
    for r in per_station:
        v = r.get(key)
        if v is None:
            continue
        rows.append({'station': r['station'], 'obs': v['obs'],
                     'null_mean': v['null_mean'], 'p': v['p_ge_obs'],
                     'pct': v['obs_percentile'], 'exc': v['excess']})
    if not rows:
        return {'n_stations': 0}
    p = np.array([r['p'] for r in rows])
    exc = np.array([r['exc'] for r in rows if r['exc'] is not None])
    z = np.array([(r['obs'] - r['null_mean']) for r in rows])
    n = len(rows)
    n05 = int(np.sum(p < 0.05)); n01 = int(np.sum(p < 0.01))
    p_safe = np.where(p > 0, p, 1.0 / (n_surr + 1))
    chi2 = float(-2 * np.sum(np.log(p_safe)))
    return {
        'n_stations': n,
        'obs_ratio_mean': float(np.mean([r['obs'] for r in rows])),
        'null_ratio_mean': float(np.mean([r['null_mean'] for r in rows])),
        'n_exceeding_null_p05': n05,
        'n_exceeding_null_p01': n01,
        'fraction_exceeding_p05': float(n05 / n),
        'mean_obs_minus_null': float(z.mean()),
        'mean_excess_ratio': float(exc.mean()) if exc.size else None,
        'median_obs_percentile_in_null': float(np.median([r['pct'] for r in rows])),
        'fisher_combined_p': float(stats.chi2.sf(chi2, df=2 * n)),
        'binomial_p_excess': float(
            stats.binomtest(n05, n, 0.05, alternative='greater').pvalue),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="TEP-SLR Step 2.8: Sampling-matched coloured-noise nulls")
    ap.add_argument("--input", default=str(OUTPUTS_DIR / "step_2_1_slr_residuals.csv"))
    ap.add_argument("--red-noise", default=str(OUTPUTS_DIR / "red_noise_null_test.json"),
                    help="Step 2.3 red-noise JSON for per-station AR(1) phi reuse")
    ap.add_argument("--n-surrogates", type=int, default=120)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--output", default=str(OUTPUTS_DIR / "step_2_8_sampling_matched_nulls.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.input, usecols=['epoch_utc', 'station', 'residual_m', 'model_range_m'])
    df = df[df['residual_m'].abs() < RESIDUAL_THRESHOLD_M].copy()
    df['epoch'] = pd.to_datetime(df['epoch_utc'], format='mixed')
    logger.info(f"Loaded {len(df)} filtered residuals, {df['station'].nunique()} stations")

    phi_map: dict = {}
    rp = Path(args.red_noise)
    if rp.exists():
        try:
            rn = json.loads(rp.read_text())
            for e in rn.get('per_station', []):
                phi_map[int(e['station'])] = float(e['phi'])
        except Exception as exc:
            logger.warning(f"Could not read AR(1) phi map: {exc}")

    tasks = []
    for sta, g in df.groupby('station'):
        g = g.copy()
        built = build_station_grid(g)
        if built is None:
            continue
        phi = phi_map.get(int(sta))
        if phi is None:
            r = g.sort_values('epoch')['residual_m'].values
            phi = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 10 else 0.5
        tasks.append({'station': int(sta), 'phi': float(phi),
                      'n_surr': int(args.n_surrogates), 'seed': int(args.seed) + int(sta),
                      'range_m': built['range_m'],
                      'obs_idx': built['obs_idx'], 'kept_idx': built['kept_idx'],
                      'stitched': built['stitched'], 't_sec': built['t_sec'],
                      'y_real': built['y_real'], 'n_grid': built['n_grid']})

    logger.info(f"Prepared {len(tasks)} station tasks (n_surr={args.n_surrogates}, workers={args.workers})")

    per_station = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(analyze_station, tasks):
                per_station.append(r)
                logger.info(f"station {r['station']} done")
    else:
        for tk in tasks:
            r = analyze_station(tk)
            per_station.append(r)
            logger.info(f"station {r['station']} done")

    per_station.sort(key=lambda r: r['station'])

    # ---- amplitude map: delta-A -> residual ----
    # Timer-rate channel: measured range R_meas = A * R_true, so
    # delta-R = R * delta-A. Required in-band excursion <= residual wander / range.
    amp_rows = []
    for r in per_station:
        rm = r.get('range_m')
        sig_m = r['residual_rms_mm'] / 1e3 if r.get('residual_rms_mm') else None
        if rm and sig_m:
            amp_rows.append({'station': r['station'],
                             'range_km': rm / 1e3,
                             'deltaA_full': sig_m / rm,
                             'deltaA_over_u_earth': (sig_m / rm) / U_EARTH})
    r_net = float(np.mean([a['range_km'] for a in amp_rows]) * 1e3)
    delta_a_full = float(np.mean([a['deltaA_full'] for a in amp_rows]))
    amplitude = {
        'channel': 'R_meas = A * R_true (timer reads conformally invariant flight time)',
        'map': 'delta_R = R * delta_A',
        'mean_range_km': r_net / 1e3,
        'u_earth_surface': U_EARTH,
        'deltaA_full_residual': delta_a_full,
        'deltaA_full_over_u_earth': delta_a_full / U_EARTH,
        'per_station': amp_rows,
        'geometric_channel_anchors': {
            'solar_well_diurnal_sampling': 2 * 9.87e-9 * (R_EARTH / 1.496e11),
            'lunar_well_diurnal_sampling': 1.419e-13 * 2 * (R_EARTH / 3.844e8),
        },
    }

    result = {
        'step': 'step_2_8_sampling_matched_nulls',
        'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
        'description': (
            'Sampling-matched coloured-noise null tests for the TEP-band '
            'spectral concentration. Surrogates are generated as continuous '
            'processes on each station full 5-min grid and passed through the '
            'identical resample/interpolate/stitch/detrend/Welch chain '
            '(stitch channel), and evaluated at the real observation epochs '
            'under a Lomb-Scargle estimator (real-epoch channel). Families: '
            'white, AR(1) with station phi, flicker 1/f, power-law grid, '
            'matched power-law, random-walk. Amplitude map converts residual '
            'wander to the required conformal excursion delta-A via the '
            'timer-rate channel delta-R = R * delta-A.'),
        'parameters': {
            'residual_threshold_m': RESIDUAL_THRESHOLD_M,
            'tep_band_hz': [F1_HZ, F2_HZ],
            'broadband_min_hz': BROADBAND_MIN_HZ,
            'fs_hz': FS_HZ,
            'n_surrogates_per_family': int(args.n_surrogates),
            'ls_n_freqs': int(len(LS_FREQS)),
            'sf_lags_h': SF_LAGS_H,
            'u_earth_surface': U_EARTH,
            'base_seed': int(args.seed),
            'station_seed_rule': 'seed = base_seed + station_id',
        },
        'per_station': per_station,
        'aggregate_stitch': {fam: _aggregate(per_station, f'stitch_null_{fam}', int(args.n_surrogates))
                             for fam in ['white', 'pl05', 'flicker', 'pl15', 'rw', 'ar1', 'pl_matched']},
        'aggregate_ls': {fam: _aggregate(per_station, f'ls_null_{fam}', int(args.n_surrogates))
                         for fam in ['white', 'pl05', 'flicker', 'pl15', 'rw', 'ar1', 'pl_matched']},
        'amplitude_map': amplitude,
    }

    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2))
    logger.info(f"Wrote {out}")

    # console digest
    agg = result['aggregate_stitch']['white']
    if agg.get('n_stations'):
        print(f"[step_2_8] stitched-white null: obs {agg['obs_ratio_mean']:.2f} "
              f"vs null {agg['null_ratio_mean']:.2f}")
    agl = result['aggregate_ls']['white']
    if agl.get('n_stations'):
        print(f"[step_2_8] real-epoch LS: obs {agl['obs_ratio_mean']:.2f} "
              f"vs white null {agl['null_ratio_mean']:.2f}")
    print(f"[step_2_8] deltaA full-residual bound: {delta_a_full:.2e} "
          f"({delta_a_full / U_EARTH:.1%} of u_earth)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
