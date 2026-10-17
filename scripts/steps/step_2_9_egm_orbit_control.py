#!/usr/bin/env python3
"""
NOTE on fidelity: Step 2.1 merges multi-week SP3 files into a
position-only ephemeris (velocities dropped) and therefore interpolates
with the Catmull-Rom tangent construction in interp_sat_pos. This control
replicates that convention exactly (vel_m_per_s=None on the merged
ephemeris) — verified 0.000 mm reproduction of model_range_m on sampled
rows. Using velocity-Hermite instead introduces metre-level discrepancies
that would dominate the |residual|<0.5 m filter.

Step 2.9 — EGM/orbit-product common-mode control for the cross-satellite
turnover-bin anticorrelation of Step 2.7.

Registered channel (corpus-review-2026-09-29, issue 8-1): the cross-satellite
pairing removes a *shared* orbit-arc error by construction, but LAGEOS-1 and
LAGEOS-2 sit at the same altitude and their orbit solutions propagate the same
Earth-gravity-model coefficients (GM, J2, tides). A geopotential deficiency
therefore injects correlated structure into both solutions and hence into both
stations' residuals — a common mode that survives the cross-satellite split.
The turnover-bin decomposition confirms the exposure is concentrated exactly
where this channel is strongest: 31 of the 32 pairs in the published
5000-7500 km bin are lageos1 x lageos2 records.

Control: recompute the model ranges for lageos1/lageos2 observations using an
*independent analysis centre's* SP3 orbit product (default ESA/ESOC; different
gravity field, arc length, and dynamical parameterization from the ILRS
combination used in Step 2.1), rebuild residuals, and re-run the identical
cross-satellite pairing estimator. If the turnover-bin anticorrelation is an
EGM/orbit-model imprint it must move or disappear under an independent orbit;
if it persists at the same amplitude the orbit-product channel is bounded.

Outputs
  results/outputs/step_2_9_egm_orbit_control.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import step_2_1_slr_residuals as s21  # noqa: E402
import step_2_7_orbit_commonmode_control as s27  # noqa: E402

RESID_CSV = REPO_ROOT / "results/outputs/step_2_1_slr_residuals.csv"
COORDS_JSON = REPO_ROOT / "results/outputs/station_coordinates.json"
SLRF2020 = REPO_ROOT / "data/slr/products/resource/SLRF2020_POS+VEL_2026.02.05.snx.gz"
ALT_CACHE = REPO_ROOT / "data/slr/products/orbits/sp3_alt"
OUT_JSON = REPO_ROOT / "results/outputs/step_2_9_egm_orbit_control.json"

C_M_S = 299792458.0
SATS = ("lageos1", "lageos2")
# ILRS SP3 sat IDs: L51 = LAGEOS-1, L52 = LAGEOS-2
SP3_ID = {"lageos1": "L51", "lageos2": "L52"}


def get_session():
    import netrc
    import requests
    auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
    sess = requests.Session()
    if auth:
        sess.auth = (auth[0], auth[2])
    return sess


def week_saturdays(epochs_unix: np.ndarray) -> List[str]:
    weeks = set()
    for t in epochs_unix:
        d = datetime.fromtimestamp(float(t), tz=timezone.utc)
        saturday = d - timedelta(days=(d.weekday() - 5) % 7)
        weeks.add(saturday.strftime("%y%m%d"))
    return sorted(weeks)


def fetch_alt_sp3(session, center: str, sat: str, week: str) -> Optional[Path]:
    """Download best-version <center>.orb.<sat>.<week>.vNN.sp3.gz."""
    dest_dir = ALT_CACHE / center / sat / week
    existing = list(dest_dir.glob("*.sp3*")) if dest_dir.exists() else []
    if existing:
        return existing[0]
    import re
    url = f"https://cddis.nasa.gov/archive/slr/products/orbits/{sat}/{week}/"
    try:
        r = session.get(url, timeout=60)
        if r.status_code != 200:
            return None
        names = sorted(set(re.findall(
            rf'{center}\.orb\.{sat}\.{week}\.v\d+\.sp3(?:\.gz)?', r.text)))
        if not names:
            return None
        name = names[-1]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / name
        with session.get(url + name, timeout=300, stream=True) as resp:
            if resp.status_code != 200:
                return None
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(1 << 20):
                    f.write(chunk)
        return dest
    except Exception:
        return None


def build_alt_ephemeris(session, center: str, sat: str,
                        weeks: List[str]) -> tuple:
    """Parse and merge all weekly SP3 files for one centre/satellite."""
    all_e, all_p, all_v = [], [], []
    missing = []
    for w in weeks:
        path = fetch_alt_sp3(session, center, sat, w)
        if path is None:
            missing.append(w)
            continue
        try:
            eph = s21.parse_sp3_positions(path, sat_id=SP3_ID[sat])
        except Exception:
            missing.append(w)
            continue
        all_e.append(eph.epochs)
        all_p.append(eph.pos_m)
        if eph.vel_m_per_s is not None:
            all_v.append(eph.vel_m_per_s)
    if not all_e:
        return None, missing
    e = np.concatenate(all_e)
    p = np.concatenate(all_p)
    order = np.argsort(e)
    e, p = e[order], p[order]
    keep = np.concatenate([[True], np.diff(e) > 1e-6])
    e, p = e[keep], p[keep]
    # Match Step 2.1's merge convention exactly: merged ephemerides are
    # position-only (velocities dropped) so interp_sat_pos uses the same
    # Catmull-Rom tangent construction. Verified to reproduce
    # model_range_m to 0.000 mm on sampled rows under the same orbit.
    return s21.SatelliteEphemeris(epochs=e, pos_m=p, source="sp3",
                                  vel_m_per_s=None), missing


def recompute_model_ranges(df: pd.DataFrame, eph_map: Dict[str, object],
                           stations: Dict[str, object]) -> pd.DataFrame:
    """Recompute model_range_m under the alternative ephemeris for each row."""
    out = df.copy()
    model_alt = np.full(len(out), np.nan)
    sta_cache: Dict[str, np.ndarray] = {}
    for idx, r in enumerate(out.itertuples()):
        eph = eph_map.get(r.satellite)
        st = stations.get(str(r.station))
        if eph is None or st is None:
            continue
        epoch = pd.to_datetime(r.epoch_utc, format="mixed").to_pydatetime()
        key = (r.station, epoch)
        sta_ecef = sta_cache.get(key)
        if sta_ecef is None:
            nom = s21.propagate_station(st, epoch)
            sta_ecef = nom + s21.compute_solid_earth_tide_displacement(nom, epoch) \
                + s21.compute_ocean_loading_displacement(nom, epoch)
            sta_cache[key] = sta_ecef
        ltime = float(r.obs_range_m) / C_M_S
        try:
            for _ in range(3):
                sat_ecef = s21.interp_sat_pos(
                    eph, epoch.timestamp() + ltime)
                mrange = float(np.linalg.norm(sat_ecef - sta_ecef))
                ltime = mrange / C_M_S
            model_alt[idx] = mrange
        except Exception:
            continue
    out["model_range_alt_m"] = model_alt
    out["residual_alt_m"] = out["obs_range_corrected_m"] - out["model_range_alt_m"]
    out["orbit_shift_mm"] = (out["model_range_alt_m"]
                             - out["model_range_m"]) * 1000.0
    return out


def run(center: str, n_perm: int) -> Dict:
    session = get_session()
    df = pd.read_csv(RESID_CSV, usecols=[
        "epoch_utc", "station", "satellite", "obs_range_m",
        "obs_range_corrected_m", "model_range_m", "residual_m"])
    df = df[df["satellite"].isin(SATS)].copy()
    # Controlled comparison: same included-epoch set as the published
    # estimator (ilrsb-filtered), then re-filtered under the alt orbit below.
    n_all = len(df)
    df = df[df["residual_m"].abs() < s27.RESIDUAL_THRESHOLD_M].copy()
    print(f"lageos rows: {n_all} -> {len(df)} after reference |res|<0.5 m",
          flush=True)
    df["t_unix"] = pd.to_datetime(df["epoch_utc"], format="mixed") \
        .map(lambda t: t.timestamp())
    stations = s21.load_slrf2020_station_states(SLRF2020)

    eph_map, missing_map = {}, {}
    for sat in SATS:
        weeks = week_saturdays(
            df.loc[df["satellite"] == sat, "t_unix"].to_numpy())
        eph, missing = build_alt_ephemeris(session, center, sat, weeks)
        eph_map[sat] = eph
        missing_map[sat] = missing
        print(f"{sat}: {len(weeks)} weeks needed, {len(missing)} missing, "
              f"ephemeris epochs={0 if eph is None else len(eph.epochs)}",
              flush=True)

    df = recompute_model_ranges(df, eph_map, stations)
    df_alt = df[df["residual_alt_m"].abs() < s27.RESIDUAL_THRESHOLD_M].copy()
    df_alt["residual_m"] = df_alt["residual_alt_m"]

    coords = {int(k): v for k, v in
              json.load(open(COORDS_JSON)).items()}
    cell = s27.build_cells(df_alt)
    recs = s27.collect_pair_records(cell, "cross", coords)
    pc = s27.pair_correlations(recs)
    binned = s27.bin_stats(pc)
    tsel = pc[(pc["km"] >= s27.TURNOVER_BIN_LO_KM)
              & (pc["km"] < s27.TURNOVER_BIN_HI_KM)]
    comp: Dict[str, int] = defaultdict(int)
    for chans in tsel["chans"]:
        for c in chans:
            comp[c] += 1

    null = s27.permutation_null(recs, pc, n_perm=n_perm, seed=0)

    shift = df["orbit_shift_mm"].dropna()
    return {
        "step": "step_2_9_egm_orbit_control",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Orbit-product common-mode control: recompute LAGEOS-1/2 model "
            "ranges under an independent analysis centre's SP3 solution and "
            "re-run the Step 2.7 cross-satellite pairing. Bounds the shared "
            "Earth-gravity-model/orbit-model imprint on the 5000-7500 km "
            "turnover-bin anticorrelation."),
        "alt_center": center,
        "reference_product": "ilrsb (ILRS combination, Step 2.1)",
        "design_note": (
            "Restricted to epochs that passed the Step 2.1 ilrsb |residual| "
            "< 0.5 m filter (same-observation controlled comparison), then "
            "re-filtered under the alternative orbit."),
        "n_rows_lageos_filtered_reference": int(len(df)),
        "n_missing_weeks": {k: len(v) for k, v in missing_map.items()},
        "orbit_shift_mm": {
            "median": float(shift.median()),
            "std": float(shift.std()),
            "p95_abs": float(shift.abs().quantile(0.95)),
            "n": int(len(shift)),
        },
        "n_pairs": int(len(pc)),
        "distance_binned": binned,
        "turnover_bin_channel_composition": dict(comp),
        "turnover_bin": {
            "n_pairs": int(len(tsel)),
            "mean_r": float(tsel["r"].mean()) if len(tsel) else None,
            "median_r": float(tsel["r"].median()) if len(tsel) else None,
            "p_one_sided_le": null.get("p_values_one_sided_le_by_bin", {})
                                  .get("5000-7500km"),
            "p_fwer_min_bin": null.get("p_fwer_min_bin"),
        },
        "comparison_reference_step_2_7": {
            "turnover_bin_mean_r": -0.2275062670570105,
            "p_one_sided_le": 0.0004997501249375312,
            "note": "ilrsb product, identical estimator",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", default="esa",
                    help="Alternative ILRS analysis centre (asi, bkg, dgfi, "
                         "esa, gfz, jcet, nsgf)")
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    res = run(args.center, args.n_perm)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["turnover_bin"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
