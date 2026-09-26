#!/usr/bin/env python3
"""
Step 2.7: Orbit-Error and Common-Mode Confound Controls for Inter-Station Correlations

The pass-bin inter-station statistic of Step 2.3 pairs stations that range the
*same* satellite inside a 15-minute window. Any residual orbit-model error on
that satellite's arc therefore enters both stations' residuals as a common mode
and trivially correlates them — the dominant mundane confound for the
distance-structured claim. This step implements the three controls required to
separate that channel from a station-located spatial process:

Control A — cross-satellite contemporaneous pairing.
  Station pairs are formed when the two stations range *different* satellites in
  the same time bin. Different satellites carry independent orbit solutions, so
  the shared-orbit-error channel is absent by construction. A genuine spatial
  field at the stations (the TEP claim) is satellite-independent and survives;
  orbit error does not. The same-satellite pairing is recomputed in parallel on
  identical machinery for a matched contrast.

Control B — network common-mode subtraction (daily aggregation).
  The daily-aggregation test is recomputed after removing the per-day network
  mean (shared secular bias drift, e.g. the network-wide offset of order a few
  hundred mm present in the raw residuals). Removing a per-day mean over N=20
  stations injects a mechanical pairwise bias of ~-1/(N-1); the expected bias is
  reported alongside so the residual structure can be read against it.

Control C — per-satellite daily aggregation.
  The daily-aggregation distance-binned correlations are computed separately per
  satellite, testing dependence on a single orbit solution. A feature robust to
  the spatial-field interpretation must reproduce under independent solutions.

Statistic: per-station-pair Pearson correlation of contemporaneous debiased
bin-mean residuals, distance-binned (same bins as Step 2.3). Significance uses
the within-pair circular-shift permutation null (destroys synchrony, preserves
marginals and pair-level autocorrelation), reported per-bin and as the
family-wise minimum-bin statistic. The turnover bin (first bin entirely beyond
the simulated lambda_T turnover under either corpus anchor — GPS-PPP 4201 km
gives a 3,947 km crossing, MGEX 1862 km gives 3,421 km, i.e. 5000-7500 km) is
pre-specified by the Section 4.2 simulation, so its per-bin p is the primary
confound-controlled test; a station-clustered bootstrap CI on the same bin
prices the shared-station non-independence of the pair list.

Inputs:  results/outputs/step_2_1_slr_residuals.csv
         results/outputs/station_coordinates.json
Output:  results/outputs/step_2_7_orbit_commonmode_control.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "outputs"
LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "step_2_7_orbit_commonmode_control.log"),
    ],
)
logger = logging.getLogger("step_2_7")

# Constants — must match step_2_3_mwpc_analysis.py
DISTANCE_BINS_KM = [0, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000]
PASS_TIME_BIN = "15min"
MIN_OBS_PER_STATION_BIN = 1
MIN_PASSES_PER_PAIR = 3
MIN_PAIRS_PER_DISTANCE_BIN = 3
RESIDUAL_THRESHOLD_M = 0.5
TEP_COHERENCE_LENGTH_KM = 4201.0  # GPS-PPP terrestrial coherence scale; MGEX anchor (1862 km) selects the same turnover bin (step_3_0 anchor scan: crossings 3,421 vs 3,947 km)
TURNOVER_BIN_LO_KM = 5000.0       # first bin entirely beyond lambda_T
TURNOVER_BIN_HI_KM = 7500.0

DAILY_AGG_MIN_OBS_PER_DAY = 3
DAILY_AGG_MIN_DAYS_PER_STATION = 10
DAILY_AGG_MIN_SHARED_DAYS = 5
TOP_STATIONS_COUNT = 20


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 3:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    den = math.sqrt(float((xm * xm).sum() * (ym * ym).sum()))
    if den == 0.0:
        return float("nan")
    return float((xm * ym).sum() / den)


def build_cells(df: pd.DataFrame, subtract_station_bias: bool = True) -> pd.DataFrame:
    """Per-(time_bin, satellite, station) mean residual, debiased by the
    station's global mean (matching Step 2.3's convention)."""
    df = df.copy()
    df["epoch"] = pd.to_datetime(df["epoch_utc"], format="mixed")
    df["time_bin"] = df["epoch"].dt.floor(PASS_TIME_BIN)
    if subtract_station_bias:
        bias = df.groupby("station")["residual_m"].mean()
        df["res"] = df["residual_m"] - df["station"].map(bias)
    else:
        df["res"] = df["residual_m"]
    cell = (
        df.groupby(["time_bin", "satellite", "station"])["res"]
        .agg(["mean", "count"])
        .reset_index()
    )
    cell = cell[cell["count"] >= MIN_OBS_PER_STATION_BIN]
    return cell


def collect_pair_records(cell: pd.DataFrame, mode: str,
                         coords: Dict[int, Dict]) -> pd.DataFrame:
    """Build contemporaneous station-pair records.

    mode='same' : pairs of stations observing the same satellite in the bin.
    mode='cross': pairs of stations observing different satellites in the bin.
    """
    recs: List[tuple] = []
    for tb, g in cell.groupby("time_bin"):
        sats = g["satellite"].unique()
        if mode == "same":
            for s in sats:
                gs = g[g["satellite"] == s]
                stas = gs["station"].tolist()
                if len(stas) < 2:
                    continue
                for a, b in combinations(stas, 2):
                    if a not in coords or b not in coords:
                        continue
                    recs.append((
                        a, b,
                        float(gs.loc[gs["station"] == a, "mean"].iloc[0]),
                        float(gs.loc[gs["station"] == b, "mean"].iloc[0]),
                        haversine_km(coords[a]["lat"], coords[a]["lon"],
                                     coords[b]["lat"], coords[b]["lon"]),
                        str(s),
                    ))
        else:
            if len(sats) < 2:
                continue
            for sa, sb in combinations(sorted(sats), 2):
                ga = g[g["satellite"] == sa]
                gb = g[g["satellite"] == sb]
                for a in ga["station"]:
                    for b in gb["station"]:
                        if a == b or a not in coords or b not in coords:
                            continue
                        recs.append((
                            a, b,
                            float(ga.loc[ga["station"] == a, "mean"].iloc[0]),
                            float(gb.loc[gb["station"] == b, "mean"].iloc[0]),
                            haversine_km(coords[a]["lat"], coords[a]["lon"],
                                         coords[b]["lat"], coords[b]["lon"]),
                            f"{sa}|{sb}",
                        ))
    return pd.DataFrame(recs, columns=["s1", "s2", "m1", "m2", "km", "chan"])


def pair_correlations(recs: pd.DataFrame) -> pd.DataFrame:
    """Per-station-pair Pearson correlation of the contemporaneous bin means."""
    rows = []
    for (a, b), g in recs.groupby(["s1", "s2"]):
        if len(g) < MIN_PASSES_PER_PAIR:
            continue
        c = pearson(g["m1"].to_numpy(), g["m2"].to_numpy())
        if np.isnan(c):
            continue
        chans = sorted(g["chan"].unique().tolist())
        rows.append({
            "s1": int(a), "s2": int(b),
            "km": float(g["km"].iloc[0]),
            "r": float(c),
            "n": int(len(g)),
            "chans": chans,
        })
    return pd.DataFrame(rows, columns=["s1", "s2", "km", "r", "n", "chans"])


def bin_stats(pc: pd.DataFrame) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for i in range(len(DISTANCE_BINS_KM) - 1):
        lo, hi = DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1]
        sel = pc[(pc["km"] >= lo) & (pc["km"] < hi)]
        if len(sel) >= MIN_PAIRS_PER_DISTANCE_BIN:
            out[f"{lo}-{hi}km"] = {
                "n_pairs": int(len(sel)),
                "mean_correlation": float(sel["r"].mean()),
                "median_correlation": float(sel["r"].median()),
                "std_correlation": float(sel["r"].std()) if len(sel) > 1 else 0.0,
            }
    return out


def permutation_null(recs: pd.DataFrame, pc: pd.DataFrame,
                     n_perm: int, seed: int = 0) -> Dict:
    """Within-pair circular-shift null: destroys bin synchrony while preserving
    each station series' marginals and autocorrelation."""
    series1, series2 = {}, {}
    for (a, b), g in recs.groupby(["s1", "s2"]):
        if len(g) >= MIN_PASSES_PER_PAIR:
            series1[(a, b)] = g["m1"].to_numpy()
            series2[(a, b)] = g["m2"].to_numpy()
    obs_means = bin_stats(pc)
    if not obs_means:
        return {}
    obs_min = float(min(v["mean_correlation"] for v in obs_means.values()))

    pairs = [
        (p["s1"], p["s2"], float(p["km"]))
        for p in pc.to_dict("records")
        if (p["s1"], p["s2"]) in series2
    ]
    rng = np.random.default_rng(seed)
    null_min: List[float] = []
    null_bin_vals: Dict[str, List[float]] = {k: [] for k in obs_means}
    bin_edges = {k: (DISTANCE_BINS_KM[i], DISTANCE_BINS_KM[i + 1])
                 for i, k in enumerate(obs_means)}
    for _ in range(int(n_perm)):
        per_bin: Dict[str, List[float]] = {k: [] for k in obs_means}
        for s1, s2, km in pairs:
            r1 = series1[(s1, s2)]
            r2 = series2[(s1, s2)]
            n = r2.size
            shift = int(rng.integers(1, n))
            c = pearson(r1, np.roll(r2, shift))
            if np.isnan(c):
                continue
            for k, (lo, hi) in bin_edges.items():
                if lo <= km < hi:
                    per_bin[k].append(c)
                    break
        ms = []
        for k, v in per_bin.items():
            if v:
                m = float(np.mean(v))
                null_bin_vals[k].append(m)
                ms.append(m)
            else:
                null_bin_vals[k].append(float("nan"))
        if ms:
            null_min.append(min(ms))
    if not null_min:
        return {}
    na = np.asarray(null_min)
    p_fwer = float((np.sum(na <= obs_min) + 1) / (na.size + 1))
    per_bin_p = {}
    for k, om in obs_means.items():
        arr = np.asarray([v for v in null_bin_vals[k] if not np.isnan(v)])
        per_bin_p[k] = (
            float((np.sum(arr <= om["mean_correlation"]) + 1) / (arr.size + 1))
            if arr.size else None
        )
    return {
        "method": "min_mean_over_bins_fwer_circular_shift",
        "n_permutations": int(na.size),
        "observed_bin_means": {k: v["mean_correlation"] for k, v in obs_means.items()},
        "observed_min_bin": float(obs_min),
        "p_fwer_min_bin": p_fwer,
        "p_values_one_sided_le_by_bin": per_bin_p,
    }


def station_cluster_bootstrap(pc: pd.DataFrame, lo: float, hi: float,
                              n_boot: int = 2000, seed: int = 0) -> Dict:
    """Resample stations (not pairs); keep pairs with both endpoints drawn.
    Prices the non-independence of pairs sharing a station."""
    sel = pc[(pc["km"] >= lo) & (pc["km"] < hi)]
    if len(sel) < MIN_PAIRS_PER_DISTANCE_BIN:
        return {}
    stations = sorted(set(sel["s1"]) | set(sel["s2"]))
    rng = np.random.default_rng(seed)
    boots: List[float] = []
    recs = sel[["s1", "s2", "r"]].to_dict("records")
    for _ in range(int(n_boot)):
        keep = set(rng.choice(stations, size=len(stations), replace=True).tolist())
        vals = [r_["r"] for r_ in recs if r_["s1"] in keep and r_["s2"] in keep]
        if len(vals) >= MIN_PAIRS_PER_DISTANCE_BIN:
            boots.append(float(np.mean(vals)))
    if not boots:
        return {}
    b = np.asarray(boots)
    return {
        "n_bootstrap_used": int(b.size),
        "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
        "fraction_below_zero": float((b < 0).mean()),
    }


def daily_series_by_station(df: pd.DataFrame, top_stations: List[int],
                            subtract_network_mean: bool = False) -> Dict[int, pd.Series]:
    sub = df[df["station"].isin(top_stations)].copy()
    bias = sub.groupby("station")["residual_m"].mean()
    sub["res"] = sub["residual_m"] - sub["station"].map(bias)
    sub["date"] = pd.to_datetime(sub["epoch_utc"], format="mixed").dt.date
    if subtract_network_mean:
        cm = sub.groupby("date")["res"].transform("mean")
        sub["res"] = sub["res"] - cm
    out: Dict[int, pd.Series] = {}
    for s in top_stations:
        d = sub[sub["station"] == s].groupby("date")["res"].agg(["mean", "count"])
        d = d[d["count"] >= DAILY_AGG_MIN_OBS_PER_DAY]
        if len(d) >= DAILY_AGG_MIN_DAYS_PER_STATION:
            out[int(s)] = d["mean"]
    return out


def daily_pair_correlations(daily: Dict[int, pd.Series],
                            coords: Dict[int, Dict]) -> pd.DataFrame:
    rows = []
    sts = list(daily)
    for i, a in enumerate(sts):
        for b in sts[i + 1:]:
            idx = daily[a].index.intersection(daily[b].index)
            if len(idx) < DAILY_AGG_MIN_SHARED_DAYS:
                continue
            c = pearson(daily[a].loc[idx].to_numpy(), daily[b].loc[idx].to_numpy())
            if np.isnan(c):
                continue
            rows.append({
                "s1": int(a), "s2": int(b),
                "km": haversine_km(coords[a]["lat"], coords[a]["lon"],
                                   coords[b]["lat"], coords[b]["lon"]),
                "r": float(c), "n": int(len(idx)),
            })
    return pd.DataFrame(rows, columns=["s1", "s2", "km", "r", "n"])


def run(df: pd.DataFrame, coords: Dict[int, Dict], n_perm: int = 2000) -> Dict:
    results: Dict[str, object] = {
        "step": "step_2_7_orbit_commonmode_control",
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Confound controls for the pass-bin inter-station correlation: "
            "(A) cross-satellite contemporaneous pairing removes the shared "
            "orbit-error channel by construction; (B) per-day network "
            "common-mode subtraction; (C) per-satellite daily aggregation "
            "tests orbit-solution dependence."
        ),
        "parameters": {
            "residual_threshold_m": RESIDUAL_THRESHOLD_M,
            "pass_time_bin": PASS_TIME_BIN,
            "min_passes_per_pair": MIN_PASSES_PER_PAIR,
            "distance_bins_km": DISTANCE_BINS_KM,
            "turnover_bin_km": [TURNOVER_BIN_LO_KM, TURNOVER_BIN_HI_KM],
            "turnover_bin_rationale": (
                "First distance bin entirely beyond the simulated lambda_T "
                "turnover under either corpus anchor (GNSS PPP 4201 km gives "
                "a 3,947 km zero crossing; MGEX 1862 km gives 3,421 km — "
                "step_3_0 anchor scan) — pre-specified by the "
                "Section 4.2 monopole-absorption turnover prediction, not "
                "selected on the data."
            ),
            "n_permutations": int(n_perm),
            "top_stations": TOP_STATIONS_COUNT,
        },
    }

    logger.info("Building (time_bin, satellite, station) cells...")
    cell = build_cells(df)
    results["n_cell_records"] = int(len(cell))

    # ---- Control A: cross-satellite pairing ----
    logger.info("Control A: collecting same-satellite and cross-satellite pairs...")
    rec_same = collect_pair_records(cell, "same", coords)
    rec_cross = collect_pair_records(cell, "cross", coords)
    pc_same = pair_correlations(rec_same)
    pc_cross = pair_correlations(rec_cross)
    logger.info(f"  same-sat pairs: {len(pc_same)} (from {len(rec_same)} bin records)")
    logger.info(f"  cross-sat pairs: {len(pc_cross)} (from {len(rec_cross)} bin records)")

    block: Dict[str, object] = {
        "n_bin_records_same": int(len(rec_same)),
        "n_bin_records_cross": int(len(rec_cross)),
        "n_station_pairs_same": int(len(pc_same)),
        "n_station_pairs_cross": int(len(pc_cross)),
        "same_satellite": {"distance_binned": bin_stats(pc_same)},
        "cross_satellite": {"distance_binned": bin_stats(pc_cross)},
    }

    # channel composition of the turnover bin (cross-sat)
    tsel = pc_cross[(pc_cross["km"] >= TURNOVER_BIN_LO_KM) & (pc_cross["km"] < TURNOVER_BIN_HI_KM)]
    comp: Dict[str, int] = {}
    for chans in tsel["chans"]:
        for c in chans:
            comp[c] = comp.get(c, 0) + 1
    block["cross_satellite"]["turnover_bin_channel_composition"] = comp  # type: ignore[index]

    # station participation in turnover bin
    if len(tsel):
        part = pd.concat([tsel["s1"], tsel["s2"]]).value_counts()
        block["cross_satellite"]["turnover_bin_station_participation"] = {  # type: ignore[index]
            int(k): int(v) for k, v in part.items()
        }

    for mode, recs_, pc_ in [("same_satellite", rec_same, pc_same),
                             ("cross_satellite", rec_cross, pc_cross)]:
        logger.info(f"  permutation null ({mode}, n={n_perm})...")
        block[mode]["null_tests"] = permutation_null(recs_, pc_, n_perm)  # type: ignore[index]
        boot = station_cluster_bootstrap(pc_, TURNOVER_BIN_LO_KM, TURNOVER_BIN_HI_KM)
        if boot:
            block[mode]["turnover_bin_station_bootstrap"] = boot  # type: ignore[index]
    results["control_A_cross_satellite_pairing"] = block

    # ---- Controls B & C: daily aggregation variants ----
    logger.info("Controls B/C: daily aggregation — pooled, CM-removed, per-satellite...")
    station_counts = df.groupby("station").size().sort_values(ascending=False)
    top = [int(s) for s in station_counts.head(TOP_STATIONS_COUNT).index if int(s) in coords]

    daily_blocks: Dict[str, object] = {}

    daily_pooled = daily_series_by_station(df, top, subtract_network_mean=False)
    pc_daily = daily_pair_correlations(daily_pooled, coords)
    daily_blocks["pooled"] = {
        "n_pairs": int(len(pc_daily)),
        "distance_binned": bin_stats(pc_daily),
    }

    daily_cm = daily_series_by_station(df, top, subtract_network_mean=True)
    pc_cm = daily_pair_correlations(daily_cm, coords)
    daily_blocks["network_common_mode_removed"] = {
        "n_pairs": int(len(pc_cm)),
        "distance_binned": bin_stats(pc_cm),
        "mechanical_bias_note": (
            "Removing the per-day network mean over N stations induces a "
            "mechanical pairwise bias ~ -1/(N-1) = %.3f; read bin means "
            "relative to that floor, not zero." % (-1.0 / (len(top) - 1))
        ),
        "expected_mechanical_bias": -1.0 / (len(top) - 1),
    }

    per_sat: Dict[str, object] = {}
    for sat in sorted(df["satellite"].unique()):
        ds = daily_series_by_station(df[df["satellite"] == sat], top)
        pc_s = daily_pair_correlations(ds, coords)
        per_sat[str(sat)] = {
            "n_pairs": int(len(pc_s)),
            "distance_binned": bin_stats(pc_s),
        }
    daily_blocks["per_satellite"] = per_sat
    results["controls_BC_daily_aggregation"] = daily_blocks

    return results


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Step 2.7: orbit/common-mode confound controls")
    ap.add_argument("--input", default=str(RESULTS_DIR / "step_2_1_slr_residuals.csv"))
    ap.add_argument("--coords", default=str(RESULTS_DIR / "station_coordinates.json"))
    ap.add_argument("--n-permutations", type=int, default=2000)
    ap.add_argument("--output", default=str(RESULTS_DIR / "step_2_7_orbit_commonmode_control.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    n0 = len(df)
    df = df[df["residual_m"].abs() < RESIDUAL_THRESHOLD_M].copy()
    logger.info(f"Loaded {n0} residuals; {len(df)} after |residual| < {RESIDUAL_THRESHOLD_M} m")

    coords = {int(k): v for k, v in json.load(open(args.coords)).items()}
    logger.info(f"Station coordinates: {len(coords)}")

    results = run(df, coords, n_perm=int(args.n_permutations))
    results["input_file"] = str(args.input)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
