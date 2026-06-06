#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force unbuffered output for real-time logging
os.environ["PYTHONUNBUFFERED"] = "1"

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import TEPLogger, set_step_logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "slr"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

for d in [DATA_DIR, LOGS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = TEPLogger("step_2_0", log_file_path=LOGS_DIR / "step_2_0_slr_analysis.log")
set_step_logger(logger)

C_M_S = 299_792_458.0


def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def infer_date_from_filename(path: Path) -> Optional[date]:
    m = re.search(r"(\d{8})(\d{4})?", path.name)
    if not m:
        return None
    ymd = m.group(1)
    try:
        return datetime.strptime(ymd, "%Y%m%d").date()
    except Exception:
        return None


def parse_float(x: str) -> Optional[float]:
    try:
        if x.lower() == "na":
            return None
        return float(x)
    except Exception:
        return None


@dataclass
class NPObs:
    file_date: str
    file_name: str
    source_kind: str
    satellite: str
    station: str
    sec_of_day: float
    tof_s: float
    range_km: float
    elev_deg: Optional[float]
    n_returns: Optional[float]
    extra_1: Optional[float]
    extra_2: Optional[float]
    extra_3: Optional[float]
    extra_4: Optional[float]


def iter_np2_observations(file_path: Path) -> Iterable[NPObs]:
    file_date = infer_date_from_filename(file_path)
    if file_date is None:
        return

    name = file_path.name
    source_kind = "daily" if re.search(r"\d{8}\.np2$", name) else "intra_day"

    current_satellite: Optional[str] = None
    current_station: Optional[str] = None

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if not parts:
                continue

            tag = parts[0].lower()

            if tag == "h3" and len(parts) >= 2:
                current_satellite = parts[1]
                continue

            if tag == "c0" and len(parts) >= 4:
                current_station = parts[3]
                continue

            if tag != "11":
                continue

            if len(parts) < 5:
                continue

            sec = parse_float(parts[1])
            tof = parse_float(parts[2])
            station = parts[3]

            if sec is None or tof is None:
                continue

            satellite = current_satellite or "unknown"
            if current_station and station.lower() != current_station.lower():
                station_effective = station
            else:
                station_effective = station if station else (current_station or "unknown")

            range_km = (tof * C_M_S / 2.0) / 1000.0

            elev_deg = parse_float(parts[8]) if len(parts) > 8 else None
            n_returns = parse_float(parts[7]) if len(parts) > 7 else None

            extra_1 = parse_float(parts[9]) if len(parts) > 9 else None
            extra_2 = parse_float(parts[10]) if len(parts) > 10 else None
            extra_3 = parse_float(parts[11]) if len(parts) > 11 else None
            extra_4 = parse_float(parts[12]) if len(parts) > 12 else None

            yield NPObs(
                file_date=file_date.isoformat(),
                file_name=name,
                source_kind=source_kind,
                satellite=satellite,
                station=station_effective,
                sec_of_day=sec,
                tof_s=tof,
                range_km=range_km,
                elev_deg=elev_deg,
                n_returns=n_returns,
                extra_1=extra_1,
                extra_2=extra_2,
                extra_3=extra_3,
                extra_4=extra_4,
            )


def collect_files(root: Path, start: date, end: date) -> List[Path]:
    all_files = sorted(root.rglob("*.np2"))
    selected: List[Path] = []

    for p in all_files:
        d = infer_date_from_filename(p)
        if d is None:
            continue
        if start <= d <= end:
            selected.append(p)

    return selected


def atomic_write_json(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="TEP-SLR Step 2.0: Parse and summarize SLR NPT CRD (.np2) files")
    parser.add_argument("--crd-version", choices=["crd", "crd_v2"], default="crd_v2")
    parser.add_argument("--source", choices=["allsat", "satellite"], default="allsat")
    parser.add_argument("--satellite", default=None)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-01-02")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    start = parse_yyyy_mm_dd(args.start)
    end = parse_yyyy_mm_dd(args.end)
    if end < start:
        logger.error("--end must be >= --start")
        return 2

    if args.crd_version == "crd_v2":
        base = DATA_DIR / "npt_crd_v2"
    else:
        base = DATA_DIR / "npt_crd"

    if args.source == "allsat":
        root = base / "allsat"
    else:
        if not args.satellite:
            logger.error("--satellite is required when --source satellite")
            return 2
        root = base / args.satellite

    if not root.exists():
        logger.error(f"Data directory not found: {root}")
        return 2

    files = collect_files(root, start, end)
    if args.max_files:
        files = files[: args.max_files]

    logger.info(f"SLR analysis input root: {root}")
    logger.info(f"Date range: {start.isoformat()} to {end.isoformat()}")
    logger.info(f"Files selected: {len(files)}")

    obs: List[NPObs] = []
    for fp in files:
        try:
            obs.extend(list(iter_np2_observations(fp)))
        except Exception as e:
            logger.warning(f"Failed parsing {fp}: {type(e).__name__}: {e}")

    if not obs:
        logger.warning("No normal-point observations parsed (no '11' records found).")
        return 1

    df = pd.DataFrame([o.__dict__ for o in obs])

    df["range_km"] = pd.to_numeric(df["range_km"], errors="coerce")
    df["tof_s"] = pd.to_numeric(df["tof_s"], errors="coerce")

    daily_station = (
        df.groupby(["file_date", "station"], dropna=False)
        .agg(
            n_obs=("range_km", "size"),
            n_files=("file_name", "nunique"),
            range_mean_km=("range_km", "mean"),
            range_std_km=("range_km", "std"),
            tof_mean_s=("tof_s", "mean"),
            tof_std_s=("tof_s", "std"),
        )
        .reset_index()
        .sort_values(["file_date", "n_obs"], ascending=[True, False])
    )

    daily_counts = df.groupby(["file_date"]).size().reset_index(name="n_obs")

    summary = {
        "spec": {
            "data_type": "npt",
            "crd_version": args.crd_version,
            "source": args.source,
            "satellite": args.satellite,
        },
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "counts": {
            "files": len(files),
            "observations_11": int(df.shape[0]),
            "stations": int(df["station"].nunique()),
            "satellites": int(df["satellite"].nunique()),
        },
        "top_stations_by_obs": (
            df.groupby("station").size().sort_values(ascending=False).head(25).to_dict()
        ),
        "top_satellites_by_obs": (
            df.groupby("satellite").size().sort_values(ascending=False).head(25).to_dict()
        ),
    }

    out_json = OUTPUTS_DIR / "step_2_0_slr_analysis_summary.json"
    out_csv = OUTPUTS_DIR / "step_2_0_slr_daily_station_summary.csv"

    atomic_write_json(summary, out_json)
    daily_station.to_csv(out_csv, index=False)

    try:
        plt.figure(figsize=(10, 4))
        plt.plot(pd.to_datetime(daily_counts["file_date"]), daily_counts["n_obs"], marker="o", linewidth=1)
        plt.xlabel("File date")
        plt.ylabel("Normal points (11 records)")
        plt.title("SLR normal-point volume by day")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "step_2_0_slr_counts_by_day.png", dpi=160)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.hist(df["range_km"].dropna().values, bins=80)
        plt.xlabel("Range (km) derived from time-of-flight")
        plt.ylabel("Count")
        plt.title("SLR range distribution (from NPT time-of-flight)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "step_2_0_slr_range_hist.png", dpi=160)
        plt.close()
    except Exception as e:
        logger.warning(f"Plotting failed: {type(e).__name__}: {e}")

    logger.success(f"Step 2.0 complete. Wrote: {out_json} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
