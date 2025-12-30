#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import sys
import time
import gzip
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force unbuffered output for real-time logging
os.environ["PYTHONUNBUFFERED"] = "1"

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import TEPLogger, set_step_logger
from utils.plot_style import apply_paper_style

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "slr"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

for d in [DATA_ROOT, LOGS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = TEPLogger("step_2_1", log_file_path=LOGS_DIR / "step_2_1_slr_residuals.log")
set_step_logger(logger)

C_M_S = 299_792_458.0


def normalize_satellite_name(s: str) -> str:
    raw = str(s or "").strip().lower()
    if not raw:
        return "unknown"

    # Common CRD naming variants: remove separators.
    key = re.sub(r"[^a-z0-9]+", "", raw)

    # Canonicalize known targets.
    aliases = {
        "l51": "lageos1",
        "lageos1": "lageos1",
        "lageosi": "lageos1",
        "lageos01": "lageos1",
        "l52": "lageos2",
        "lageos2": "lageos2",
        "lageosii": "lageos2",
        "lageos02": "lageos2",
        "l53": "lares",
        "lares": "lares",
        "lares1": "lares",
        "l54": "lares2",
        "lares2": "lares2",
        "et1": "etalon1",
        "etalon1": "etalon1",
        "etalon01": "etalon1",
        "et2": "etalon2",
        "etalon2": "etalon2",
        "etalon02": "etalon2",
    }

    return aliases.get(key, raw)


def get_auth() -> Optional[Tuple[str, str]]:
    import netrc

    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            return (auth[0], auth[2])
    except Exception:
        pass

    user = os.getenv("CDDIS_USER")
    passwd = os.getenv("CDDIS_PASS")
    if user and passwd:
        return (user, passwd)

    return None


def parse_yyyy_mm_dd(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _extract_filename_from_list_line(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None
    if s.lower().startswith("total"):
        return None

    parts = s.split()
    if not parts:
        return None

    if parts[0].startswith("-") or parts[0].startswith("d"):
        name = parts[-1]
    else:
        name = parts[0]

    name = name.strip()
    if name in {".", ".."}:
        return None
    return name


def list_remote_files(session: requests.Session, dir_url: str, timeout_s: int = 60) -> List[str]:
    candidates = [
        f"{dir_url.rstrip('/') }/*?list",
        f"{dir_url.rstrip('/') }/?list",
        f"{dir_url.rstrip('/') }/"
    ]

    last_err = None
    for url in candidates:
        try:
            resp = session.get(url, timeout=timeout_s)
            if resp.status_code == 200:
                files: List[str] = []
                
                # Check for HTML content (directory index)
                if "<html" in resp.text.lower() or "<!doctype" in resp.text.lower():
                    # Parse links from HTML
                    for line in resp.text.splitlines():
                        m = re.search(r'href="([^"]+)"', line)
                        if m:
                            name = m.group(1)
                            # Filter out parent dirs, query params, absolute paths if needed
                            if name in ['../', './'] or name.startswith('?') or name.startswith('/'):
                                continue
                            # Remove trailing slash
                            files.append(name.rstrip('/'))
                    if files:
                        return files
                        
                for ln in resp.text.splitlines():
                    name = _extract_filename_from_list_line(ln)
                    if name:
                        files.append(name)
                
                if files:
                    return files
                    
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    raise RuntimeError(f"Failed to list remote directory: {dir_url} ({last_err})")


def download_file(session: requests.Session, url: str, dest_path: Path, timeout_s: int = 600, retries: int = 3) -> Path:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_path.with_suffix(dest_path.suffix + ".part")

    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, timeout=timeout_s, stream=True) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}")

                n = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        n += len(chunk)

            if n <= 0:
                raise RuntimeError("Downloaded 0 bytes")

            tmp.replace(dest_path)
            return dest_path
        except Exception as e:
            last_exc = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(min(30, 2**attempt))

    raise RuntimeError(f"Failed to download after {retries} attempts: {url} ({type(last_exc).__name__}: {last_exc})")


def infer_date_from_np2_filename(path: Path) -> Optional[date]:
    m = re.search(r"(\d{8})(\d{4})?", path.name)
    if not m:
        return None
    ymd = m.group(1)
    try:
        return datetime.strptime(ymd, "%Y%m%d").date()
    except Exception:
        return None


def collect_np2_files(root: Path, start: date, end: date, max_files: int = 0) -> List[Path]:
    # Collect both .np2 and .npt files (CRD format)
    files_all = sorted(list(root.rglob("*.np2")) + list(root.rglob("*.npt")))
    selected: List[Path] = []
    for p in files_all:
        d = infer_date_from_np2_filename(p)
        if d is None:
            continue
        if start <= d <= end:
            selected.append(p)

    if max_files:
        selected = selected[:max_files]

    return selected


def parse_float(x: str) -> Optional[float]:
    try:
        if x.lower() == "na":
            return None
        return float(x)
    except Exception:
        return None


@dataclass
class CRDObs:
    epoch_utc: datetime
    station: str
    satellite: str
    tof_s: float
    obs_range_m: float
    elev_deg_reported: Optional[float]
    file_name: str


def iter_crd_observations(np2_path: Path) -> Iterable[CRDObs]:
    current_satellite: Optional[str] = None
    current_station: Optional[str] = None
    current_station_cdp: Optional[str] = None
    current_station_timescale: Optional[int] = None
    current_sysconfig_id: Optional[str] = None
    current_session_date: Optional[date] = None
    last_elev_deg: Optional[float] = None

    with open(np2_path, "r", encoding="utf-8", errors="replace") as f:
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

            if tag == "h2" and len(parts) >= 3:
                # H2 <station_name> <cdp_pad_id> ...
                current_station = parts[1]
                try:
                    current_station_cdp = str(int(parts[2]))
                except Exception:
                    current_station_cdp = None
                try:
                    current_station_timescale = int(parts[5]) if len(parts) >= 6 else None
                except Exception:
                    current_station_timescale = None
                continue

            if tag == "h4" and len(parts) >= 5:
                # h4 1 YYYY MM DD HH MM SS YYYY MM DD HH MM SS ...
                try:
                    y = int(parts[2])
                    m = int(parts[3])
                    d = int(parts[4])
                    current_session_date = date(y, m, d)
                except Exception:
                    pass
                continue

            if tag == "c0" and len(parts) >= 4:
                # C0 format: C0 <version> <CDP_number> <station_code> ...
                # Extract CDP numeric code (column 2) for SLRF2020 matching
                # Keep alphanumeric code as fallback
                current_sysconfig_id = parts[3]
                continue

            if tag == "30" and len(parts) >= 4:
                # 30 sec_of_day azimuth elevation ...
                last_elev_deg = parse_float(parts[3])
                continue

            if tag != "11":
                continue

            if len(parts) < 3:
                continue

            if current_session_date is None:
                # Fallback to file date if session date missing
                fd = infer_date_from_np2_filename(np2_path)
                if fd is None:
                    continue
                current_session_date = fd

            sec_of_day = parse_float(parts[1])
            tof_s = parse_float(parts[2])  # CRD format: time-of-flight in SECONDS
            if sec_of_day is None or tof_s is None:
                continue

            obs_range_m = tof_s * C_M_S / 2.0

            elev_deg = last_elev_deg

            epoch = datetime(
                current_session_date.year,
                current_session_date.month,
                current_session_date.day,
                tzinfo=timezone.utc,
            ) + timedelta(seconds=float(sec_of_day))

            sat = normalize_satellite_name(current_satellite or "unknown")
            # Prioritize CDP numeric code for SLRF2020 matching, fallback to alphanumeric
            sta = current_station_cdp or (current_station or "unknown")

            yield CRDObs(
                epoch_utc=epoch,
                station=sta,
                satellite=sat,
                tof_s=float(tof_s),
                obs_range_m=float(obs_range_m),
                elev_deg_reported=elev_deg,
                file_name=np2_path.name,
            )


# --- Simple ECEF helpers ---

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def ecef_to_lla(x: float, y: float, z: float) -> Tuple[float, float, float]:
    # Returns lat (rad), lon (rad), h (m). Iterative Bowring-like
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


def ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> np.ndarray:
    slat = math.sin(lat_rad)
    clat = math.cos(lat_rad)
    slon = math.sin(lon_rad)
    clon = math.cos(lon_rad)

    # Rows: [E; N; U]
    return np.array(
        [
            [-slon, clon, 0.0],
            [-slat * clon, -slat * slon, clat],
            [clat * clon, clat * slon, slat],
        ],
        dtype=float,
    )


# --- SINEX station positions (SLRF2020) ---


def sinex_epoch_to_datetime(epoch: str) -> Optional[datetime]:
    # Common SINEX epoch representation: YY:DOY:SSSSS
    m = re.match(r"^(\d{2}):(\d{3}):(\d{5})$", epoch.strip())
    if not m:
        return None
    yy = int(m.group(1))
    doy = int(m.group(2))
    sec = int(m.group(3))
    year = 2000 + yy if yy <= 79 else 1900 + yy
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, seconds=sec)
    return dt


@dataclass
class StationState:
    x_m: float
    y_m: float
    z_m: float
    vx_m_per_yr: float
    vy_m_per_yr: float
    vz_m_per_yr: float
    ref_epoch_utc: datetime


def build_station_code_mapping(sinex_path: Path) -> Dict[str, str]:
    """
    Build mapping from CRD alphanumeric station codes to SINEX numeric CDP codes.
    Parses SITE/ID block from ILRS Data Handling File or SLRF2020 SINEX.
    
    SINEX SITE/ID format (fixed columns):
    Columns 1-6: CDP numeric code (e.g., "7396")
    Columns 30-50: Station description containing abbreviations
    
    Returns: dict mapping lowercase alphanumeric codes to numeric CDP codes.
    """
    # Legacy station code mappings for obsolete CDP codes
    # These stations were replaced/renumbered in SLRF2020
    mapping = {
        "532": "7821",  # Shanghai Observatory (old) -> Shanghai SO FIXED (SLRF2020)
        "shao": "7821", # Shanghai Observatory alphanumeric code
    }
    opener = gzip.open if sinex_path.suffix == ".gz" else open
    
    with opener(sinex_path, "rt", encoding="utf-8", errors="replace") as f:
        in_site_id = False
        for line in f:
            if line.startswith("+SITE/ID"):
                in_site_id = True
                continue
            if line.startswith("-SITE/ID"):
                break
            if not in_site_id or line.startswith("*") or not line.strip():
                continue
            
            # Extract CDP code (first field, columns 1-6)
            parts = line.split()
            if len(parts) < 5:
                continue
            
            numeric_code = parts[0]
            
            # Extract station description (columns ~30-50)
            # Format: "Station_Name ABBREVIATION FIXED/QUASAR/etc"
            if len(line) > 30:
                desc = line[30:60].strip()
                # Split description into words
                words = desc.split()
                
                for word in words:
                    # Clean and normalize
                    clean = word.strip('.,;:').lower()
                    
                    # Skip common non-station identifiers
                    if clean in ['fixed', 'quasar', 'a', 'l', 'p', 'so', 'gdr', 'kiev', 
                                  'lviv', 'mdn', 'mdn2', 'kom', 'sim', 'mdvs', 'altl',
                                  'riga', 'arkl', 'bail', 'irkl', 'crimea', 'cuba',
                                  'tlrs-1', 'tlrs-2', 'tlrs-3', 'tlrs-4', 'mtlrs-1',
                                  'moblas-1', 'moblas-2', 'moblas-3', 'moblas-4',
                                  'moblas-5', 'moblas-6', 'moblas-7', 'moblas-8']:
                        continue
                    
                    # Valid station codes are typically 3-8 characters
                    if clean and 3 <= len(clean) <= 8:
                        # Map alphanumeric code to numeric CDP code
                        # Don't overwrite existing mappings (first occurrence wins)
                        if clean not in mapping:
                            mapping[clean] = numeric_code
    
    return mapping


def load_slrf2020_station_states(sinex_path: Path) -> Dict[str, StationState]:
    stations: Dict[str, Dict[str, Tuple[float, datetime]]] = {}

    in_est = False
    opener = gzip.open if sinex_path.suffix == ".gz" else open
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

            # SINEX estimate lines are fixed-format; split is OK for our needs.
            parts = line.split()
            if len(parts) < 9:
                continue

            param = parts[1].upper()
            site = parts[2].lower()
            epoch = parts[5]
            value = parts[8]

            if param not in {"STAX", "STAY", "STAZ", "VELX", "VELY", "VELZ"}:
                continue

            v = parse_float(value)
            if v is None:
                continue

            dt = sinex_epoch_to_datetime(epoch)
            if dt is None:
                continue

            if site not in stations:
                stations[site] = {}

            # Keep the most recent reference epoch for each parameter
            prev = stations[site].get(param)
            if prev is None or dt > prev[1]:
                stations[site][param] = (float(v), dt)

    out: Dict[str, StationState] = {}
    for site, params in stations.items():
        if not all(k in params for k in ["STAX", "STAY", "STAZ"]):
            continue

        ref_epoch = params["STAX"][1]
        x = params["STAX"][0]
        y = params["STAY"][0]
        z = params["STAZ"][0]

        vx = params.get("VELX", (0.0, ref_epoch))[0]
        vy = params.get("VELY", (0.0, ref_epoch))[0]
        vz = params.get("VELZ", (0.0, ref_epoch))[0]

        out[site] = StationState(
            x_m=x,
            y_m=y,
            z_m=z,
            vx_m_per_yr=vx,
            vy_m_per_yr=vy,
            vz_m_per_yr=vz,
            ref_epoch_utc=ref_epoch,
        )

    return out


def propagate_station(state: StationState, t: datetime) -> np.ndarray:
    dt_sec = (t - state.ref_epoch_utc).total_seconds()
    dt_yr = dt_sec / (365.25 * 86400.0)
    return np.array(
        [
            state.x_m + state.vx_m_per_yr * dt_yr,
            state.y_m + state.vy_m_per_yr * dt_yr,
            state.z_m + state.vz_m_per_yr * dt_yr,
        ],
        dtype=float,
    )


# --- Orbit parsing (CPF predictions or SP3 precise orbits) ---


@dataclass
class SatelliteEphemeris:
    """Satellite ephemeris from CPF (predicted) or SP3 (precise) orbits."""
    epochs: np.ndarray  # unix seconds float
    pos_m: np.ndarray  # shape (n,3)
    source: str  # 'cpf' or 'sp3'
    vel_m_per_s: Optional[np.ndarray] = None  # shape (n,3), if present (SP3 'V' records)


def parse_sp3_positions(sp3_path: Path, sat_id: Optional[str] = None) -> SatelliteEphemeris:
    """
    Parse SP3 precise orbit file (ILRS standard format).
    SP3 format: Position records start with 'P' followed by satellite ID and XYZ coordinates.
    Epoch records start with '*' followed by date/time.
    
    Args:
        sp3_path: Path to SP3 file
        sat_id: Satellite ID to extract (e.g., 'L51' for LAGEOS-1, 'L52' for LAGEOS-2).
                If None, extracts the first satellite found.
    
    Returns ephemeris with ~1-2 cm accuracy for LAGEOS.
    """
    epochs: List[float] = []
    pos: List[List[float]] = []
    vel: List[Optional[List[float]]] = []
    current_epoch_unix: Optional[float] = None
    detected_sat_id: Optional[str] = None
    
    opener = gzip.open if sp3_path.suffix == '.gz' else open
    
    with opener(sp3_path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            
            # Epoch record: * YYYY MM DD HH MM SS.SSSSSSSS
            if s.startswith('*'):
                parts = s.split()
                if len(parts) >= 7:
                    try:
                        year = int(parts[1])
                        month = int(parts[2])
                        day = int(parts[3])
                        hour = int(parts[4])
                        minute = int(parts[5])
                        second = float(parts[6])
                        
                        # Handle malformed SP3 epochs where minute=60 (should roll to next hour)
                        if minute >= 60:
                            hour += minute // 60
                            minute = minute % 60
                        if hour >= 24:
                            day += hour // 24
                            hour = hour % 24
                        
                        epoch = datetime(year, month, day, hour, minute, int(second), tzinfo=timezone.utc)
                        epoch += timedelta(seconds=second - int(second))
                        current_epoch_unix = epoch.timestamp()
                    except Exception:
                        continue
            
            # Position record: P<SAT_ID> X Y Z CLOCK
            # Example: PL51 -6259769.495  9366893.831  4906622.372    123.456789
            # Satellite IDs: L51=LAGEOS-1, L52=LAGEOS-2
            elif s.startswith('P') and current_epoch_unix is not None:
                # Extract satellite ID (characters 1-4, e.g., "L51 ")
                record_sat_id = s[1:4].strip()
                
                # Auto-detect satellite ID from first record if not specified
                if sat_id is None and detected_sat_id is None:
                    detected_sat_id = record_sat_id
                    logger.info(f"SP3 auto-detected satellite ID: {detected_sat_id}")
                
                # Filter by satellite ID
                target_sat_id = sat_id if sat_id is not None else detected_sat_id
                if record_sat_id != target_sat_id:
                    continue
                
                parts = s.split()
                if len(parts) >= 4:
                    try:
                        # SP3 positions are in km, convert to meters
                        x = float(parts[1]) * 1000.0
                        y = float(parts[2]) * 1000.0
                        z = float(parts[3]) * 1000.0

                        # Some SP3 files can contain repeated epochs.
                        # Ensure epochs are strictly unique to support interpolation.
                        if epochs and abs(current_epoch_unix - epochs[-1]) < 1e-6:
                            pos[-1] = [x, y, z]
                            vel[-1] = None
                        else:
                            epochs.append(current_epoch_unix)
                            pos.append([x, y, z])
                            vel.append(None)
                    except Exception:
                        continue

            # Velocity record: V<SAT_ID> X Y Z CLOCK_RATE
            # Example: VL51  -9659.511893 -30055.311193 -51948.510855
            # In SP3-c, velocities are given in 10^-4 km/s (i.e., 0.1 m/s) for X,Y,Z.
            # (This matches typical magnitudes of a few*10^4 -> a few km/s.)
            elif s.startswith('V') and current_epoch_unix is not None:
                record_sat_id = s[1:4].strip()
                target_sat_id = sat_id if sat_id is not None else detected_sat_id
                if target_sat_id is None:
                    # If we haven't seen a position record yet, we can't reliably bind this.
                    continue
                if record_sat_id != target_sat_id:
                    continue
                parts = s.split()
                if len(parts) >= 4 and epochs and abs(current_epoch_unix - epochs[-1]) < 1e-6:
                    try:
                        vx = float(parts[1]) * 0.1
                        vy = float(parts[2]) * 0.1
                        vz = float(parts[3]) * 0.1
                        vel[-1] = [vx, vy, vz]
                    except Exception:
                        continue
    
    if not epochs:
        raise RuntimeError(f"No position records parsed from SP3: {sp3_path}")

    # Fill missing intermediate epochs (some ILRS SP3 have occasional 2*dt gaps).
    # This avoids interpolation overshoot from uneven time spacing.
    if len(epochs) >= 4:
        dts = np.diff(np.array(epochs, dtype=float))
        dt_med = float(np.median(dts)) if len(dts) else 0.0
        if dt_med > 0:
            e_out: List[float] = [epochs[0]]
            p_out: List[List[float]] = [pos[0]]
            v_out: List[Optional[List[float]]] = [vel[0]]
            for i in range(1, len(epochs)):
                e_prev = e_out[-1]
                p_prev = np.array(p_out[-1], dtype=float)
                v_prev = v_out[-1]
                e_next = epochs[i]
                p_next = np.array(pos[i], dtype=float)
                v_next = vel[i]
                gap = e_next - e_prev
                if gap > 1.5 * dt_med and gap < 10.0 * dt_med:
                    steps = int(round(gap / dt_med))
                    if steps >= 2:
                        for j in range(1, steps):
                            ej = e_prev + j * dt_med
                            u = (ej - e_prev) / (e_next - e_prev)

                            # Prefer a smooth cubic Hermite fill using neighbor points
                            # (fallback to linear if neighbors unavailable).
                            if 2 <= i < (len(epochs) - 1):
                                e_before = epochs[i - 2]
                                p_before = np.array(pos[i - 2], dtype=float)
                                e_after = epochs[i + 1]
                                p_after = np.array(pos[i + 1], dtype=float)

                                v0 = (p_next - p_before) / (e_next - e_before)
                                v1 = (p_after - p_prev) / (e_after - e_prev)
                                dt = (e_next - e_prev)

                                u2 = u * u
                                u3 = u2 * u
                                h00 = 2.0 * u3 - 3.0 * u2 + 1.0
                                h10 = u3 - 2.0 * u2 + u
                                h01 = -2.0 * u3 + 3.0 * u2
                                h11 = u3 - u2
                                pj = h00 * p_prev + h10 * (dt * v0) + h01 * p_next + h11 * (dt * v1)
                            else:
                                pj = (1.0 - u) * p_prev + u * p_next
                            if v_prev is not None and v_next is not None:
                                vj = (1.0 - u) * np.array(v_prev, dtype=float) + u * np.array(v_next, dtype=float)
                                vj_out: Optional[List[float]] = [float(vj[0]), float(vj[1]), float(vj[2])]
                            else:
                                vj_out = None
                            e_out.append(float(ej))
                            p_out.append([float(pj[0]), float(pj[1]), float(pj[2])])
                            v_out.append(vj_out)
                e_out.append(float(e_next))
                p_out.append([float(p_next[0]), float(p_next[1]), float(p_next[2])])
                v_out.append(v_next)

            epochs = e_out
            pos = p_out
            vel = v_out
    
    return SatelliteEphemeris(
        epochs=np.array(epochs, dtype=float),
        pos_m=np.array(pos, dtype=float),
        source='sp3',
        vel_m_per_s=(
            np.array(vel, dtype=float)
            if vel and all(vv is not None for vv in vel)
            else None
        ),
    )


def parse_cpf_positions(cpf_path: Path) -> SatelliteEphemeris:
    epochs: List[float] = []
    pos: List[List[float]] = []

    with open(cpf_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if not parts:
                continue

            # CPF data record types are numeric; position records are typically type 10.
            if parts[0] != "10":
                continue

            # CPF v2 Record 10 format:
            # 10 Direction MJD Seconds LeapSec X Y Z
            # Example: 10 0 61032 0.00000 0 -5834072.853 -4278955.326 -3106022.413

            try:
                # Check if it looks like the YYYY MM DD format (old v1 or alternate v2)
                # Or MJD format (standard v2)
                
                # Heuristic: MJD is typically 5 digits (e.g. 59000), Year is 4 digits (2024)
                p2 = float(parts[2])

                if p2 > 30000: # MJD format
                    mjd = int(p2)
                    sec_of_day = float(parts[3])
                    # MJD 40587 = 1970-01-01
                    t_unix = (mjd - 40587) * 86400.0 + sec_of_day
                    
                    x = float(parts[5])
                    yv = float(parts[6])
                    z = float(parts[7])
                else:
                    # YYYY MM DD HH MM SS format
                    y = int(parts[1])
                    mo = int(parts[2])
                    d = int(parts[3])
                    hh = int(parts[4])
                    mm = int(parts[5])
                    ss = float(parts[6])
                    
                    epoch = datetime(y, mo, d, hh, mm, int(ss), tzinfo=timezone.utc) + timedelta(seconds=ss - int(ss))
                    t_unix = epoch.timestamp()
                    
                    x = float(parts[7])
                    yv = float(parts[8])
                    z = float(parts[9])

            except Exception:
                continue

            # Units heuristic: CPF positions are typically in meters
            # LAGEOS orbital radius ~12 million meters from Earth center
            # If values are < 50,000 (typical km range), assume km and convert to meters
            scale = 1000.0 if max(abs(x), abs(yv), abs(z)) < 50000.0 else 1.0
            epochs.append(t_unix)
            pos.append([x * scale, yv * scale, z * scale])

    if not epochs:
        raise RuntimeError(f"No position records parsed from CPF: {cpf_path}")

    return SatelliteEphemeris(
        epochs=np.array(epochs, dtype=float),
        pos_m=np.array(pos, dtype=float),
        source='cpf'
    )


def interp_sat_pos(eph: SatelliteEphemeris, t_unix: float) -> np.ndarray:
    # If SP3 velocities are available, prefer cubic Hermite interpolation using
    # endpoint velocities (avoids large Catmull–Rom overshoot on short arcs).
    t = eph.epochs
    p = eph.pos_m
    v = eph.vel_m_per_s

    n = len(t)
    if n == 0:
        raise RuntimeError("Empty ephemeris")
    if n == 1:
        return p[0]

    if t_unix <= t[0]:
        return p[0]
    if t_unix >= t[-1]:
        return p[-1]

    idx = int(np.searchsorted(t, t_unix))

    # Linear fallback near boundaries
    if idx <= 1 or idx >= n - 1:
        i0 = max(0, idx - 1)
        i1 = min(n - 1, idx)
        if i0 == i1:
            return p[i0]
        t0, t1 = t[i0], t[i1]
        w = (t_unix - t0) / (t1 - t0)
        return (1.0 - w) * p[i0] + w * p[i1]

    # Velocity-based Hermite between bracketing points
    if v is not None:
        i1 = idx - 1
        i2 = idx
        t1, t2 = t[i1], t[i2]
        if t2 == t1:
            return p[i1]
        u = (t_unix - t1) / (t2 - t1)
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        dt = (t2 - t1)
        return h00 * p[i1] + h10 * (dt * v[i1]) + h01 * p[i2] + h11 * (dt * v[i2])

    # Local cubic Hermite (Catmull–Rom style) using 4 neighboring points.
    # Used only when explicit velocities are unavailable.
    i1 = idx - 1
    i2 = idx
    i0 = i1 - 1
    i3 = i2 + 1

    t1, t2 = t[i1], t[i2]
    if t2 == t1:
        return p[i1]

    u = (t_unix - t1) / (t2 - t1)
    u2 = u * u
    u3 = u2 * u

    # Tangents estimated from adjacent segment velocities (stable for uneven spacing)
    v10 = (p[i1] - p[i0]) / (t[i1] - t[i0])
    v21 = (p[i2] - p[i1]) / (t[i2] - t[i1])
    v32 = (p[i3] - p[i2]) / (t[i3] - t[i2])
    m1 = 0.5 * (v10 + v21)
    m2 = 0.5 * (v21 + v32)

    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2

    dt = (t2 - t1)
    return h00 * p[i1] + h10 * (dt * m1) + h01 * p[i2] + h11 * (dt * m2)


def compute_sagnac_effect_m(sta_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    """
    Compute Sagnac correction for one-way signal path in ECEF.
    Corrects for Earth rotation during signal flight time when working in ECEF.
    
    Formula: delta_L = (omega_e / c) * (xs * yr - ys * xr)
    
    Args:
        sta_ecef: Receiver position [x, y, z] (m)
        sat_ecef: Satellite position [x, y, z] (m)
        
    Returns:
        Correction in meters.
        Note: The geometric range in ECEF is smaller than the true inertial path
        for eastward propagation. This correction should be ADDED to geometric ECEF range
        to get proper inertial range equivalent.
    """
    OMEGA_E = 7.2921151467e-5  # rad/s
    C = 299792458.0
    
    # Cross product z-component: (r_sta x r_sat)_z  (Sender x Receiver)
    # Sagnac effect extends the path if signal propagates East (with rotation).
    # Standard formula for one-way range increase:
    # dL = (omega / c) * (x_sta * y_sat - y_sta * x_sat)
    
    # Ensure inputs are floats
    xs, ys = float(sta_ecef[0]), float(sta_ecef[1])
    xr, yr = float(sat_ecef[0]), float(sat_ecef[1])
    
    term = xs * yr - ys * xr
    val = (OMEGA_E / C) * term
    # Debug print to stderr to see in command output
    if abs(val) < 1e-9:
        sys.stderr.write(f"SAGNAC ZERO: sta=({xs:.1f},{ys:.1f}) sat=({xr:.1f},{yr:.1f}) term={term:.1f} val={val:.1e}\n")
    return val


def compute_elevation_deg(sta_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> float:
    lat, lon, _h = ecef_to_lla(float(sta_ecef_m[0]), float(sta_ecef_m[1]), float(sta_ecef_m[2]))
    r_enu = ecef_to_enu_matrix(lat, lon)
    los = sat_ecef_m - sta_ecef_m
    enu = r_enu @ los
    e, n, u = float(enu[0]), float(enu[1]), float(enu[2])
    horiz = math.sqrt(e * e + n * n)
    return math.degrees(math.atan2(u, horiz))


def compute_tropospheric_delay_m(elev_deg: float, lat_deg: float, height_m: float) -> float:
    """
    Compute two-way tropospheric delay using Marini-Murray model.
    
    Reference: Marini & Murray (1973), Correction of laser range tracking data
               for atmospheric refraction at elevations above 10 degrees.
    
    Args:
        elev_deg: Elevation angle in degrees
        lat_deg: Station geodetic latitude in degrees
        height_m: Station height above ellipsoid in meters
    
    Returns:
        Two-way tropospheric delay in meters (to be SUBTRACTED from observed range)
    """
    if elev_deg < 2.0:
        elev_deg = 2.0  # Avoid numerical issues at very low elevations
    
    elev_rad = math.radians(elev_deg)
    lat_rad = math.radians(lat_deg)
    height_km = height_m / 1000.0
    
    # Standard atmosphere pressure at station height (mbar)
    P = 1013.25 * math.exp(-height_km / 8.5)
    
    # Zenith delay (one-way) using Marini-Murray formula
    f_lat = 1.0 - 0.00266 * math.cos(2 * lat_rad) - 0.00028 * height_km
    zenith_delay = 0.002277 * P / f_lat
    
    # Mapping function (Marini-Murray)
    sin_e = math.sin(elev_rad)
    tan_e = math.tan(elev_rad) if elev_deg < 89.9 else 1e10
    
    # Mapping function
    mapping = 1.0 / (sin_e + 0.00143 / (tan_e + 0.0455))
    
    # Two-way delay (laser goes up and back)
    return 2.0 * zenith_delay * mapping


# LAGEOS center-of-mass corrections (meters)
# From ILRS: retroreflector array offset from satellite center of mass
LAGEOS_COM_CORRECTION_M = {
    'lageos1': 0.251,  # 251 mm
    'lageos2': 0.251,  # 251 mm
}


def compute_sun_moon_positions(epoch: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute approximate Sun and Moon positions in GCRS (geocentric).
    Uses simplified analytical ephemerides sufficient for solid Earth tide calculations.
    
    Returns:
        (sun_gcrs_m, moon_gcrs_m): Position vectors in meters
    """
    # Julian date
    jd = (epoch - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 2451545.0
    T = (jd - 2451545.0) / 36525.0  # Julian centuries from J2000
    
    # Sun position (simplified)
    # Mean longitude of Sun
    L0 = math.radians(280.46646 + 36000.76983 * T)
    # Mean anomaly of Sun
    M = math.radians(357.52911 + 35999.05029 * T)
    # Equation of center
    C = math.radians((1.914602 - 0.004817 * T) * math.sin(M) + 0.019993 * math.sin(2 * M))
    # True longitude
    sun_lon = L0 + C
    # Obliquity of ecliptic
    eps = math.radians(23.439291 - 0.0130042 * T)
    # Distance (AU)
    sun_r_au = 1.000001018 * (1 - 0.016708634 * math.cos(M))
    AU_M = 149597870700.0  # meters per AU
    sun_r_m = sun_r_au * AU_M
    
    # Convert to GCRS
    sun_x = sun_r_m * math.cos(sun_lon)
    sun_y = sun_r_m * math.sin(sun_lon) * math.cos(eps)
    sun_z = sun_r_m * math.sin(sun_lon) * math.sin(eps)
    sun_gcrs = np.array([sun_x, sun_y, sun_z])
    
    # Moon position (simplified Brown theory)
    # Mean longitude of Moon
    Lm = math.radians(218.3165 + 481267.8813 * T)
    # Mean anomaly of Moon
    Mm = math.radians(134.9634 + 477198.8675 * T)
    # Mean elongation
    D = math.radians(297.8502 + 445267.1115 * T)
    # Argument of latitude
    F = math.radians(93.2721 + 483202.0175 * T)
    
    # Longitude correction
    dL = 6.289 * math.sin(Mm) - 1.274 * math.sin(2*D - Mm) + 0.658 * math.sin(2*D)
    moon_lon = Lm + math.radians(dL)
    
    # Latitude
    moon_lat = math.radians(5.128 * math.sin(F))
    
    # Distance (km)
    moon_r_km = 385000.56 - 20905.355 * math.cos(Mm)
    moon_r_m = moon_r_km * 1000.0
    
    # Convert to GCRS
    moon_x = moon_r_m * math.cos(moon_lat) * math.cos(moon_lon)
    moon_y = moon_r_m * math.cos(moon_lat) * math.sin(moon_lon) * math.cos(eps) - moon_r_m * math.sin(moon_lat) * math.sin(eps)
    moon_z = moon_r_m * math.cos(moon_lat) * math.sin(moon_lon) * math.sin(eps) + moon_r_m * math.sin(moon_lat) * math.cos(eps)
    moon_gcrs = np.array([moon_x, moon_y, moon_z])
    
    return sun_gcrs, moon_gcrs


def compute_solid_earth_tide_displacement(sta_ecef: np.ndarray, epoch: datetime) -> np.ndarray:
    """
    Compute solid Earth tide displacement at a station.
    Uses IERS 2010 conventions (simplified degree-2 terms).
    
    The displacement is primarily vertical (radial) with smaller horizontal components.
    Typical magnitude: 20-50 mm radial, 10-30 mm horizontal.
    
    Args:
        sta_ecef: Station ECEF coordinates in meters
        epoch: Observation epoch (UTC)
    
    Returns:
        Displacement vector in ECEF (meters) to be ADDED to nominal station position
    """
    # Love and Shida numbers (degree 2)
    h2 = 0.6078  # Vertical Love number
    l2 = 0.0847  # Horizontal Shida number
    
    # Gravitational constants
    GM_SUN = 1.32712440018e20  # m^3/s^2
    GM_MOON = 4.902800066e12   # m^3/s^2
    GM_EARTH = 3.986004418e14  # m^3/s^2
    RE = 6378136.6  # Earth equatorial radius (m)
    
    # Get Sun and Moon positions
    sun_gcrs, moon_gcrs = compute_sun_moon_positions(epoch)
    
    # Station unit vector
    sta_unit = sta_ecef / np.linalg.norm(sta_ecef)
    
    displacement = np.zeros(3)
    
    for body_pos, gm_body in [(sun_gcrs, GM_SUN), (moon_gcrs, GM_MOON)]:
        r_body = np.linalg.norm(body_pos)
        body_unit = body_pos / r_body
        
        # Cosine of angle between station direction and body direction
        cos_psi = np.dot(sta_unit, body_unit)
        
        # IERS 2010 Eq 7.5 - degree-2 solid tide
        # Factor = (GM_j/GM_E) * (R_E/r_j)^3 * R_E
        factor = (gm_body / GM_EARTH) * (RE / r_body)**3 * RE
        
        # Radial displacement
        dr = factor * h2 * (1.5 * cos_psi**2 - 0.5)
        
        # Transverse displacement (in plane containing station and body)
        sin_psi = math.sqrt(1.0 - cos_psi**2) if abs(cos_psi) < 0.9999 else 0.0
        dt = factor * 3.0 * l2 * cos_psi * sin_psi
        
        # Direction of transverse component
        transverse_dir = body_unit - cos_psi * sta_unit
        tn = np.linalg.norm(transverse_dir)
        if tn > 1e-10:
            transverse_dir = transverse_dir / tn
        else:
            transverse_dir = np.zeros(3)
        
        displacement += dr * sta_unit + dt * transverse_dir
    
    return displacement


def compute_ocean_loading_displacement(sta_ecef: np.ndarray, epoch: datetime) -> np.ndarray:
    """
    Compute ocean tide loading displacement at a station.
    Uses a simplified harmonic model for major tidal constituents.
    
    Ocean loading causes vertical displacements of 1-5 cm in coastal areas,
    less in continental interiors. The effect is smaller than solid Earth tides
    but can be significant for cm-level SLR.
    
    This is a simplified model using approximate global average amplitudes.
    For precise work, station-specific coefficients from IERS should be used.
    
    Args:
        sta_ecef: Station ECEF coordinates in meters
        epoch: Observation epoch (UTC)
    
    Returns:
        Displacement vector in ECEF (meters) to be ADDED to nominal station position
    """
    # Convert epoch to hours since J2000
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    hours_since_j2000 = (epoch - j2000).total_seconds() / 3600.0
    
    # Major tidal constituents (simplified global average amplitudes in mm)
    # M2: Principal lunar semidiurnal (period 12.42 hours)
    # S2: Principal solar semidiurnal (period 12.00 hours)
    # K1: Lunar-solar diurnal (period 23.93 hours)
    # O1: Principal lunar diurnal (period 25.82 hours)
    
    constituents = [
        ("M2", 12.4206, 15.0),   # period (hours), amplitude (mm)
        ("S2", 12.0000, 7.0),
        ("K1", 23.9345, 10.0),
        ("O1", 25.8193, 8.0),
    ]
    
    # Get station geodetic coordinates for coastal proximity estimate
    sta_norm = np.linalg.norm(sta_ecef)
    lat_rad = math.asin(sta_ecef[2] / sta_norm)
    lon_rad = math.atan2(sta_ecef[1], sta_ecef[0])
    
    # Simple coastal proximity factor (0.3-1.0 depending on assumed location)
    # Real implementation would use station-specific loading coefficients
    coastal_factor = 0.5  # Conservative average
    
    # Sum displacement from all constituents
    vertical_mm = 0.0
    for name, period_h, amp_mm in constituents:
        phase = 2.0 * math.pi * hours_since_j2000 / period_h
        # Add longitude-dependent phase shift
        if name in ["M2", "S2"]:
            phase += 2.0 * lon_rad  # Semidiurnal
        else:
            phase += lon_rad  # Diurnal
        vertical_mm += coastal_factor * amp_mm * math.cos(phase)
    
    # Convert to meters and apply as radial displacement
    vertical_m = vertical_mm / 1000.0
    sta_unit = sta_ecef / sta_norm
    
    return vertical_m * sta_unit


def compute_shapiro_delay_m(sta_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    """
    Compute Shapiro relativistic delay for the laser signal path.
    
    The Shapiro delay is the extra time taken by light passing through
    a gravitational field. For Earth's gravity, this is typically 1-2 cm
    for LAGEOS ranges.
    
    Formula: Δρ = (2 GM/c²) ln[(r_sta + r_sat + ρ) / (r_sta + r_sat - ρ)]
    
    Args:
        sta_ecef: Station ECEF coordinates (meters)
        sat_ecef: Satellite ECEF coordinates at bounce time (meters)
    
    Returns:
        One-way Shapiro delay in meters (multiply by 2 for two-way)
    """
    GM_EARTH = 3.986004418e14  # m^3/s^2
    
    r_sta = np.linalg.norm(sta_ecef)
    r_sat = np.linalg.norm(sat_ecef)
    rho = np.linalg.norm(sat_ecef - sta_ecef)
    
    # Shapiro delay formula
    numerator = r_sta + r_sat + rho
    denominator = r_sta + r_sat - rho
    
    if denominator <= 0:
        return 0.0
    
    shapiro_m = (2.0 * GM_EARTH / (C_M_S * C_M_S)) * math.log(numerator / denominator)
    
    return shapiro_m


def select_slrf2020_sinex_filename(files: List[str]) -> Optional[str]:
    """Pick the best SLRF2020 SINEX filename from a CDDIS directory listing."""
    sinex = [f for f in files if f.lower().endswith((".snx", ".snx.gz"))]
    if not sinex:
        return None

    # Prefer explicit SLRF2020 products over generic ILRS data handling files.
    slrf = [f for f in sinex if "slrf2020" in f.lower()]
    if slrf:
        return sorted(slrf)[-1]

    # Fallback: some archives may use different casing or naming; keep a conservative fallback.
    slrf_like = [f for f in sinex if "slrf" in f.lower() and "2020" in f.lower()]
    if slrf_like:
        return sorted(slrf_like)[-1]

    return None


def select_sp3_file_for_satellite(session: requests.Session, sat: str, start_date: datetime, end_date: datetime, cache_dir: Path) -> Optional[Path]:
    """
    Find SP3 precise orbit file for a given satellite and date range.
    First checks local cache, then downloads from CDDIS if needed.
    SP3 files are organized by week: YYMMDD corresponds to the Saturday of that week.
    Each file covers ~7 days, so one file typically covers multiple observation days.
    
    For date ranges >7 days, collects ALL overlapping SP3 files and merges them.
    
    Returns path to merged SP3 file, or None if not found.
    """
    base = "https://cddis.nasa.gov/archive/slr/products/orbits"
    sat_lower = sat.lower()
    sat_cache = cache_dir / sat_lower
    
    # Collect all SP3 files that overlap with the date range
    overlapping_files = []
    
    # First check local cache for existing SP3 files that cover the date range
    if sat_cache.exists():
        for week_dir in sorted(sat_cache.iterdir()):
            if not week_dir.is_dir():
                continue
            sp3_files = list(week_dir.glob("*.sp3*"))
            if not sp3_files:
                continue
            
            # Prefer ILRS combined solutions
            ilrs_files = [f for f in sp3_files if 'ilrsa' in f.name.lower() or 'ilrsb' in f.name.lower()]
            chosen = sorted(ilrs_files)[-1] if ilrs_files else sorted(sp3_files)[-1]
            
            try:
                test_eph = parse_sp3_positions(chosen, sat_id=None)
                if len(test_eph.epochs) == 0:
                    continue
                
                file_start = datetime.fromtimestamp(test_eph.epochs[0], tz=timezone.utc)
                file_end = datetime.fromtimestamp(test_eph.epochs[-1], tz=timezone.utc)
                
                # Accept file if it overlaps with observation period
                if file_end >= start_date and file_start <= end_date:
                    overlapping_files.append((chosen, file_start, file_end))
            except Exception:
                continue
    
    # If we found overlapping files, merge them if needed
    if overlapping_files:
        if len(overlapping_files) == 1:
            chosen = overlapping_files[0][0]
            logger.info(f"SP3 orbit found (cached): {chosen.name} (covers {overlapping_files[0][1].date()} to {overlapping_files[0][2].date()})")
            return chosen
        else:
            # Multiple files - merge them
            logger.info(f"SP3 orbit: merging {len(overlapping_files)} files for {sat}")
            for f, s, e in overlapping_files:
                logger.info(f"  {f.name} (covers {s.date()} to {e.date()})")
            
            # Merge ephemerides
            merged_epochs = []
            merged_pos = []
            
            for sp3_file, _, _ in sorted(overlapping_files, key=lambda x: x[1]):
                eph = parse_sp3_positions(sp3_file, sat_id=None)
                merged_epochs.extend(eph.epochs.tolist())
                merged_pos.extend(eph.pos_m.tolist())
            
            # Sort by epoch and remove duplicates
            combined = list(zip(merged_epochs, merged_pos))
            combined.sort(key=lambda x: x[0])
            
            # Remove duplicate epochs (keep first occurrence)
            seen_epochs = set()
            unique_combined = []
            for epoch, pos in combined:
                if epoch not in seen_epochs:
                    seen_epochs.add(epoch)
                    unique_combined.append((epoch, pos))
            
            merged_epochs = [x[0] for x in unique_combined]
            merged_pos = [x[1] for x in unique_combined]
            
            # Create merged ephemeris
            merged_eph = SatelliteEphemeris(
                epochs=np.array(merged_epochs),
                pos_m=np.array(merged_pos),
                source='sp3',
                vel_m_per_s=None
            )
            
            # Cache the merged ephemeris as a temporary file
            merged_path = sat_cache / f"merged_{sat_lower}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.sp3"
            # Store as pickle for fast reload
            import pickle
            with open(merged_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(merged_eph, f)
            
            logger.info(f"SP3 merged: {len(merged_epochs)} epochs from {datetime.fromtimestamp(merged_epochs[0], tz=timezone.utc).date()} to {datetime.fromtimestamp(merged_epochs[-1], tz=timezone.utc).date()}")
            return merged_path.with_suffix('.pkl')
    
    # Not in cache - try downloading from CDDIS
    # Try weeks from 14 days before to 7 days after start date
    for days_offset in range(-14, 8):
        check_date = start_date + timedelta(days=days_offset)
        # Find the Saturday that starts the week containing check_date
        days_since_saturday = (check_date.weekday() - 5) % 7
        saturday = check_date - timedelta(days=days_since_saturday)
        
        week_str = saturday.strftime("%y%m%d")
        remote_dir = f"{base}/{sat_lower}/{week_str}/"
        
        try:
            files = list_remote_files(session, remote_dir)
            sp3_files = [f for f in files if '.sp3' in f.lower()]
            if not sp3_files:
                continue
            
            ilrs_files = [f for f in sp3_files if 'ilrsa' in f.lower() or 'ilrsb' in f.lower()]
            chosen = sorted(ilrs_files)[-1] if ilrs_files else sorted(sp3_files)[-1]
            
            url = remote_dir.rstrip("/") + "/" + chosen
            dest = cache_dir / sat_lower / week_str / chosen
            
            try:
                downloaded_file = download_file(session, url, dest)
                
                test_eph = parse_sp3_positions(downloaded_file, sat_id=None)
                if len(test_eph.epochs) == 0:
                    continue
                
                file_start = datetime.fromtimestamp(test_eph.epochs[0], tz=timezone.utc)
                file_end = datetime.fromtimestamp(test_eph.epochs[-1], tz=timezone.utc)
                
                # Accept file if it overlaps with observation period
                if file_end >= start_date and file_start <= end_date:
                    logger.info(f"SP3 orbit found: {chosen} (covers {file_start.date()} to {file_end.date()})")
                    return downloaded_file
            except Exception as e:
                logger.debug(f"Failed to verify SP3 {chosen}: {type(e).__name__}: {e}")
                continue
            
        except Exception:
            continue
    
    logger.warning(f"No SP3 orbit found for {sat} covering {start_date.date()} to {end_date.date()}")
    return None


def select_cpf_file_for_satellite(session: requests.Session, sat: str, year: int, cache_dir: Path) -> Optional[Path]:
    """
    Find and download the best CPF file for a given satellite and year.
    Prioritizes archive/slr/cpf_predicts_v2/{year}/{sat}/ over current.
    NOTE: CPF files are PREDICTIONS with 100-1000 km errors. Use SP3 for precise analysis.
    """
    base = "https://cddis.nasa.gov/archive/slr/cpf_predicts_v2"

    candidates = [
        f"{base}/{year}/{sat}/",        # Standard archive: 2024/lageos1/
        f"{base}/{year}/{sat.upper()}/", # Capitalized variant
    ]

    for remote_dir in candidates:
        try:
            files = list_remote_files(session, remote_dir)
        except Exception:
            continue

        # Prefer files containing the year; otherwise just take the last file.
        cpf_files = [f for f in files if f.lower().endswith((".cpf", ".cpf.gz", ".txt")) or "cpf" in f.lower()]
        if not cpf_files:
            cpf_files = files

        cpf_files = sorted(set(cpf_files))
        if not cpf_files:
            continue

        chosen = cpf_files[-1]
        url = remote_dir.rstrip("/") + "/" + chosen
        dest = cache_dir / sat / str(year) / chosen
        try:
            return download_file(session, url, dest)
        except Exception:
            continue

    # Fallback to current
    try:
        remote_dir = f"{base}/current/"
        files = list_remote_files(session, remote_dir)
        # Choose a file that contains the satellite name if possible
        sat_lower = sat.lower()
        matches = [f for f in files if sat_lower in f.lower()]
        pool = sorted(matches) if matches else sorted(files)
        if not pool:
            return None
        chosen = pool[-1]
        url = remote_dir.rstrip("/") + "/" + chosen
        dest = cache_dir / "current" / chosen
        return download_file(session, url, dest)
    except Exception:
        return None


def _fetch_sp3_worker(args):
    """Worker function for parallel SP3 fetching."""
    session, sat, start_date, end_date, cache_dir = args
    try:
        path = select_sp3_file_for_satellite(session, sat, start_date, end_date, cache_dir)
        return (sat, path)
    except Exception as e:
        logger.error(f"SP3 worker failed for {sat}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return (sat, None)


def _fetch_cpf_worker(args):
    """Worker function for parallel CPF fetching."""
    session, sat, year, cache_dir = args
    try:
        path = select_cpf_file_for_satellite(session, sat, year, cache_dir)
        return (sat, year, path)
    except Exception:
        return (sat, year, None)


def atomic_write_json(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


# Satellites with known SP3 precise orbit availability from ILRS
SATELLITES_WITH_SP3 = {'lageos1', 'lageos2', 'lares', 'lares2', 'etalon1', 'etalon2'}


def main() -> int:
    parser = argparse.ArgumentParser(description="TEP-SLR Step 2.1: Compute SLR range residuals using SP3 precise orbits and SLRF2020 station coordinates")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-01-02")
    parser.add_argument("--max-np2-files", type=int, default=0)
    parser.add_argument("--min-elevation-deg", type=float, default=10.0)
    parser.add_argument("--satellites", default="lageos1,lageos2,etalon1,etalon2,lares",
                        help="Comma-separated list of satellites to process (default: lageos1,lageos2,etalon1,etalon2,lares). Use 'all' to try all satellites with known SP3 availability.")
    args = parser.parse_args()

    start = parse_yyyy_mm_dd(args.start)
    end = parse_yyyy_mm_dd(args.end)
    if end < start:
        logger.error("--end must be >= --start")
        return 2

    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)

    auth = get_auth()
    if not auth:
        logger.error("CDDIS/Earthdata auth not found. Configure ~/.netrc for urs.earthdata.nasa.gov or set CDDIS_USER/CDDIS_PASS.")
        return 2

    session = requests.Session()
    session.auth = auth
    session.headers.update({"User-Agent": "TEP-SLR/step_2_1_slr_residuals"})

    # Check both CRD v2 (2022+) and CRD v1 (pre-2022) directories
    np2_roots = [
        DATA_ROOT / "npt_crd_v2" / "allsat",
        DATA_ROOT / "npt_crd" / "allsat"
    ]
    
    np2_files = []
    for np2_root in np2_roots:
        if np2_root.exists():
            files = collect_np2_files(np2_root, start, end, max_files=args.max_np2_files)
            np2_files.extend(files)
            logger.info(f"Found {len(files)} files in {np2_root}")
    
    logger.info(f"NP2 files selected: {len(np2_files)}")

    if not np2_files:
        logger.error("No NP2 files found for date range")
        return 2

    # Download station SINEX (SLRF2020) - authoritative coords
    resource_dir = DATA_ROOT / "products" / "resource"
    resource_dir.mkdir(parents=True, exist_ok=True)

    resource_url = "https://cddis.nasa.gov/archive/slr/products/resource/"
    try:
        resource_files = list_remote_files(session, resource_url)
    except Exception as e:
        logger.error(f"Failed to list CDDIS resource directory: {type(e).__name__}: {e}")
        return 2

    slrf_name = select_slrf2020_sinex_filename(resource_files)
    if not slrf_name:
        logger.error("Could not find an SLRF2020 SINEX file in CDDIS resource directory.")
        logger.error(f"Checked {len(resource_files)} entries under {resource_url}")
        return 2

    slrf2020_url = resource_url.rstrip("/") + "/" + slrf_name
    slrf2020_local = resource_dir / slrf_name

    try:
        download_file(session, slrf2020_url, slrf2020_local)
    except Exception as e:
        logger.error(f"Failed to download SLRF2020 SINEX: {type(e).__name__}: {e}")
        logger.error(f"URL attempted: {slrf2020_url}")
        return 2

    stations = load_slrf2020_station_states(slrf2020_local)
    logger.info(f"Stations loaded from SLRF2020: {len(stations)}")

    # Download ILRS Data Handling File for station code mapping
    ilrs_dhf_name = "ILRS_Data_Handling_File_20250513.snx.gz"
    ilrs_dhf_url = resource_url.rstrip("/") + "/" + ilrs_dhf_name
    ilrs_dhf_local = resource_dir / ilrs_dhf_name
    
    try:
        download_file(session, ilrs_dhf_url, ilrs_dhf_local)
    except Exception as e:
        logger.warning(f"Failed to download ILRS DHF, will use SLRF2020 for mapping: {type(e).__name__}: {e}")
        ilrs_dhf_local = slrf2020_local
    
    # Build station code mapping (CRD alphanumeric -> SINEX numeric CDP codes)
    station_code_map = build_station_code_mapping(ilrs_dhf_local)
    logger.info(f"Station code mapping built: {len(station_code_map)} entries")

    # Parse CRD obs
    obs: List[CRDObs] = []
    for fp in np2_files:
        obs.extend(list(iter_crd_observations(fp)))

    if not obs:
        logger.error("No CRD '11' normal-point observations parsed")
        return 2

    df_obs = pd.DataFrame(
        [
            {
                "epoch_utc": o.epoch_utc,
                "station": o.station,
                "satellite": normalize_satellite_name(o.satellite),
                "obs_range_m": o.obs_range_m,
                "tof_s": o.tof_s,
                "elev_reported_deg": o.elev_deg_reported,
                "np2_file": o.file_name,
            }
            for o in obs
        ]
    )

    # Safety clamp: enforce the requested [start, end] window.
    # Some archives can contain occasional out-of-range epochs; if not clamped,
    # orbit fetching can expand far beyond the intended window.
    df_obs = df_obs[(df_obs["epoch_utc"] >= start_dt) & (df_obs["epoch_utc"] <= end_dt)].copy()
    if df_obs.empty:
        logger.error("No observations remain after clamping to requested [start, end] window")
        return 2

    # Apply satellite filter (default: lageos1,lageos2)
    if args.satellites.lower() != 'all':
        requested_sats = [s.strip().lower() for s in args.satellites.split(",")]
        initial_count = len(df_obs)
        df_obs = df_obs[df_obs["satellite"].str.lower().isin(requested_sats)]
        logger.info(f"Satellite filter applied: {requested_sats} -> {len(df_obs)}/{initial_count} observations retained")
        
        if df_obs.empty:
            logger.error(f"No observations found for requested satellites: {requested_sats}")
            return 2
    else:
        # When 'all' is specified, filter to only satellites with known SP3 availability
        initial_count = len(df_obs)
        df_obs = df_obs[df_obs["satellite"].str.lower().isin(SATELLITES_WITH_SP3)]
        logger.info(f"Processing all satellites with SP3: {sorted(SATELLITES_WITH_SP3)} -> {len(df_obs)}/{initial_count} observations retained")

    # Determine satellites needed
    sats = sorted(df_obs["satellite"].unique().tolist())
    
    # Determine observation date range for orbit fetching
    # Use the requested window (not min/max of parsed epochs) to avoid spurious out-of-range dates.
    obs_start = start_dt
    obs_end = end_dt

    sp3_cache = DATA_ROOT / "products" / "orbits" / "sp3"
    sp3_cache.mkdir(parents=True, exist_ok=True)

    # Load SP3 precise ephemerides per satellite
    eph_map: Dict[str, SatelliteEphemeris] = {}
    missing_sp3: List[str] = []

    # Parallelize SP3 fetching (listing + downloading)
    sp3_tasks = []
    for sat in sats:
        sp3_tasks.append((session, sat, obs_start, obs_end, sp3_cache))

    logger.info(f"Fetching SP3 precise orbit files for {len(sp3_tasks)} satellites in parallel...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_fetch_sp3_worker, task) for task in sp3_tasks]
        
        for future in as_completed(futures):
            sat, sp3_path = future.result()
            
            if sp3_path is None:
                missing_sp3.append(sat)
                logger.warning(f"SP3 orbit not found for {sat}")
                continue
                
            try:
                # Map satellite names to SP3 IDs
                sp3_id_map = {
                    'lageos1': 'L51',
                    'lageos2': 'L52',
                    'lares': 'L53',
                    'lares2': 'L54',
                    # Etalon SP3 IDs vary by provider; omit here and allow auto-detect.
                }
                sp3_id = sp3_id_map.get(sat.lower())
                
                # Check if this is a merged pickle file
                if sp3_path.suffix == '.pkl':
                    import pickle
                    with open(sp3_path, 'rb') as f:
                        eph_map[sat] = pickle.load(f)
                    logger.info(f"SP3 loaded (merged): {sat} (ID={sp3_id}) -> {sp3_path.name}")
                else:
                    eph_map[sat] = parse_sp3_positions(sp3_path, sat_id=sp3_id)
                    logger.info(f"SP3 loaded: {sat} (ID={sp3_id}) -> {sp3_path.name}")
            except Exception as e:
                logger.warning(f"SP3 parse failed for {sat} ({sp3_path}): {type(e).__name__}: {e}")

    if not eph_map:
        logger.error("No SP3 precise orbits could be loaded. Step 2.1 cannot proceed.")
        return 2

    rows: List[Dict[str, object]] = []
    missing_station = 0
    missing_ephem = 0
    dropped_low_elev = 0
    unmapped_stations = set()

    for r in df_obs.itertuples(index=False):
        epoch: datetime = r.epoch_utc
        sat = str(r.satellite)
        sta_crd = str(r.station)  # CRD alphanumeric code
        
        # Translate CRD station code to SINEX numeric CDP code
        sta_sinex = station_code_map.get(sta_crd.lower())
        if sta_sinex is None:
            # Try direct lookup (in case it's already numeric)
            sta_sinex = sta_crd
        
        st_state = stations.get(sta_sinex)
        if st_state is None:
            missing_station += 1
            unmapped_stations.add(sta_crd)
            continue

        eph = eph_map.get(sat)
        if eph is None:
            missing_ephem += 1
            continue

        sta_ecef_nominal = propagate_station(st_state, epoch)
        
        # Apply solid Earth tide displacement to station position
        tide_displacement = compute_solid_earth_tide_displacement(sta_ecef_nominal, epoch)
        
        # Apply ocean loading displacement
        ocean_displacement = compute_ocean_loading_displacement(sta_ecef_nominal, epoch)
        
        # Total station displacement
        sta_ecef = sta_ecef_nominal + tide_displacement + ocean_displacement
        tide_radial_m = float(np.dot(tide_displacement, sta_ecef_nominal / np.linalg.norm(sta_ecef_nominal)))
        ocean_radial_m = float(np.dot(ocean_displacement, sta_ecef_nominal / np.linalg.norm(sta_ecef_nominal)))
        
        # Light-time iteration for satellite position
        # CRD epoch_event=2 means epoch is TRANSMIT time (when laser left station).
        # Bounce time = transmit_time + one_way_light_time
        # Use observed range for initial estimate, then iterate.
        obs_range_m = float(r.obs_range_m)
        one_way_light_time = obs_range_m / C_M_S  # Initial estimate
        
        # Iterate to refine bounce time (usually converges in 2-3 iterations)
        for _ in range(3):
            bounce_time = epoch.timestamp() + one_way_light_time  # ADD for transmit epoch
            sat_ecef = interp_sat_pos(eph, bounce_time)
            model_range_m = float(np.linalg.norm(sat_ecef - sta_ecef))
            one_way_light_time = model_range_m / C_M_S
        
        elev_deg = compute_elevation_deg(sta_ecef, sat_ecef)
        if elev_deg < float(args.min_elevation_deg):
            dropped_low_elev += 1
            continue

        # Get station geodetic coordinates for tropospheric correction
        sta_lat, sta_lon, sta_h = ecef_to_lla(float(sta_ecef[0]), float(sta_ecef[1]), float(sta_ecef[2]))
        
        # Compute tropospheric delay (two-way returned, need one-way)
        tropo_delay_2way_m = compute_tropospheric_delay_m(elev_deg, sta_lat, sta_h)
        tropo_delay_1way_m = tropo_delay_2way_m / 2.0
        
        # Get center-of-mass correction for this satellite
        com_correction_m = LAGEOS_COM_CORRECTION_M.get(sat.lower(), 0.0)
        
        # Compute Shapiro relativistic delay (one-way)
        shapiro_delay_1way_m = compute_shapiro_delay_m(sta_ecef, sat_ecef)
        
        # Sagnac effect: Corrects for Earth rotation during signal flight time
        # Range in ECEF is smaller than inertial path. We must subtract Sagnac from obs (or add to model).
        # Here we subtract from obs to match geometric ECEF model.
        sagnac_delay_m = compute_sagnac_effect_m(sta_ecef, sat_ecef)
        
        # Corrected observed range: subtract tropo delay, subtract Shapiro, add CoM offset, subtract Sagnac
        # (CoM is added because obs range is to retroreflector, model is to CoM)
        # (Shapiro/Sagnac are subtracted because they are extra delays in the observed TOF)
        obs_range_corrected_m = float(r.obs_range_m) - tropo_delay_1way_m - shapiro_delay_1way_m - sagnac_delay_m + com_correction_m
        
        # Residual with corrections applied
        resid_m = obs_range_corrected_m - model_range_m

        rows.append(
            {
                "epoch_utc": epoch.isoformat().replace("+00:00", "Z"),
                "station": sta_sinex,
                "satellite": sat,
                "obs_range_m": float(r.obs_range_m),
                "obs_range_corrected_m": obs_range_corrected_m,
                "model_range_m": model_range_m,
                "tropo_delay_m": tropo_delay_1way_m,
                "com_correction_m": com_correction_m,
                "tide_radial_m": tide_radial_m,
                "ocean_radial_m": ocean_radial_m,
                "shapiro_delay_m": shapiro_delay_1way_m,
                "sagnac_delay_m": sagnac_delay_m,
                "residual_m": resid_m,
                "residual_mm": resid_m * 1000.0,
                "elevation_deg": elev_deg,
                "elev_reported_deg": r.elev_reported_deg,
                "np2_file": r.np2_file,
            }
        )

    df = pd.DataFrame(rows)

    out_csv = OUTPUTS_DIR / "step_2_1_slr_residuals.csv"
    out_json = OUTPUTS_DIR / "step_2_1_slr_residuals_summary.json"

    if df.empty:
        logger.error("No residuals computed (empty dataset after matching station+SP3 and elevation filtering).")
        logger.error(f"Missing station coords for {missing_station} obs; missing ephemeris for {missing_ephem} obs; dropped low elev {dropped_low_elev} obs")
        if unmapped_stations:
            logger.error(f"Unmapped CRD station codes: {sorted(unmapped_stations)}")
        return 2

    df.to_csv(out_csv, index=False)

    # Filter for valid residuals for statistics (outlier rejection 0.5m = 500mm)
    # This avoids skewing mean/std with gross errors
    df_valid = df[df["residual_mm"].abs() < 500.0]
    
    if df_valid.empty:
        logger.warning("No residuals remaining after 0.5m outlier rejection!")
        df_valid = df # Fallback

    summary = {
        "spec": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "min_elevation_deg": float(args.min_elevation_deg),
            "station_source": str(slrf2020_local),
            "orbit_source": "https://cddis.nasa.gov/archive/slr/products/orbits/ (SP3 precise)",
        },
        "counts": {
            "input_np2_files": len(np2_files),
            "input_obs": int(df_obs.shape[0]),
            "output_residuals": int(df.shape[0]),
            "valid_residuals_5m": int(df_valid.shape[0]),
            "stations": int(df["station"].nunique()),
            "satellites": int(df["satellite"].nunique()),
            "missing_station": int(missing_station),
            "missing_ephemeris": int(missing_ephem),
            "dropped_low_elev": int(dropped_low_elev),
            "missing_sp3_satellites": missing_sp3,
        },
        "residual_mm": {
            "mean": float(df_valid["residual_mm"].mean()),
            "std": float(df_valid["residual_mm"].std(ddof=1)) if df_valid.shape[0] > 1 else 0.0,
            "p05": float(df_valid["residual_mm"].quantile(0.05)),
            "p50": float(df_valid["residual_mm"].quantile(0.50)),
            "p95": float(df_valid["residual_mm"].quantile(0.95)),
        },
        "by_station_rms_mm": (
            df_valid.groupby("station")["residual_mm"].apply(lambda s: float(np.sqrt(np.mean(np.square(s.values))))).sort_values(ascending=False).head(25).to_dict()
        ),
        "by_satellite_rms_mm": (
            df_valid.groupby("satellite")["residual_mm"].apply(lambda s: float(np.sqrt(np.mean(np.square(s.values))))).sort_values(ascending=False).head(25).to_dict()
        ),
    }

    atomic_write_json(summary, out_json)

    try:
        apply_paper_style()
        plt.figure(figsize=(8, 4))
        plt.hist(df["residual_mm"].values, bins=120, color="#4A90C2", alpha=0.75)
        plt.xlabel("Residual (mm) = observed range − SP3 modeled geometric range")
        plt.ylabel("Count")
        plt.title("SLR residual distribution")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "step_2_1_slr_residual_hist.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.scatter(df["elevation_deg"].values, df["residual_mm"].values, s=2, alpha=0.25, color="#2D0140")
        plt.xlabel("Elevation (deg)")
        plt.ylabel("Residual (mm)")
        plt.title("SLR residual vs elevation")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "step_2_1_slr_residual_vs_elevation.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Plotting failed: {type(e).__name__}: {e}")

    logger.success(f"Step 2.1 complete. Wrote: {out_csv} and {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
