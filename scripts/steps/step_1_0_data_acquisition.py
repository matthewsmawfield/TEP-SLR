#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

# Force unbuffered output for real-time logging
os.environ["PYTHONUNBUFFERED"] = "1"

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import TEPLogger, set_step_logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"

for d in [DATA_DIR, LOGS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger = TEPLogger("step_1_0", log_file_path=LOGS_DIR / "step_1_0_data_acquisition.log")
set_step_logger(logger)


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


def daterange_years(start: date, end: date) -> List[int]:
    years = set()
    d = start
    while d <= end:
        years.add(d.year)
        d = d + timedelta(days=1)
    return sorted(years)


_DATE_PATTERNS = [
    re.compile(r"(?P<ymd>\d{8})"),
    re.compile(r"(?P<ydoy>\d{4}\d{3})"),
    re.compile(r"(?P<yymd>\d{6})"),
    re.compile(r"(?P<yydoy>\d{2}\d{3})"),
]


def _parse_date_token(token: str) -> Optional[date]:
    try:
        if len(token) == 8:
            return datetime.strptime(token, "%Y%m%d").date()
        if len(token) == 7:
            y = int(token[:4])
            doy = int(token[4:])
            return date(y, 1, 1) + timedelta(days=doy - 1)
        if len(token) == 6:
            yy = int(token[:2])
            y = 2000 + yy if yy <= 79 else 1900 + yy
            return datetime.strptime(f"{y}{token[2:]}", "%Y%m%d").date()
        if len(token) == 5:
            yy = int(token[:2])
            y = 2000 + yy if yy <= 79 else 1900 + yy
            doy = int(token[2:])
            return date(y, 1, 1) + timedelta(days=doy - 1)
    except Exception:
        return None
    return None


def infer_date_from_filename(filename: str) -> Optional[date]:
    for pat in _DATE_PATTERNS:
        m = pat.search(filename)
        if not m:
            continue
        token = next((v for v in m.groupdict().values() if v), None)
        if not token:
            continue
        d = _parse_date_token(token)
        if d is not None:
            return d
    return None


def _extract_filename_from_list_line(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None
    if s.lower().startswith("total"):
        return None

    parts = s.split()
    if not parts:
        return None

    # Two common formats observed on CDDIS:
    # 1) "filename.ext 742153"
    # 2) "-rw-r--r-- 1 user group 742153 Jan 01 00:00 filename.ext"
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
    ]

    last_err = None
    for url in candidates:
        try:
            resp = session.get(url, timeout=timeout_s)
            if resp.status_code == 200:
                files: List[str] = []
                for ln in resp.text.splitlines():
                    name = _extract_filename_from_list_line(ln)
                    if name:
                        files.append(name)
                return files
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"

    raise RuntimeError(f"Failed to list remote directory: {dir_url} ({last_err})")


def atomic_write_json(obj: object, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)


def download_file(session: requests.Session, url: str, dest_path: Path, timeout_s: int = 600, retries: int = 3) -> Tuple[bool, int]:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return False, dest_path.stat().st_size

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
            return True, n
        except Exception as e:
            last_exc = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(min(30, 2**attempt))

    raise RuntimeError(f"Failed to download after {retries} attempts: {url} ({type(last_exc).__name__}: {last_exc})")


def download_worker(args):
    """Worker function for parallel downloads."""
    session, url, dest, timeout_s, retries = args
    try:
        did_download, n_bytes = download_file(session, url, dest, timeout_s, retries)
        return {"success": True, "did_download": did_download, "n_bytes": n_bytes, "dest": dest, "url": url}
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {e}", "dest": dest, "url": url}


@dataclass(frozen=True)
class ProductSpec:
    data_type: str
    crd_version: str
    source: str

    @property
    def remote_subdir(self) -> str:
        if self.data_type == "npt":
            return "npt_crd_v2" if self.crd_version == "crd_v2" else "npt_crd"
        if self.data_type == "fr":
            return "fr_crd_v2" if self.crd_version == "crd_v2" else "fr_crd"
        raise ValueError(f"Unsupported data_type: {self.data_type}")


def build_remote_year_dir(spec: ProductSpec, year: int, satellite: Optional[str]) -> str:
    root = f"https://cddis.nasa.gov/archive/slr/data/{spec.remote_subdir}/"

    if spec.source == "allsat":
        return f"{root}allsat/{year}/"

    if spec.source == "satellite":
        if not satellite:
            raise ValueError("--satellite is required when --source satellite")
        return f"{root}{satellite}/{year}/"

    raise ValueError(f"Unsupported source: {spec.source}")


def build_local_base(spec: ProductSpec) -> Path:
    return DATA_DIR / "slr" / spec.remote_subdir


def main() -> int:
    parser = argparse.ArgumentParser(description="TEP-SLR Step 1.0: Download SLR data from CDDIS (Earthdata auth required)")
    parser.add_argument("--data-type", choices=["npt", "fr"], default="npt")
    parser.add_argument("--crd-version", choices=["crd", "crd_v2"], default="crd_v2")
    parser.add_argument("--source", choices=["allsat", "satellite"], default="allsat")
    parser.add_argument("--satellite", default=None)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel download workers")
    args = parser.parse_args()

    start = parse_yyyy_mm_dd(args.start)
    end = parse_yyyy_mm_dd(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    auth = get_auth()
    if not auth:
        logger.error("CDDIS/Earthdata auth not found. Configure ~/.netrc for urs.earthdata.nasa.gov or set CDDIS_USER/CDDIS_PASS.")
        return 2

    spec = ProductSpec(data_type=args.data_type, crd_version=args.crd_version, source=args.source)

    session = requests.Session()
    session.auth = auth
    session.headers.update({"User-Agent": "TEP-SLR/step_1_0_data_acquisition"})

    years = daterange_years(start, end)
    local_base = build_local_base(spec)

    logger.info(f"SLR acquisition: data_type={args.data_type}, crd_version={args.crd_version}, source={args.source}, satellite={args.satellite}")
    logger.info(f"Date range: {start.isoformat()} to {end.isoformat()} ({len(years)} years)")

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    downloaded: List[str] = []
    skipped: List[str] = []
    failed: List[Dict[str, str]] = []
    total_bytes = 0

    n_selected_total = 0

    for year in years:
        remote_dir = build_remote_year_dir(spec, year, args.satellite)
        logger.process(f"Listing: {remote_dir}")

        try:
            files = list_remote_files(session, remote_dir)
        except Exception as e:
            failed.append({"url": remote_dir, "error": f"{type(e).__name__}: {e}"})
            logger.error(f"Failed to list {remote_dir}: {type(e).__name__}: {e}")
            continue

        selected: List[str] = []
        for fn in files:
            d = infer_date_from_filename(fn)
            if d is None:
                continue
            if start <= d <= end:
                selected.append(fn)

        selected = sorted(set(selected))
        n_selected_total += len(selected)
        logger.info(f"{year}: {len(selected)} files selected (out of {len(files)} listed)")

        if args.max_files and len(downloaded) + len(skipped) >= args.max_files:
            break

        # Prepare download tasks
        tasks = []
        for fn in selected:
            if args.max_files and (len(downloaded) + len(skipped)) >= args.max_files:
                break

            url = remote_dir.rstrip("/") + "/" + fn
            dest = local_base / args.source / str(year) / fn

            if args.dry_run:
                skipped.append(str(dest))
                continue

            tasks.append((session, url, dest, 600, 3))

        if not tasks:
            continue

        # Parallel download with progress bar
        logger.info(f"Downloading {len(tasks)} files with {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_worker, task): task for task in tasks}
            
            with tqdm(total=len(tasks), desc=f"Year {year}", unit="file", disable=args.dry_run) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    
                    if result["success"]:
                        total_bytes += result["n_bytes"]
                        if result["did_download"]:
                            downloaded.append(str(result["dest"]))
                        else:
                            skipped.append(str(result["dest"]))
                    else:
                        failed.append({"url": result["url"], "error": result["error"]})
                        logger.error(f"Failed: {result['url']} ({result['error']})")
                    
                    pbar.update(1)

    finished_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    output = {
        "started_at": started_at,
        "finished_at": finished_at,
        "spec": {
            "data_type": spec.data_type,
            "crd_version": spec.crd_version,
            "source": spec.source,
            "satellite": args.satellite,
        },
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "summary": {
            "selected_remote_files": n_selected_total,
            "downloaded_files": len(downloaded),
            "skipped_files": len(skipped),
            "failed": len(failed),
            "total_bytes": total_bytes,
        },
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }

    out_path = OUTPUTS_DIR / "step_1_0_data_acquisition.json"
    atomic_write_json(output, out_path)

    if failed:
        logger.warning(f"Completed with failures ({len(failed)}). Output: {out_path}")
        return 1

    logger.success(f"Data acquisition complete. Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
