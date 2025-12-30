#!/usr/bin/env python3

import argparse
import os
import re
import sys
import time
import requests
from datetime import date, datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.logger import TEPLogger, set_step_logger

# CDDIS Configuration
CDDIS_ORBIT_BASE = "https://cddis.nasa.gov/archive/slr/products/orbits"
SATELLITES = ["lageos1", "lageos2"]

# Setup Logging
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger("download_orbits", log_file_path=LOGS_DIR / "download_orbits.log")
set_step_logger(logger)

def get_auth():
    import netrc
    try:
        auth = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        if auth:
            return (auth[0], auth[2])
    except:
        pass
    return (os.getenv("CDDIS_USER"), os.getenv("CDDIS_PASS"))

def list_remote_files(session, url):
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code != 200:
            logger.warning(f"HTTP {resp.status_code} for {url}")
            return []
        
        files = []
        # Handle CDDIS listing
        for m in re.finditer(r'href="([^"]+)"', resp.text):
            name = m.group(1).rstrip('/').strip()
            if name not in ['../', './'] and not name.startswith('?') and not name.startswith('/'):
                files.append(name)
        return files
    except Exception as e:
        logger.error(f"Error listing {url}: {e}")
        return []

def download_file(session, url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return True # Already exists
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Retry logic
        for attempt in range(3):
            try:
                with session.get(url, stream=True, timeout=60) as r:
                    if r.status_code == 200:
                        with open(dest, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        return True
                    else:
                        if attempt == 2:
                            logger.warning(f"Failed to download {url}: HTTP {r.status_code}")
            except Exception as e:
                if attempt == 2:
                    raise e
                time.sleep(1)
                
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        if dest.exists():
            dest.unlink()
    return False

def get_saturdays(year):
    """Get all Saturdays for a given year."""
    d = date(year, 1, 1)
    # Find first Saturday
    d += timedelta(days=(5 - d.weekday() + 7) % 7)
    
    saturdays = []
    while d.year == year:
        saturdays.append(d)
        d += timedelta(days=7)
    return saturdays

def process_satellite(session, sat, year):
    base_url = f"{CDDIS_ORBIT_BASE}/{sat}"
    local_base = PROJECT_ROOT / "data/slr/products/orbits/sp3" / sat
    
    # Generate target weeks (YYMMDD format)
    weeks = get_saturdays(year)
    target_dirs = [d.strftime("%y%m%d") for d in weeks]
    
    logger.info(f"Targeting {len(target_dirs)} weeks for {sat} {year}")
    
    count = 0
    downloaded_new = 0
    
    # Process each week directly
    for d in target_dirs:
        week_url = f"{base_url}/{d}/"
        files = list_remote_files(session, week_url)
        
        if not files:
            # logger.debug(f"  No files found for {d}")
            continue

        # Find best SP3 file using strict priority
        sp3s = [f for f in files if f.endswith(".sp3.gz") or f.endswith(".sp3")]
        if not sp3s:
            continue
            
        # Priority list: Combined (ILRS) > ASI > GFZ > DGFI > others
        priority_order = ['ilrsb', 'ilrsa', 'asi', 'gfz', 'dgfi', 'nsgf', 'jcet', 'esa']
        
        chosen = None
        for center in priority_order:
            candidates = [f for f in sp3s if center in f.lower()]
            if candidates:
                # If multiple versions, pick the latest version (vXX)
                chosen = sorted(candidates)[-1]
                break
        
        if not chosen:
            # Fallback to whatever is there (sorted to be deterministic)
            chosen = sorted(sp3s)[-1]
        
        dest = local_base / d / chosen
        url = week_url + chosen
        
        if dest.exists():
            count += 1
            continue

        if download_file(session, url, dest):
            count += 1
            downloaded_new += 1
            logger.process(f"[{count}/{len(target_dirs)}] Downloaded {sat}/{d}/{chosen}")
            
    logger.success(f"Processed {count} weeks for {sat} ({downloaded_new} new)")
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    
    auth = get_auth()
    if not auth:
        logger.error("Error: No authentication found (netrc or env vars)")
        sys.exit(1)
        
    session = requests.Session()
    session.auth = auth
    
    for sat in SATELLITES:
        logger.info(f"Processing {sat} for {args.year}...")
        process_satellite(session, sat, args.year)

if __name__ == "__main__":
    main()
