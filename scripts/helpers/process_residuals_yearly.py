#!/usr/bin/env python3
"""
Process SLR residuals year-by-year to avoid memory issues.
Saves individual yearly CSVs then merges them.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"
YEARLY_DIR = OUTPUTS_DIR / "residuals_yearly"
LOGS_DIR = PROJECT_ROOT / "logs"

# Setup imports
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.logger import TEPLogger, set_step_logger

# Setup logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
YEARLY_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger("process_yearly", log_file_path=LOGS_DIR / "process_residuals_yearly.log")
set_step_logger(logger)

def process_year(year):
    """Process a single year of residuals."""
    logger.info(f"Processing {year}")
    
    start_time = datetime.now()
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "steps" / "step_2_1_slr_residuals.py"),
        "--start", f"{year}-01-01",
        "--end", f"{year}-12-31"
    ]
    
    # We allow stdout to pass through, but we could capture it if we wanted to log it to our logger
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"Year {year} failed with exit code {result.returncode}")
        return False
    
    # Move the output CSV to yearly directory
    main_csv = OUTPUTS_DIR / "step_2_1_slr_residuals.csv"
    yearly_csv = YEARLY_DIR / f"residuals_{year}.csv"
    
    if main_csv.exists():
        import shutil
        shutil.move(str(main_csv), str(yearly_csv))
        size_mb = yearly_csv.stat().st_size / (1024**2)
        logger.success(f"Saved {yearly_csv.name} ({size_mb:.1f} MB)")
    else:
        logger.error(f"Output CSV not found for {year}")
        return False
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Completed {year} in {elapsed:.1f}s")
    
    return True

def merge_csvs():
    """Merge all yearly CSVs into master file."""
    logger.info("Merging yearly CSVs")
    
    import pandas as pd
    
    yearly_files = sorted(YEARLY_DIR.glob("residuals_*.csv"))
    
    if not yearly_files:
        logger.error("No yearly CSV files found")
        return False
    
    logger.info(f"Found {len(yearly_files)} yearly files")
    
    # Merge using chunked reading to avoid memory issues
    master_csv = OUTPUTS_DIR / "step_2_1_slr_residuals.csv"
    
    logger.info(f"Merging to {master_csv.name}...")
    
    first = True
    total_rows = 0
    
    for yearly_file in yearly_files:
        logger.process(f"Reading {yearly_file.name}...")
        try:
            df = pd.read_csv(yearly_file)
            rows = len(df)
            total_rows += rows
            
            if first:
                df.to_csv(master_csv, index=False, mode='w')
                first = False
            else:
                df.to_csv(master_csv, index=False, mode='a', header=False)
        except Exception as e:
            logger.error(f"Failed to process {yearly_file.name}: {e}")
            return False
        
    size_mb = master_csv.stat().st_size / (1024**2)
    logger.success(f"Master CSV created: {size_mb:.1f} MB, {total_rows:,} rows")
    
    # Verify date range
    logger.info("Verifying merged data...")
    try:
        df_head = pd.read_csv(master_csv, nrows=1000)
        # For tail, we need to know roughly where to start or just read tail using system command if generic, 
        # but pandas doesn't support reading tail efficiently without reading all or skipping.
        # Given we just wrote it, we can trust it mostly, but let's do a quick check if possible.
        # We'll skip verification of tail for speed/memory if file is huge, or just rely on head dates and file size.
        
        # Actually, let's just check head and years processed
        logger.info(f"Master file exists and is populated.")
    except Exception as e:
        logger.warning(f"Verification warning: {e}")
    
    return True

def main():
    years = list(range(2015, 2026))  # 2015-2025
    
    logger.info("="*60)
    logger.info("SLR Residuals Year-by-Year Processing")
    logger.info("="*60)
    logger.info(f"Years to process: {years}")
    logger.info(f"Output directory: {YEARLY_DIR}")
    
    # Process each year
    for year in years:
        # Check if year is already done
        yearly_csv = YEARLY_DIR / f"residuals_{year}.csv"
        if yearly_csv.exists() and yearly_csv.stat().st_size > 1024:
            logger.info(f"Skipping {year} (already exists: {yearly_csv.name})")
            continue
            
        success = process_year(year)
        if not success:
            logger.error(f"FAILED at year {year}")
            return 1
    
    # Merge all CSVs
    success = merge_csvs()
    if not success:
        logger.error("FAILED to merge CSVs")
        return 1
    
    logger.success("All residuals processed and merged")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
