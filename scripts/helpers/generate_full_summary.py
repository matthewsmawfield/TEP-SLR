#!/usr/bin/env python3
"""
Generate a comprehensive summary for the full merged SLR dataset.
Corrects the issue where the summary JSON only reflects the last processed year.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from utils.logger import TEPLogger, set_step_logger
from utils.plot_style import apply_paper_style

OUTPUTS_DIR = PROJECT_ROOT / "results" / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
CSV_PATH = OUTPUTS_DIR / "step_2_1_slr_residuals.csv"
JSON_PATH = OUTPUTS_DIR / "step_2_1_slr_residuals_summary.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger = TEPLogger("generate_summary", log_file_path=LOGS_DIR / "generate_full_summary.log")
set_step_logger(logger)

def main():
    logger.info(f"Reading merged dataset: {CSV_PATH}")
    if not CSV_PATH.exists():
        logger.error("Error: Merged CSV not found.")
        return 1

    # Load dataframe (optimize types for memory)
    df = pd.read_csv(CSV_PATH, dtype={
        'station': str,
        'satellite': str,
        'residual_mm': float,
        'elevation_deg': float
    })
    
    # Explicitly convert dates using ISO8601 format
    df['epoch_utc'] = pd.to_datetime(df['epoch_utc'], format='ISO8601')
    
    total_rows = len(df)
    logger.info(f"Loaded {total_rows:,} observations.")

    # Filter outliers for statistics (consistent with MWPC step)
    # Using strict 5m (5000mm) threshold as per TEP methodology
    mask_valid = np.abs(df['residual_mm']) < 5000
    df_clean = df[mask_valid]
    
    outliers = total_rows - len(df_clean)
    logger.info(f"Excluded {outliers:,} outliers (> 5m). Valid stats based on {len(df_clean):,} obs.")

    # Compute statistics
    stats = {
        "spec": {
            "start": df['epoch_utc'].min().isoformat(),
            "end": df['epoch_utc'].max().isoformat(),
            "total_years": int(df['epoch_utc'].dt.year.max() - df['epoch_utc'].dt.year.min() + 1),
            "orbit_source": "SP3 (ILRS/ASI/GFZ) - Precise",
            "station_source": "SLRF2020"
        },
        "counts": {
            "total_observations": total_rows,
            "valid_observations": len(df_clean),
            "outliers_excluded": outliers,
            "outlier_percentage": round(outliers / total_rows * 100, 2),
            "stations": int(df['station'].nunique()),
            "satellites": int(df['satellite'].nunique())
        },
        "residual_stats_mm": {
            "mean": float(df_clean['residual_mm'].mean()),
            "std": float(df_clean['residual_mm'].std()),
            "rms": float(np.sqrt(np.mean(np.square(df_clean['residual_mm'])))),
            "min": float(df_clean['residual_mm'].min()),
            "max": float(df_clean['residual_mm'].max()),
            "p05": float(df_clean['residual_mm'].quantile(0.05)),
            "p50": float(df_clean['residual_mm'].quantile(0.50)),
            "p95": float(df_clean['residual_mm'].quantile(0.95))
        },
        "station_stats": [],
        "satellite_stats": []
    }

    # Per-station stats (top 50 by count)
    logger.process("Computing station statistics...")
    station_groups = df_clean.groupby('station')
    for station, group in station_groups:
        stats["station_stats"].append({
            "station": station,
            "count": int(len(group)),
            "mean_mm": float(group['residual_mm'].mean()),
            "rms_mm": float(np.sqrt(np.mean(np.square(group['residual_mm']))))
        })
    stats["station_stats"].sort(key=lambda x: x["count"], reverse=True)

    # Per-satellite stats
    logger.process("Computing satellite statistics...")
    sat_groups = df_clean.groupby('satellite')
    for sat, group in sat_groups:
        stats["satellite_stats"].append({
            "satellite": sat,
            "count": int(len(group)),
            "mean_mm": float(group['residual_mm'].mean()),
            "rms_mm": float(np.sqrt(np.mean(np.square(group['residual_mm']))))
        })

    # Save updated JSON
    logger.info(f"Saving summary to {JSON_PATH}")
    with open(JSON_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    # Generate Diagnostic Plots
    logger.process("Generating diagnostic plots for full dataset...")
    FIGURES_DIR = OUTPUTS_DIR.parent / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    import matplotlib.pyplot as plt
    apply_paper_style()
    
    # 1. Residual Histogram
    try:
        plt.figure(figsize=(10, 6))
        # Clip for visualization to +/- 2000mm (2m) to see the core distribution
        data_vis = df_clean['residual_mm'][np.abs(df_clean['residual_mm']) < 2000]
        plt.hist(data_vis, bins=100, color="#4A90C2", alpha=0.75)
        plt.xlabel("Residual (mm)")
        plt.ylabel("Count")
        plt.title(fr"SLR Residual Distribution (2015-2025)\nN={len(data_vis):,} (Clipped to $\pm$2m)")
        plt.grid(True, alpha=0.3)
        plt.savefig(FIGURES_DIR / "slr_residual_histogram_full.png", dpi=300)
        logger.success(f"Saved {FIGURES_DIR / 'slr_residual_histogram_full.png'}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot histogram: {e}")

    # 2. Residual vs Elevation
    try:
        plt.figure(figsize=(10, 6))
        # Downsample for scatter plot if huge
        if len(df_clean) > 50000:
            df_plot = df_clean.sample(50000, random_state=42)
        else:
            df_plot = df_clean
            
        plt.scatter(df_plot['elevation_deg'], df_plot['residual_mm'], 
                   alpha=0.1, s=2, color="#2D0140")
        plt.ylim(-1000, 1000) # Zoom in to +/- 1m
        plt.xlabel("Elevation (deg)")
        plt.ylabel("Residual (mm)")
        plt.title("SLR Residuals vs Elevation (2015-2025)\n(Sampled 50k points, Zoomed $\pm$1m)")
        plt.grid(True, alpha=0.3)
        plt.savefig(FIGURES_DIR / "slr_residual_vs_elevation_full.png", dpi=300)
        logger.success(f"Saved {FIGURES_DIR / 'slr_residual_vs_elevation_full.png'}")
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot elevation scatter: {e}")

    logger.success("Summary generation complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
