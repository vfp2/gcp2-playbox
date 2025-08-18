#!/usr/bin/env python3
import os
import argparse
import logging
import time
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BAD_DATES = {
    "2023-06-01","2023-06-02","2023-06-03","2023-06-04","2023-06-05","2023-06-06","2023-06-07","2023-06-08","2023-06-09",
    "2023-09-28","2023-09-29","2023-11-27",
    "2024-02-06","2024-02-27","2024-02-28","2024-02-29","2024-03-01","2024-03-02","2024-03-03","2024-03-04","2024-03-05","2024-03-06","2024-03-07","2024-03-08","2024-03-09","2024-03-10","2024-03-11","2024-03-12","2024-03-13"
}


def read_gcp2_csv(path: str) -> pd.DataFrame:
    """Read GCP2 CSV file with progress logging."""
    logger.info(f"Reading GCP2 CSV from: {path}")
    
    # Expect header row, comma-delimited
    dtype = {
        "id": "int64",
        "time_epoch": "int64",
        "group_id": "int64",
        "netvar_calculation_id": "int64",
        "netvar_count_ff": "float64",
        "netvar_count_xor": "float64",
        "netvar_count_xor_alt": "float64",
        "netvar_count_xor_gt_medians": "float64",
        "reporting_devices": "int64",
        "aggregate_start_epoch": "int64",
        "aggregate_end_epoch": "int64",
    }
    
    csv_path = path
    if os.path.isdir(path):
        # pick first *.csv in directory if a directory is provided
        logger.info(f"Directory provided, searching for CSV files in: {path}")
        candidates = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')]
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        # choose the largest file as default
        csv_path = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)[0]
        logger.info(f"Selected largest CSV file: {os.path.basename(csv_path)} ({os.path.getsize(csv_path) / (1024*1024):.2f} MB)")
    
    logger.info(f"Loading CSV data...")
    start_time = time.time()
    df = pd.read_csv(csv_path, dtype=dtype)
    load_time = time.time() - start_time
    
    logger.info(f"CSV loaded successfully: {len(df):,} rows, {len(df.columns)} columns in {load_time:.2f}s")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    return df


def clean_and_partition(df: pd.DataFrame, out_dir: str) -> None:
    """Clean data and partition into parquet files with progress logging."""
    logger.info("Starting data cleaning and partitioning...")
    start_time = time.time()
    
    # Drop duplicates and any gaps are ignored (simply skipped)
    initial_rows = len(df)
    logger.info(f"Removing duplicates from {initial_rows:,} rows...")
    df = df.drop_duplicates(subset=["time_epoch", "group_id"]).copy()
    rows_after_dedup = len(df)
    logger.info(f"Removed {initial_rows - rows_after_dedup:,} duplicate rows, {rows_after_dedup:,} remaining")
    
    # Create date column for partitioning but don't add it to keep list
    df["date"] = pd.to_datetime(df["time_epoch"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    
    # Filter bad dates
    logger.info("Filtering out bad dates...")
    rows_before_filter = len(df)
    df = df[~df["date"].isin(BAD_DATES)]
    rows_after_filter = len(df)
    logger.info(f"Filtered out {rows_before_filter - rows_after_filter:,} rows with bad dates, {rows_after_filter:,} remaining")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Keep only required columns for backtests
    keep = [
        "time_epoch",
        "group_id",
        "netvar_count_xor_alt",
        "reporting_devices",
    ]
    logger.info(f"Kept {len(keep)} columns: {', '.join(keep)}")

    # Partition by date then group_id; no compression
    logger.info("Starting partitioning by date and group_id...")
    partitions = list(df.groupby(["date", "group_id"], sort=True))
    total_partitions = len(partitions)
    logger.info(f"Creating {total_partitions:,} partitions...")
    
    processed_partitions = 0
    total_rows_written = 0
    
    for i, ((date_str, group_id), part) in enumerate(partitions, 1):
        partition_start = time.time()
        
        target = os.path.join(out_dir, f"date={date_str}", f"group_id={group_id}")
        os.makedirs(target, exist_ok=True)
        
        # Drop the date column before saving, keeping only the columns we want
        part_out = part[keep].sort_values("time_epoch")
        output_file = os.path.join(target, "gcp2.parquet")
        
        part_out.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
        
        partition_time = time.time() - partition_start
        processed_partitions += 1
        total_rows_written += len(part_out)
        
        # Log progress every 100 partitions or for the last partition
        if i % 100 == 0 or i == total_partitions:
            progress_pct = (i / total_partitions) * 100
            logger.info(f"Progress: {i:,}/{total_partitions:,} partitions ({progress_pct:.1f}%) - "
                       f"Partition {date_str}/group_{group_id}: {len(part_out):,} rows in {partition_time:.3f}s")
    
    total_time = time.time() - start_time
    logger.info(f"Partitioning completed in {total_time:.2f}s")
    logger.info(f"Total partitions created: {total_partitions:,}")
    logger.info(f"Total rows written: {total_rows_written:,}")
    logger.info(f"Average time per partition: {total_time/total_partitions:.3f}s")


def main():
    """Main function with comprehensive logging."""
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Ingest GCP2 CSV to Parquet for Nautilus")
    parser.add_argument("--csv", required=True, help="Absolute path to GCP2 CSV")
    parser.add_argument("--out", default="experiments/7-financial-backtesting/parquet_out/gcp2")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting GCP2 CSV ingestion process")
    logger.info("=" * 60)
    logger.info(f"Input CSV: {args.csv}")
    logger.info(f"Output directory: {args.out}")
    
    try:
        start_time = time.time()
        
        df = read_gcp2_csv(args.csv)
        if df.empty:
            logger.warning("No rows found in CSV")
            return
        
        clean_and_partition(df, args.out)
        
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"GCP2 CSV ingestion completed successfully in {total_time:.2f}s")
        logger.info(f"Output written to: {args.out}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()


