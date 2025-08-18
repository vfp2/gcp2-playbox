#!/usr/bin/env python3
import os
import argparse
from datetime import datetime, timezone
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv


BAD_DATES = {
    "2023-06-01","2023-06-02","2023-06-03","2023-06-04","2023-06-05","2023-06-06","2023-06-07","2023-06-08","2023-06-09",
    "2023-09-28","2023-09-29","2023-11-27",
    "2024-02-06","2024-02-27","2024-02-28","2024-02-29","2024-03-01","2024-03-02","2024-03-03","2024-03-04","2024-03-05","2024-03-06","2024-03-07","2024-03-08","2024-03-09","2024-03-10","2024-03-11","2024-03-12","2024-03-13"
}


def read_gcp2_csv(path: str) -> pd.DataFrame:
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
        candidates = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.csv')]
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        # choose the largest file as default
        csv_path = sorted(candidates, key=lambda p: os.path.getsize(p), reverse=True)[0]
    df = pd.read_csv(csv_path, dtype=dtype)
    return df


def clean_and_partition(df: pd.DataFrame, out_dir: str) -> None:
    # Drop duplicates and any gaps are ignored (simply skipped)
    df = df.drop_duplicates(subset=["time_epoch", "group_id"]).copy()
    # Filter bad dates
    df["date"] = pd.to_datetime(df["time_epoch"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    df = df[~df["date"].isin(BAD_DATES)]

    # Keep only required columns for backtests
    keep = [
        "time_epoch",
        "group_id",
        "netvar_count_xor_alt",
        "reporting_devices",
    ]
    df = df[keep]

    # Partition by date then group_id; no compression
    for (date_str, group_id), part in df.groupby(["date", "group_id"], sort=True):
        target = os.path.join(out_dir, f"date={date_str}", f"group_id={group_id}")
        os.makedirs(target, exist_ok=True)
        part_out = part.drop(columns=["date"]).sort_values("time_epoch")
        part_out.to_parquet(os.path.join(target, "gcp2.parquet"), engine="pyarrow", compression=None, index=False)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Ingest GCP2 CSV to Parquet for Nautilus")
    parser.add_argument("--csv", required=True, help="Absolute path to GCP2 CSV")
    parser.add_argument("--out", default="experiments/7-financial-backtesting/parquet_out/gcp2")
    args = parser.parse_args()

    df = read_gcp2_csv(args.csv)
    if df.empty:
        print("No rows found in CSV")
        return
    clean_and_partition(df, args.out)
    print(f"Wrote GCP2 parquet to {args.out}")


if __name__ == "__main__":
    main()


