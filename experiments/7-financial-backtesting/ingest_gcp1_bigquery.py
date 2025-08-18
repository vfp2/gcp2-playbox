#!/usr/bin/env python3
import os
import argparse
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery_storage_v1 import BigQueryReadClient
from dotenv import load_dotenv


def setup_logging():
    """Setup logging configuration for progress tracking"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def _parse_dt(s: str) -> datetime:
    try:
        if s.endswith('Z'):
            return datetime.fromisoformat(s.replace('Z', '+00:00')).astimezone(timezone.utc)
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid datetime: {s}")


def query_gcp1_seconds(bq: bigquery.Client, bqs: BigQueryReadClient, start: datetime, end: datetime, project: str, dataset: str, table: str, filter_zero: bool, logger: logging.Logger) -> pd.DataFrame:
    # Mirrors logic from experiments/4-rolling-windows/gcp_egg_web_app.py
    # 1) ASCII() per egg column, 2) z-score with mu=100, sigma=7.0712, 3) Stouffer Z, 4) (Z^2 - 1) per second
    # We will output columns: ts (UTC), chi2_stouffer (float), active_eggs (int)
    # EGG column list (copied to avoid importing from a non-package path)
    EGG_COLS = [
        "egg_1","egg_28","egg_33","egg_34","egg_37","egg_100","egg_101","egg_102","egg_103","egg_104",
        "egg_105","egg_106","egg_107","egg_108","egg_109","egg_110","egg_111","egg_112","egg_114","egg_115",
        "egg_116","egg_117","egg_118","egg_119","egg_134","egg_142","egg_161","egg_223","egg_224","egg_226",
        "egg_227","egg_228","egg_230","egg_231","egg_233","egg_237","egg_1000","egg_1003","egg_1004","egg_1005",
        "egg_1013","egg_1021","egg_1022","egg_1023","egg_1024","egg_1025","egg_1026","egg_1027","egg_1029",
        "egg_1051","egg_1063","egg_1066","egg_1070","egg_1082","egg_1092","egg_1095","egg_1096","egg_1101",
        "egg_1113","egg_1223","egg_1237","egg_1245","egg_1251","egg_1295","egg_2000","egg_2001","egg_2002",
        "egg_2006","egg_2007","egg_2008","egg_2009","egg_2013","egg_2022","egg_2023","egg_2024","egg_2026",
        "egg_2027","egg_2028","egg_2040","egg_2041","egg_2042","egg_2043","egg_2044","egg_2045","egg_2046",
        "egg_2047","egg_2048","egg_2049","egg_2052","egg_2060","egg_2061","egg_2062","egg_2064","egg_2069",
        "egg_2070","egg_2073","egg_2080","egg_2083","egg_2084","egg_2088","egg_2091","egg_2093","egg_2094",
        "egg_2097","egg_2120","egg_2165","egg_2173","egg_2178","egg_2201","egg_2202","egg_2220","egg_2221",
        "egg_2222","egg_2225","egg_2230","egg_2231","egg_2232","egg_2234","egg_2235","egg_2236","egg_2239",
        "egg_2240","egg_2241","egg_2242","egg_2243","egg_2244","egg_2247","egg_2248","egg_2249","egg_2250",
        "egg_3005","egg_3023","egg_3043","egg_3045","egg_3066","egg_3101","egg_3103","egg_3104","egg_3106",
        "egg_3107","egg_3108","egg_3115","egg_3142","egg_3240","egg_3247","egg_4002","egg_4101","egg_4234",
        "egg_4251"
    ]

    logger.info(f"Building SQL query for {len(EGG_COLS)} egg columns")
    logger.info(f"Query time range: {start} to {end}")
    logger.info(f"Filter zero values: {filter_zero}")

    ascii_block = ",\n".join([f"    ASCII({c}) AS {c}" for c in EGG_COLS])

    # Build aliased z_ columns and Stouffer sum
    z_block_list = []
    for c in EGG_COLS:
        if filter_zero:
            expr = f"IF({c} IS NOT NULL AND {c} != 0, SAFE_DIVIDE(({c} - 100), 7.0712), NULL)"
        else:
            expr = f"IF({c} IS NOT NULL, SAFE_DIVIDE(({c} - 100), 7.0712), NULL)"
        z_block_list.append(f"    {expr} AS z_{c}")
    z_block = ",\n".join(z_block_list)
    stouffer_sum = " + ".join([f"IF(z_{c} IS NOT NULL, z_{c}, 0)" for c in EGG_COLS])
    null_count_block = " + ".join([f"IF(z_{c} IS NULL, 0, 1)" for c in EGG_COLS])

    sql = f"""
    WITH raw AS (
      SELECT recorded_at,
{ascii_block}
      FROM `{project}.{dataset}.{table}`
      WHERE recorded_at BETWEEN TIMESTAMP(@start_ts) AND TIMESTAMP(@end_ts)
    ),
    z AS (
      SELECT recorded_at,
{z_block}
      FROM raw
    ),
    sec AS (
      SELECT
        recorded_at AS ts,
        POW(SAFE_DIVIDE({stouffer_sum}, SQRT({null_count_block})), 2) - 1 AS chi2_stouffer,
        {null_count_block} AS active_eggs
      FROM z
    )
    SELECT ts, chi2_stouffer, active_eggs
    FROM sec
    ORDER BY ts
    """

    logger.info("Executing BigQuery job...")
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP", start),
            bigquery.ScalarQueryParameter("end_ts", "TIMESTAMP", end),
        ]
    ))
    
    logger.info(f"BigQuery job ID: {job.job_id}")
    logger.info("Waiting for job completion...")
    
    # Wait for the job to complete
    job.result()
    
    logger.info("Job completed successfully. Converting to DataFrame...")
    df = job.to_dataframe(bqstorage_client=bqs, create_bqstorage_client=True)
    
    logger.info(f"DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    logger.info(f"DataFrame columns: {list(df.columns)}")
    logger.info(f"DataFrame shape: {df.shape}")
    
    if not df.empty:
        logger.info(f"Time range in data: {df['ts'].min()} to {df['ts'].max()}")
        logger.info(f"Active eggs range: {df['active_eggs'].min()} to {df['active_eggs'].max()}")
        logger.info(f"Chi2 Stouffer range: {df['chi2_stouffer'].min():.4f} to {df['chi2_stouffer'].max():.4f}")
    
    return df


def write_parquet(df: pd.DataFrame, out_dir: str, logger: logging.Logger) -> None:
    logger.info(f"Preparing to write parquet files to {out_dir}")
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["ts"], utc=True).dt.strftime("%Y-%m-%d")
    
    # Get unique dates for partitioning
    unique_dates = df["date"].unique()
    logger.info(f"Found {len(unique_dates)} unique dates for partitioning")
    
    total_rows = 0
    # Partition by date
    for i, date_str in enumerate(unique_dates, 1):
        part = df[df["date"] == date_str]
        target = os.path.join(out_dir, f"date={date_str}")
        os.makedirs(target, exist_ok=True)
        
        part_copy = part.drop(columns=["date"])
        output_file = os.path.join(target, "gcp1.parquet")
        
        logger.info(f"Writing date {i}/{len(unique_dates)}: {date_str} ({len(part)} rows) to {output_file}")
        part_copy.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
        
        total_rows += len(part)
        logger.info(f"Completed {date_str}: {len(part)} rows written")
    
    logger.info(f"Parquet writing completed. Total {total_rows} rows written across {len(unique_dates)} date partitions")


def main():
    load_dotenv()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting GCP1 BigQuery ingestion script")
    
    parser = argparse.ArgumentParser(description="Ingest GCP1 BigQuery to Parquet for Nautilus")
    parser.add_argument("--start", required=True, type=_parse_dt, help="UTC start ISO (e.g. 1998-08-05T00:00:00Z)")
    parser.add_argument("--end", required=True, type=_parse_dt, help="UTC end ISO")
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT", "gcpingcp"))
    parser.add_argument("--dataset", default="eggs_us")
    parser.add_argument("--table", default="basket")
    parser.add_argument("--filter-zero", action="store_true", default=True, help="Exclude zero trial sums")
    parser.add_argument("--out", default="experiments/7-financial-backtesting/parquet_out/gcp1")
    args = parser.parse_args()

    logger.info(f"Arguments parsed:")
    logger.info(f"  Start: {args.start}")
    logger.info(f"  End: {args.end}")
    logger.info(f"  Project: {args.project}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Table: {args.table}")
    logger.info(f"  Filter zero: {args.filter_zero}")
    logger.info(f"  Output directory: {args.out}")

    logger.info("Initializing BigQuery clients...")
    bq = bigquery.Client(project=args.project)
    bqs = BigQueryReadClient()
    logger.info("BigQuery clients initialized successfully")

    logger.info("Executing GCP1 query...")
    df = query_gcp1_seconds(bq, bqs, args.start, args.end, args.project, args.dataset, args.table, args.filter_zero, logger)
    
    if df.empty:
        logger.warning("No rows returned from query. Exiting.")
        return
    
    logger.info("Writing data to parquet format...")
    write_parquet(df, args.out, logger)
    
    logger.info(f"GCP1 ingestion completed successfully!")
    logger.info(f"Output location: {args.out}")


if __name__ == "__main__":
    main()


