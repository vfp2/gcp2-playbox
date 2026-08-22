#!/usr/bin/env python3
"""Export the full GCP1 egg dataset from BigQuery to Parquet, ready for S3.

Source : gcpingcp.eggs_us.basket  (866M rows, 153 cols, MONTH-partitioned on recorded_at)
Output : Hive-partitioned Parquet  month=YYYY-MM/part-*.parquet

The 152 egg_* columns are stored in BigQuery as single BYTES values whose ASCII
code is the per-second trial sum (200 bits, mean 100, sd 7.0712). ASCII() converts
each to a real INT64 in the range 0-255, so the Parquet holds integers rather than
binary blobs. NULL (egg offline) is preserved as NULL; 0 is the legacy
"no data" marker used by the upstream feed.

Two stages, each independently resumable via a JSON manifest:
  export   BigQuery EXPORT DATA -> gs://<bucket>/<prefix>/month=YYYY-MM/
  download gcloud storage rsync -> <out>/month=YYYY-MM/

Usage:
  python export_gcp1_to_parquet.py export             # all months -> GCS
  python export_gcp1_to_parquet.py download           # GCS -> local parquet
  python export_gcp1_to_parquet.py all                # both stages
  python export_gcp1_to_parquet.py verify             # row counts local vs BigQuery
  python export_gcp1_to_parquet.py all --start 2015-01 --end 2015-12
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from google.cloud import bigquery

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.getenv("GCP_PROJECT", "gcpingcp")
DATASET = os.getenv("GCP_DATASET", "eggs_us")
TABLE = os.getenv("GCP_TABLE", "basket")
BUCKET = os.getenv("GCP1_EXPORT_BUCKET", "gcpingcp-gcp1-parquet-us")
PREFIX = os.getenv("GCP1_EXPORT_PREFIX", "gcp1_basket")
OUT_DIR = os.path.join(HERE, "data")
MANIFEST = os.path.join(HERE, "manifest.json")
COLUMNS_JSON = os.path.join(HERE, "egg_columns.json")

_lock = threading.Lock()


# --------------------------------------------------------------------------- manifest


def load_manifest() -> dict:
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            return json.load(fh)
    return {"exported": {}, "downloaded": {}}


def save_manifest(m: dict) -> None:
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
    os.replace(tmp, MANIFEST)


def mark(m: dict, stage: str, month: str, info: dict) -> None:
    with _lock:
        m[stage][month] = info
        save_manifest(m)


# --------------------------------------------------------------------------- discovery


def egg_columns(client: bigquery.Client) -> list:
    """Egg column names, cached to disk so the layout is reproducible."""
    if os.path.exists(COLUMNS_JSON):
        with open(COLUMNS_JSON) as fh:
            return json.load(fh)
    table = client.get_table(f"{PROJECT}.{DATASET}.{TABLE}")
    cols = [f.name for f in table.schema if f.name != "recorded_at"]
    with open(COLUMNS_JSON, "w") as fh:
        json.dump(cols, fh, indent=1)
    return cols


def all_months(client: bigquery.Client) -> list:
    """Distinct YYYY-MM buckets that actually contain rows."""
    sql = f"""
    SELECT FORMAT_TIMESTAMP('%Y-%m', recorded_at) AS month, COUNT(*) AS n
    FROM `{PROJECT}.{DATASET}.{TABLE}`
    GROUP BY month ORDER BY month
    """
    return [(r["month"], int(r["n"])) for r in client.query(sql).result()]


def month_bounds(month: str) -> tuple:
    y, mth = int(month[:4]), int(month[5:7])
    start = datetime(y, mth, 1, tzinfo=timezone.utc)
    end = datetime(y + (mth == 12), (mth % 12) + 1, 1, tzinfo=timezone.utc)
    return start, end


# --------------------------------------------------------------------------- export


def export_sql(cols: list, month: str) -> str:
    start, end = month_bounds(month)
    select = ",\n      ".join(f"ASCII({c}) AS {c}" for c in cols)
    uri = f"gs://{BUCKET}/{PREFIX}/month={month}/part-*.parquet"
    return f"""
    EXPORT DATA OPTIONS(
      uri='{uri}',
      format='PARQUET',
      compression='SNAPPY',
      overwrite=true
    ) AS
    SELECT
      recorded_at,
      {select}
    FROM `{PROJECT}.{DATASET}.{TABLE}`
    WHERE recorded_at >= TIMESTAMP('{start:%Y-%m-%d %H:%M:%S}')
      AND recorded_at <  TIMESTAMP('{end:%Y-%m-%d %H:%M:%S}')
    ORDER BY recorded_at
    """


def export_month(client: bigquery.Client, cols: list, month: str, rows: int, m: dict) -> str:
    job = client.query(export_sql(cols, month))
    job.result()
    info = {
        "rows": rows,
        "job_id": job.job_id,
        "bytes_billed": job.total_bytes_billed,
        "gcs": f"gs://{BUCKET}/{PREFIX}/month={month}/",
    }
    mark(m, "exported", month, info)
    gb = (job.total_bytes_billed or 0) / 1e9
    return f"exported {month}  rows={rows:>9,}  scanned={gb:5.2f} GB"


def stage_export(client: bigquery.Client, months: list, workers: int, force: bool) -> None:
    m = load_manifest()
    cols = egg_columns(client)
    todo = [(mo, n) for mo, n in months if force or mo not in m["exported"]]
    print(f"[export] {len(todo)} months to export ({len(months) - len(todo)} already done)")
    if not todo:
        return
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(export_month, client, cols, mo, n, m): mo for mo, n in todo}
        for fut in as_completed(futs):
            done += 1
            try:
                print(f"[export] {done}/{len(todo)}  {fut.result()}", flush=True)
            except Exception as exc:
                print(f"[export] {done}/{len(todo)}  FAILED {futs[fut]}: {exc}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- download


def download_month(month: str, m: dict) -> str:
    dest = os.path.join(OUT_DIR, f"month={month}")
    os.makedirs(dest, exist_ok=True)
    src = f"gs://{BUCKET}/{PREFIX}/month={month}/"
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "--delete-unmatched-destination-objects",
         "--project", PROJECT, src, dest],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip()[-400:])
    files = [f for f in os.listdir(dest) if f.endswith(".parquet")]
    size = sum(os.path.getsize(os.path.join(dest, f)) for f in files)
    mark(m, "downloaded", month, {"files": len(files), "bytes": size})
    return f"downloaded {month}  files={len(files):>3}  {size/1e6:8.1f} MB"


def stage_download(months: list, workers: int, force: bool) -> None:
    m = load_manifest()
    os.makedirs(OUT_DIR, exist_ok=True)
    avail = [mo for mo, _ in months if mo in m["exported"]]
    todo = [mo for mo in avail if force or mo not in m["downloaded"]]
    print(f"[download] {len(todo)} months to download ({len(avail) - len(todo)} already local)")
    if not todo:
        return
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(download_month, mo, m): mo for mo in todo}
        for fut in as_completed(futs):
            done += 1
            try:
                print(f"[download] {done}/{len(todo)}  {fut.result()}", flush=True)
            except Exception as exc:
                print(f"[download] {done}/{len(todo)}  FAILED {futs[fut]}: {exc}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- verify


def stage_verify(months: list) -> None:
    import pyarrow.parquet as pq

    expected = dict(months)
    bad, total_rows, total_bytes = [], 0, 0
    for month in sorted(expected):
        d = os.path.join(OUT_DIR, f"month={month}")
        if not os.path.isdir(d):
            bad.append((month, expected[month], "MISSING"))
            continue
        n = 0
        for f in sorted(os.listdir(d)):
            if f.endswith(".parquet"):
                p = os.path.join(d, f)
                n += pq.ParquetFile(p).metadata.num_rows
                total_bytes += os.path.getsize(p)
        total_rows += n
        if n != expected[month]:
            bad.append((month, expected[month], n))
    print(f"\n[verify] months checked : {len(expected)}")
    print(f"[verify] local rows     : {total_rows:,}")
    print(f"[verify] BigQuery rows  : {sum(expected.values()):,}")
    print(f"[verify] local size     : {total_bytes/1e9:.2f} GB")
    if bad:
        print(f"[verify] MISMATCHES ({len(bad)}):")
        for month, exp, got in bad:
            print(f"           month={month}  bigquery={exp:,}  local={got}")
        sys.exit(1)
    print("[verify] OK - every month matches BigQuery row counts")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["export", "download", "all", "verify", "months"])
    ap.add_argument("--start", help="first month YYYY-MM (inclusive)")
    ap.add_argument("--end", help="last month YYYY-MM (inclusive)")
    ap.add_argument("--workers", type=int, default=8, help="parallel months (default 8)")
    ap.add_argument("--force", action="store_true", help="redo months already in the manifest")
    args = ap.parse_args()

    client = bigquery.Client(project=PROJECT)
    months = all_months(client)
    if args.start:
        months = [x for x in months if x[0] >= args.start]
    if args.end:
        months = [x for x in months if x[0] <= args.end]

    print(f"source : {PROJECT}.{DATASET}.{TABLE}")
    print(f"staging: gs://{BUCKET}/{PREFIX}/")
    print(f"output : {OUT_DIR}")
    print(f"months : {len(months)}  ({months[0][0]} .. {months[-1][0]})  rows={sum(n for _, n in months):,}\n")

    if args.stage == "months":
        for mo, n in months:
            print(f"  {mo}  {n:>9,}")
        return
    if args.stage in ("export", "all"):
        stage_export(client, months, args.workers, args.force)
    if args.stage in ("download", "all"):
        stage_download(months, args.workers, args.force)
    if args.stage == "verify":
        stage_verify(months)


if __name__ == "__main__":
    main()
