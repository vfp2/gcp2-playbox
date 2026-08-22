# GCP1 egg data — BigQuery → Parquet → S3

Full export of the GCP1 (Global Consciousness Project) egg dataset out of BigQuery
into Hive-partitioned Parquet, ready to upload to S3 and query with Athena, Glue,
EMR, Redshift Spectrum, or anything else that reads Parquet.

## Source

| | |
|---|---|
| Table | `gcpingcp.eggs_us.basket` |
| Rows | 866,749,950 (one per UTC second) |
| Span | 1998-08-03 → 2026-02-13 (331 months) |
| Columns | 153 = `recorded_at` + 152 `egg_*` |
| BigQuery layout | MONTH-partitioned + clustered on `recorded_at` |
| Logical size | 116.4 GB |

In BigQuery each `egg_*` column is a **single `BYTES` value**. Its ASCII code is
that egg's per-second trial sum — 200 bits, mean 100, sd 7.0712. Raw `BYTES`
would land in Parquet as opaque binary, which is painful in Athena, so the export
applies `ASCII()` and stores each as a real **`INT64` in 0–255**.

Value semantics, preserved faithfully from the source:

- `NULL` — egg was offline / not reporting that second
- `0` — upstream "no data" marker (a genuine sum of 0 has probability ~1e-60)
- `1..255` — the trial sum

## Output layout

```
data/
  month=1998-08/part-000000000000.parquet
  month=1998-08/part-000000000001.parquet
  ...
  month=2026-02/part-00000000000N.parquet
```

Parquet is SNAPPY-compressed and sorted by `recorded_at` within each month.
`month=YYYY-MM` is a Hive partition key, so Athena picks it up via
`MSCK REPAIR TABLE` and can prune whole months.

> Storing the eggs as int64 rather than int16/uint8 costs nothing — Parquet
> dictionary+RLE encoding already collapses the 256 distinct values, and a
> measured int16 downcast came out 2.5% *larger*. No post-processing needed.

## Usage

The scripts read credentials from `../bigquery_service_account.json`:

```bash
cd /home/soliax/sites/gcp2-playbox
export GOOGLE_APPLICATION_CREDENTIALS=$PWD/bigquery_service_account.json

# both stages, 8 months in parallel
.venv/bin/python gcp1-bigquery-export/export_gcp1_to_parquet.py all --workers 8

# stages individually
... export_gcp1_to_parquet.py export      # BigQuery EXPORT DATA -> GCS
... export_gcp1_to_parquet.py download    # GCS -> local data/

# a subset
... export_gcp1_to_parquet.py all --start 2015-01 --end 2015-12

# confirm local Parquet row counts match BigQuery, month by month
... export_gcp1_to_parquet.py verify
```

Both stages are **resumable**: every completed month is recorded in
`manifest.json` and skipped on the next run, so an interrupted export just
picks up where it stopped. Use `--force` to redo months.

`egg_columns.json` pins the column order so the layout stays reproducible.

## How it works

BigQuery cannot write Parquet straight to your disk, so the export goes through
a GCS bucket that must be **co-located with the dataset**. The dataset is in the
`US` multi-region and the pre-existing `gs://gcp_eggs_data` bucket is in
`ASIA-SOUTHEAST1`, so this pipeline uses a dedicated bucket:

```
gs://gcpingcp-gcp1-parquet-us/gcp1_basket/month=YYYY-MM/
```

That staging bucket was deleted after the S3 upload was verified, so re-running
the export needs it recreated first:

```bash
gcloud storage buckets create gs://gcpingcp-gcp1-parquet-us \
    --project=gcpingcp --location=US --uniform-bucket-level-access
```

Per month it runs an `EXPORT DATA` statement (which writes Parquet directly to
GCS — no intermediate BigQuery table, so no extra storage cost), then
`gcloud storage rsync` pulls that month down to `data/`.

Override with env vars: `GCP_PROJECT`, `GCP_DATASET`, `GCP_TABLE`,
`GCP1_EXPORT_BUCKET`, `GCP1_EXPORT_PREFIX`.

## Live location

The dataset is uploaded and queryable:

| | |
|---|---|
| S3 | `s3://global-consciousness-project-data/gcp1/basket/` (us-east-1) |
| Athena database | `global_consciousness_project` |
| Athena table | `gcp1_basket` (331 month partitions) |
| Verified | 2,159 objects / 33,119,115,445 bytes / 866,749,950 rows |

Athena `SELECT count(*)` returns 866,749,950 — an exact match with BigQuery.

## Upload to S3

```bash
./upload_to_s3.sh s3://my-bucket/gcp1/basket
./upload_to_s3.sh s3://my-bucket/gcp1/basket --dryrun   # preview
```

Uses `aws s3 sync --size-only`, so it is resumable and cheap to re-run.

Then register the table in Athena:

```bash
.venv/bin/python gcp1-bigquery-export/generate_athena_ddl.py \
    s3://my-bucket/gcp1/basket > gcp1_basket.sql
```

Run that SQL in Athena (it ends with `MSCK REPAIR TABLE`) and query. The
pre-generated DDL for the live bucket is checked in as
`gcp1_basket_athena.sql`. Athena must run in the same region as the bucket
(us-east-1). Example:

```sql
-- Stouffer Z and Netvar chi-square for one day, mirroring the Nelson analysis
SELECT recorded_at,
       pow(z_sum / sqrt(n_eggs), 2) - 1 AS chi2_stouffer,
       n_eggs
FROM (
  SELECT recorded_at,
         (coalesce(if(egg_1   > 0, (egg_1   - 100) / 7.0712), 0) +
          coalesce(if(egg_28  > 0, (egg_28  - 100) / 7.0712), 0)
          /* ... one term per egg ... */) AS z_sum,
         (if(egg_1 > 0, 1, 0) + if(egg_28 > 0, 1, 0) /* ... */) AS n_eggs
  FROM gcp1_basket
  WHERE month = '2015-06'
)
ORDER BY recorded_at;
```

Filter on the `month` partition wherever possible — Athena bills by bytes
scanned, and a full-table scan reads the whole ~34 GB.

### Skipping the local hop

If the local copy is only a staging step, GCS → S3 can go direct without
touching this machine — relay it from a cloud VM (rclone on a GCE instance)
rather than pulling 33 GB down a 3 MB/s link and pushing it back up. That
avoids both the wall-clock cost and paying GCS egress twice.

## Costs (one-off, approximate)

| Item | Estimate |
|---|---|
| BigQuery query scan (116 GB @ $6.25/TB) | ~$0.73 |
| `EXPORT DATA` to GCS | free |
| GCS storage for the staging copy | ~$0.60/month for ~34 GB (delete when done) |
| GCS egress to download locally (~34 GB @ $0.12/GB) | ~$4 |

The GCS staging copy was deleted once S3 was verified, so those two lines are
one-off costs that are no longer accruing.

Note on throughput: this server's upstream is capped around 3 MB/s, measured
against both S3 and GCS, so the 33 GB upload took ~2.5 h. Parallel streams do
not help - the link, not concurrency, is the bottleneck. To move the data
faster in future, relay it from a cloud VM rather than through this machine.
