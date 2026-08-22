#!/usr/bin/env bash
# Upload the exported GCP1 Parquet dataset to S3.
#
#   ./upload_to_s3.sh s3://my-bucket/gcp1/basket
#   ./upload_to_s3.sh s3://my-bucket/gcp1/basket --dryrun
#
# Layout on S3 mirrors the local Hive partitioning, so Athena/Glue can
# discover partitions with MSCK REPAIR TABLE:
#   s3://<bucket>/<prefix>/month=YYYY-MM/part-*.parquet
set -euo pipefail

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  echo "usage: $0 s3://bucket/prefix [extra aws-cli args...]" >&2
  exit 1
fi
shift

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data"
[[ -d "$SRC" ]] || { echo "no data directory at $SRC - run the export first" >&2; exit 1; }

echo "source : $SRC  ($(du -sh "$SRC" | cut -f1))"
echo "dest   : $DEST"

# Parquet is already SNAPPY-compressed; skip client-side compression.
# --size-only keeps re-runs cheap and makes the sync resumable.
aws s3 sync "$SRC" "$DEST" \
  --exclude "*" --include "*/part-*.parquet" \
  --size-only \
  --no-progress \
  "$@"

echo "done. Register in Athena with the DDL from generate_athena_ddl.py, then:"
echo "  MSCK REPAIR TABLE gcp1_basket;"
