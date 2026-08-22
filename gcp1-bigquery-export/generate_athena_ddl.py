#!/usr/bin/env python3
"""Emit the Athena / Glue external-table DDL for the exported GCP1 Parquet.

  python generate_athena_ddl.py s3://my-bucket/gcp1/basket > gcp1_basket.sql
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: generate_athena_ddl.py s3://bucket/prefix [table_name]")
    location = sys.argv[1].rstrip("/") + "/"
    table = sys.argv[2] if len(sys.argv) > 2 else "gcp1_basket"

    with open(os.path.join(HERE, "egg_columns.json")) as fh:
        eggs = json.load(fh)

    cols = ["  `recorded_at` timestamp"] + [f"  `{e}` int" for e in eggs]
    print(f"""-- GCP1 egg dataset (Global Consciousness Project, 1998-08 .. 2026-02)
-- One row per UTC second. Each egg_* column is that egg's 200-bit trial sum
-- (0-255, mean 100, sd 7.0712). NULL = egg offline. 0 = upstream no-data marker.
CREATE EXTERNAL TABLE IF NOT EXISTS `{table}` (
{",\n".join(cols)}
)
PARTITIONED BY (`month` string)
STORED AS PARQUET
LOCATION '{location}'
TBLPROPERTIES ('parquet.compression'='SNAPPY');

MSCK REPAIR TABLE `{table}`;""")


if __name__ == "__main__":
    main()
