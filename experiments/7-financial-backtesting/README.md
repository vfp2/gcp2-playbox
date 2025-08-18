## Financial backtesting with GCP 1 and GCP 2 (NautilusTrader-ready)

**Python Version Required: 3.11**

This experiment ingests:
- GCP 1 EGG data from BigQuery, computes per-second Network Coherence proxy using the published Stouffer-Z method, and exports Parquet.
- GCP 2 CSV, selects `netvar_count_xor_alt` per `group_id` at 1 Hz, filters specified bad days, and exports Parquet partitioned by date and group.
- Alpaca historical second bars for tickers, exported to Parquet for backtests.

It also scaffolds a NautilusTrader strategy that consumes the Parquet signals and bars. Strategy is modular to plug other signal processors.

### Setup
1) Create and populate `.env` at repo root (already provided):
```
GCP_PROJECT=gcpingcp
GOOGLE_APPLICATION_CREDENTIALS=./bigquery_service_account.json
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

2) Install deps (recommend venv):
```
python -m venv .venv && source .venv/bin/activate
pip install -r experiments/7-financial-backtesting/requirements.txt
```

### Paths
- GCP2 CSV directory (contains the CSV): `/home/soliax/dev/vfp2/gcp2-playbox/experiments/7-financial-backtesting`
- Parquet output: `experiments/7-financial-backtesting/parquet_out/`

### Commands

- Ingest GCP 1 from BigQuery to Parquet (UTC seconds, partitioned by date):
```
python experiments/7-financial-backtesting/ingest_gcp1_bigquery.py \
  --start "1998-08-05T00:00:00Z" \
  --end   "2025-07-31T23:59:59Z"
```

**Note:** To match GCP 2 time frame (2024-02-15 to 2024-07-18), use:
```
python experiments/7-financial-backtesting/ingest_gcp1_bigquery.py \
  --start "2024-02-15T00:00:00Z" \
  --end   "2024-07-18T23:59:59Z"
```

- Ingest GCP 2 CSV to Parquet (per `group_id`, filter known-bad days):
```
python experiments/7-financial-backtesting/ingest_gcp2_csv.py \
  --csv /home/soliax/dev/vfp2/gcp2-playbox/experiments/7-financial-backtesting/gcp2.csv
```

- Download Alpaca second bars for tickers and export Parquet:
```
python experiments/7-financial-backtesting/build_market_data_alpaca.py \
  --tickers IVV,VOO,VXX,SPY,UVXY \
  --start-epoch 1707955200 --end-epoch 1721179439
```

- Run Nautilus backtest (scaffold):
```
python experiments/7-financial-backtesting/run_backtest.py \
  --tickers IVV,VOO,VXX,SPY,UVXY \
  --z-window 3600 --lag-min -3600 --lag-max 3600 --lag-step 300
```

Notes:
- GCP2 uses `netvar_count_xor_alt` (fully whitened). We drop specified outlier dates and skip missing/duplicate seconds.
- Partitions: `date=YYYY-MM-DD/group_id=NNN` for GCP2; `date=YYYY-MM-DD` for GCP1; symbols for market bars.
- No compression is used for Parquet, as requested.


