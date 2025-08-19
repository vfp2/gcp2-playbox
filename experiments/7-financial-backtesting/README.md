## Financial backtesting with GCP 1 and GCP 2 (NautilusTrader-ready)

**Python Version Required: 3.11**

This experiment ingests:
- GCP 1 EGG data from BigQuery, computes per-second Network Coherence proxy using the published Stouffer-Z method, and exports Parquet.
- GCP 2 CSV, selects `netvar_count_xor_alt` per `group_id` at 1 Hz, filters specified bad days, and exports Parquet partitioned by date and group.
- Alpaca historical trade data for tickers, exported to Parquet for backtests.

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
- Parquet output: `parquet_out/`

### Commands

- Ingest GCP 1 from BigQuery to Parquet (UTC seconds, partitioned by date):
```
python experiments/7-financial-backtesting/ingest_gcp1_bigquery.py \
  --start "1998-08-05T00:00:00Z" \
  --end   "2025-07-31T23:59:59Z"
```

**Note:** To match GCP 2 time frame (2024-02-15 to 2024-07-17), use:
```
python experiments/7-financial-backtesting/ingest_gcp1_bigquery.py \
  --start "2024-02-15T00:00:00Z" \
  --end   "2024-07-17T23:59:59Z"
```

- Ingest GCP 2 CSV to Parquet (per `group_id`, filter known-bad days):
```
python experiments/7-financial-backtesting/ingest_gcp2_csv.py \
  --csv /home/soliax/dev/vfp2/gcp2-playbox/experiments/7-financial-backtesting/gcp2.net_netvar_2-15-24_7-15-24.csv
```

- Download Alpaca trade data for tickers and export Parquet (matches GCP 2 dataset exactly):
```
python experiments/7-financial-backtesting/build_market_data_alpaca.py \
  --tickers IVV,VOO,VXX,SPY,UVXY \
  --start-epoch 1707955200 --end-epoch 1721179439
```

- Run GCP2 backtest with per-second price data:
```
python experiments/7-financial-backtesting/run_backtest.py \
  --tickers IVV,VOO,VXX,SPY,UVXY \
  --z-window 3600 --lag-min -3600 --lag-max 3600 --lag-step 300
```

**Updated Features:**
- Now works with per-second price data from Alpaca (instead of bars)
- Comprehensive logging to both console and `backtest.log` file
- Processes all GCP2 groups for each ticker
- Generates detailed summary statistics (PnL, Sharpe ratio, max drawdown, etc.)
- Outputs organized by `symbol/SYMBOL/group_id/NNN/` structure

Notes:
- GCP2 uses `netvar_count_xor_alt` (fully whitened). We drop specified outlier dates and skip missing/duplicate seconds.
- Partitions: `date=YYYY-MM-DD/group_id=NNN` for GCP2; `date=YYYY-MM-DD` for GCP1; symbols for market data.
- No compression is used for Parquet, as requested.
- All datasets now use consistent time range: 2024-02-15 00:00:00 UTC to 2024-07-17 23:59:59 UTC

### Backtest Output Structure

The backtest script generates organized output in `parquet_out/backtests/`:

```
parquet_out/backtests/
├── symbol=IVV/
│   ├── group_id=1/
│   │   ├── backtest.parquet    # Full backtest data (price, signal, position, PnL)
│   │   └── summary.json        # Performance metrics
│   ├── group_id=137/
│   │   ├── backtest.parquet
│   │   └── summary.json
│   └── ...
├── symbol=VOO/
│   ├── group_id=1/
│   │   ├── backtest.parquet
│   │   └── summary.json
│   └── ...
└── ...
```

Each `summary.json` contains:
- Trading period (start/end dates)
- Total PnL and Sharpe ratio (annualized)
- Maximum drawdown
- Total number of trades
- Win rate percentage


