#!/usr/bin/env python3
import os
import argparse
from datetime import datetime, timezone
from typing import List

import pandas as pd
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_V2 = True
except Exception:
    ALPACA_V2 = False


def _parse_epoch(s: str) -> int:
    try:
        return int(s)
    except Exception:
        raise argparse.ArgumentTypeError("Invalid epoch")


def fetch_second_bars(tickers: List[str], start_epoch: int, end_epoch: int) -> pd.DataFrame:
    if not ALPACA_V2:
        raise RuntimeError("alpaca-py not available; please install and configure.")

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_epoch, tz=timezone.utc)

    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Second,
        start=start_dt,
        end=end_dt,
        adjustment=None,
        feed="sip"
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()
    # Standardize columns
    # Index columns: symbol, timestamp
    df.rename(columns={
        "timestamp": "ts",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "trade_count": "trade_count",
        "vwap": "vwap"
    }, inplace=True)
    return df


def write_parquet(df: pd.DataFrame, out_dir: str) -> None:
    if df.empty:
        return
    df["date"] = pd.to_datetime(df["ts"], utc=True).dt.strftime("%Y-%m-%d")
    # Add RTH flag (09:30-16:00 America/New_York)
    ny = ZoneInfo("America/New_York")
    ts_ny = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ny)
    is_rth = (ts_ny.dt.weekday < 5) & (
        ((ts_ny.dt.hour > 9) | ((ts_ny.dt.hour == 9) & (ts_ny.dt.minute >= 30))) &
        ((ts_ny.dt.hour < 16) | ((ts_ny.dt.hour == 16) & (ts_ny.dt.minute == 0)))
    )
    df["is_rth"] = is_rth.astype("int8")
    for (symbol, date_str), part in df.groupby(["symbol", "date"], sort=True):
        target = os.path.join(out_dir, f"symbol={symbol}", f"date={date_str}")
        os.makedirs(target, exist_ok=True)
        part.drop(columns=["date"], inplace=True)
        part.to_parquet(os.path.join(target, "bars.parquet"), engine="pyarrow", compression=None, index=False)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build Alpaca second bars to Parquet")
    parser.add_argument("--tickers", required=True, help="Comma-separated (e.g., IVV,VOO,VXX,SPY,UVXY)")
    parser.add_argument("--start-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--end-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--out", default="experiments/7-financial-backtesting/parquet_out/market")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    df = fetch_second_bars(tickers, args.start_epoch, args.end_epoch)
    write_parquet(df, args.out)
    print(f"Wrote market bars to {args.out}")


if __name__ == "__main__":
    main()


