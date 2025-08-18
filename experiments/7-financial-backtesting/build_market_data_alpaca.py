#!/usr/bin/env python3
import os
import argparse
import logging
from datetime import datetime, timezone
from typing import List

import pandas as pd
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from tqdm import tqdm

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.requests import StockTradesRequest
    ALPACA_V2 = True
except Exception:
    ALPACA_V2 = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration with appropriate level and format.
    
    Per Scott Wilber (canon.yaml), logging should provide clear visibility
    into data processing operations for debugging and monitoring.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def _parse_epoch(s: str) -> int:
    try:
        return int(s)
    except Exception:
        raise argparse.ArgumentTypeError("Invalid epoch")


def fetch_trade_data(tickers: List[str], start_epoch: int, end_epoch: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetch trade-level data from Alpaca API.
    
    This gives us per-second precision since individual trades have microsecond timestamps.
    Per Scott Wilber (canon.yaml), we use the highest frequency data available for backtesting.
    """
    if not ALPACA_V2:
        raise RuntimeError("alpaca-py not available; please install and configure.")

    logger.info(f"Starting trade data fetch for {len(tickers)} tickers")
    logger.info(f"Time range: {datetime.fromtimestamp(start_epoch, tz=timezone.utc)} to {datetime.fromtimestamp(end_epoch, tz=timezone.utc)}")
    
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

    logger.info("Connecting to Alpaca API...")
    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_epoch, tz=timezone.utc)

    logger.info("Requesting trade data from Alpaca (per-second precision available)...")
    
    all_trades = []
    for ticker in tickers:
        logger.info(f"Fetching trades for {ticker}...")
        req = StockTradesRequest(
            symbol_or_symbols=ticker,
            start=start_dt,
            end=end_dt,
            limit=10000,  # Max limit per request
            feed="sip"
        )
        
        trades = client.get_stock_trades(req)
        if ticker in trades.data:
            ticker_trades = trades.data[ticker]
            logger.info(f"Received {len(ticker_trades)} trades for {ticker}")
            
            # Convert trades to list of dicts
            for trade in ticker_trades:
                all_trades.append({
                    'symbol': trade.symbol,
                    'ts': trade.timestamp,
                    'price': trade.price,
                    'size': trade.size,
                    'exchange': trade.exchange,
                    'conditions': trade.conditions,
                    'tape': trade.tape,
                    'id': trade.id
                })
        else:
            logger.warning(f"No trades found for {ticker}")
    
    logger.info(f"Total trades collected: {len(all_trades)}")
    
    if not all_trades:
        logger.warning("No trade data found for any ticker")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    logger.info(f"Data processing complete. DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def fetch_minute_bars(tickers: List[str], start_epoch: int, end_epoch: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetch minute-level bars from Alpaca API.
    
    Note: Alpaca API only supports predefined timeframes: Day, Hour, Minute, Month, Week.
    The finest granularity available is 1-minute bars. Per Scott Wilber (canon.yaml), 
    we use the highest frequency data available for backtesting purposes.
    """
    if not ALPACA_V2:
        raise RuntimeError("alpaca-py not available; please install and configure.")

    logger.info(f"Starting data fetch for {len(tickers)} tickers")
    logger.info(f"Time range: {datetime.fromtimestamp(start_epoch, tz=timezone.utc)} to {datetime.fromtimestamp(end_epoch, tz=timezone.utc)}")
    
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

    logger.info("Connecting to Alpaca API...")
    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_epoch, tz=timezone.utc)

    logger.info("Requesting minute bars from Alpaca...")
    req = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        adjustment=None,
        feed="sip"
    )
    
    logger.info("Fetching data...")
    bars = client.get_stock_bars(req)
    
    # Convert BarSet to DataFrame to get the actual count
    df = bars.df.reset_index()
    logger.info(f"Received {len(df)} records")
    
    logger.info("Processing and standardizing data...")
    
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
    
    logger.info(f"Data processing complete. DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def write_parquet(df: pd.DataFrame, out_dir: str, logger: logging.Logger) -> None:
    """Write DataFrame to parquet files with progress logging."""
    if df.empty:
        logger.warning("DataFrame is empty, nothing to write")
        return
    
    logger.info(f"Starting parquet write to {out_dir}")
    logger.info(f"Processing {len(df)} rows of data")
    
    df["date"] = pd.to_datetime(df["ts"], utc=True).dt.strftime("%Y-%m-%d")
    
    # Add RTH flag (09:30-16:00 America/New_York)
    logger.info("Computing Regular Trading Hours (RTH) flags...")
    ny = ZoneInfo("America/New_York")
    ts_ny = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ny)
    is_rth = (ts_ny.dt.weekday < 5) & (
        ((ts_ny.dt.hour > 9) | ((ts_ny.dt.hour == 9) & (ts_ny.dt.minute >= 30))) &
        ((ts_ny.dt.hour < 16) | ((ts_ny.dt.hour == 16) & (ts_ny.dt.minute == 0)))
    )
    df["is_rth"] = is_rth.astype("int8")
    
    # Group by symbol and date for writing
    grouped = df.groupby(["symbol", "date"], sort=True)
    logger.info(f"Writing {len(grouped)} parquet files...")
    
    total_files = len(grouped)
    for i, ((symbol, date_str), part) in enumerate(grouped, 1):
        target = os.path.join(out_dir, f"symbol={symbol}", f"date={date_str}")
        os.makedirs(target, exist_ok=True)
        
        part_copy = part.copy()
        part_copy.drop(columns=["date"], inplace=True)
        
        output_file = os.path.join(target, "data.parquet")
        part_copy.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
        
        logger.info(f"Progress: {i}/{total_files} - Wrote {symbol} {date_str} ({len(part)} rows) to {output_file}")
    
    logger.info(f"Parquet write complete. Total files written: {total_files}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build Alpaca market data to Parquet (bars or trades)")
    parser.add_argument("--tickers", required=True, help="Comma-separated (e.g., IVV,VOO,VXX,SPY,UVXY)")
    parser.add_argument("--start-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--end-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--out", default="experiments/7-financial-backtesting/parquet_out/market")
    parser.add_argument("--data-type", choices=["bars", "trades"], default="bars", 
                       help="Data type: 'bars' for minute-level OHLCV, 'trades' for per-second trade data")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("Alpaca Market Data Collection Script Started")
    logger.info("=" * 60)
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    logger.info(f"Processing tickers: {tickers}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Output directory: {args.out}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    logger.info(f"Ensured output directory exists: {args.out}")
    
    try:
        # Fetch data based on data type
        logger.info("Starting data collection phase...")
        if args.data_type == "trades":
            df = fetch_trade_data(tickers, args.start_epoch, args.end_epoch, logger)
            data_description = "per-second trade data"
        else:
            df = fetch_minute_bars(tickers, args.start_epoch, args.end_epoch, logger)
            data_description = "minute-level bars"
        
        # Write data
        logger.info("Starting data writing phase...")
        write_parquet(df, args.out, logger)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("SCRIPT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total rows processed: {len(df)}")
        logger.info(f"Tickers processed: {len(tickers)}")
        logger.info(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
        logger.info(f"Output location: {args.out}")
        logger.info("=" * 60)
        
        print(f"✓ Successfully wrote {data_description} to {args.out}")
        print(f"  - Processed {len(df)} rows for {len(tickers)} tickers")
        if args.data_type == "trades":
            print(f"  - Per-second precision with microsecond timestamps")
        else:
            print(f"  - Note: Alpaca only provides minute-level bars (finest granularity available)")
        
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    main()


