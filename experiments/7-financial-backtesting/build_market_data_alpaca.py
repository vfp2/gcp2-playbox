#!/usr/bin/env python3
import os
import argparse
import logging
from datetime import datetime, timezone
from typing import List
import time

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


class RateLimiter:
    """Simple rate limiter to respect Alpaca API limits."""
    
    def __init__(self, calls_per_second: int = 10):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration with appropriate level and format.
    
    Provides clear visibility into data processing operations for debugging and monitoring.
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
    """Fetch trade-level data from Alpaca API with proper pagination.
    
    This gives us per-second precision since individual trades have microsecond timestamps.
    We use the highest frequency data available for backtesting.
    
    Implements proper pagination to collect ALL available data across the entire time range,
    not just the first 10,000 trades per ticker.
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
    
    # Initialize rate limiter (Alpaca allows ~10 calls per second)
    rate_limiter = RateLimiter(calls_per_second=8)  # Conservative rate limiting
    
    all_trades = []
    total_api_calls = 0
    
    for ticker in tickers:
        logger.info(f"Fetching trades for {ticker}...")
        
        # Initialize pagination variables
        current_start = start_dt
        ticker_trades_count = 0
        page_count = 0
        consecutive_empty_pages = 0
        max_consecutive_empty = 3  # Stop if we get 3 empty pages in a row
        
        while current_start < end_dt and consecutive_empty_pages < max_consecutive_empty:
            page_count += 1
            total_api_calls += 1
            
            # Rate limit before making API call
            rate_limiter.wait_if_needed()
            
            # Request trades for this page
            req = StockTradesRequest(
                symbol_or_symbols=ticker,
                start=current_start,
                end=end_dt,
                limit=10000,  # Max limit per request
                feed="sip"
            )
            
            try:
                trades = client.get_stock_trades(req)
                
                if ticker in trades.data and trades.data[ticker]:
                    ticker_trades = trades.data[ticker]
                    page_trades_count = len(ticker_trades)
                    ticker_trades_count += page_trades_count
                    consecutive_empty_pages = 0  # Reset counter
                    
                    logger.info(f"  Page {page_count}: Received {page_trades_count} trades for {ticker}")
                    
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
                    
                    # Check if we got a full page (means there might be more data)
                    if page_trades_count < 10000:
                        logger.info(f"  Received partial page ({page_trades_count} < 10000), no more data available for {ticker}")
                        break
                    
                    # Update start time for next page - use the timestamp of the last trade
                    last_trade_time = ticker_trades[-1].timestamp
                    if last_trade_time <= current_start:
                        logger.warning(f"  Last trade time ({last_trade_time}) not advancing, stopping pagination for {ticker}")
                        break
                    
                    current_start = last_trade_time + pd.Timedelta(microseconds=1)
                    
                else:
                    consecutive_empty_pages += 1
                    logger.info(f"  Page {page_count}: No trades found for {ticker} (empty page {consecutive_empty_pages}/{max_consecutive_empty})")
                    
                    # If we get empty pages, try advancing time slightly
                    if consecutive_empty_pages == 1:
                        current_start += pd.Timedelta(hours=1)
                        logger.info(f"  Advancing time by 1 hour to {current_start}")
                    elif consecutive_empty_pages == 2:
                        current_start += pd.Timedelta(days=1)
                        logger.info(f"  Advancing time by 1 day to {current_start}")
                    else:
                        logger.warning(f"  Too many consecutive empty pages, stopping pagination for {ticker}")
                        break
                    
            except Exception as e:
                logger.error(f"  Error fetching page {page_count} for {ticker}: {str(e)}")
                # Wait a bit longer on errors
                time.sleep(1)
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    logger.error(f"  Too many consecutive errors, stopping pagination for {ticker}")
                    break
        
        logger.info(f"Completed {ticker}: {ticker_trades_count} trades across {page_count} pages")
    
    logger.info(f"Total trades collected: {len(all_trades)}")
    logger.info(f"Total API calls made: {total_api_calls}")
    
    if not all_trades:
        logger.warning("No trade data found for any ticker")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    logger.info(f"Data processing complete. DataFrame shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Log actual time range of collected data
    if not df.empty:
        actual_start = df['ts'].min()
        actual_end = df['ts'].max()
        logger.info(f"Actual data time range: {actual_start} to {actual_end}")
        
        # Check if we got the full requested range
        requested_start = pd.Timestamp(start_dt)
        requested_end = pd.Timestamp(end_dt)
        
        if actual_start > requested_start + pd.Timedelta(hours=1):
            logger.warning(f"Data collection may be incomplete: actual start ({actual_start}) is much later than requested start ({requested_start})")
        
        if actual_end < requested_end - pd.Timedelta(hours=1):
            logger.warning(f"Data collection may be incomplete: actual end ({actual_end}) is much earlier than requested end ({requested_end})")
    
    return df


def fetch_minute_bars(tickers: List[str], start_epoch: int, end_epoch: int, logger: logging.Logger) -> pd.DataFrame:
    """Fetch minute-level bars from Alpaca API.
    
    Note: Alpaca API only supports predefined timeframes: Day, Hour, Minute, Month, Week.
    The finest granularity available is 1-minute bars. We use the highest frequency data 
    available for backtesting purposes.
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
    parser = argparse.ArgumentParser(description="Build Alpaca market data to Parquet (trades by default, bars optional)")
    parser.add_argument("--tickers", required=True, help="Comma-separated (e.g., IVV,VOO,VXX,SPY,UVXY)")
    parser.add_argument("--start-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--end-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--out", default="parquet_out/market")
    parser.add_argument("--data-type", choices=["bars", "trades"], default="trades", 
                       help="Data type: 'trades' for per-second trade data (default), 'bars' for minute-level OHLCV")
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
            print("🔄 Collecting trade data from Alpaca API...")
            df = fetch_trade_data(tickers, args.start_epoch, args.end_epoch, logger)
            data_description = "per-second trade data"
        else:
            print("🔄 Collecting minute bars from Alpaca API...")
            df = fetch_minute_bars(tickers, args.start_epoch, args.end_epoch, logger)
            data_description = "minute-level bars"
        
        if df.empty:
            logger.error("No data collected! Check your API credentials and time range.")
            return
        
        print(f"✅ Data collection complete: {len(df):,} records")
        
        # Write data
        logger.info("Starting data writing phase...")
        print("💾 Writing data to parquet files...")
        write_parquet(df, args.out, logger)
        print("✅ Data writing complete!")
        
        # Enhanced final summary with data completeness validation
        logger.info("=" * 60)
        logger.info("SCRIPT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total rows processed: {len(df)}")
        logger.info(f"Tickers processed: {len(tickers)}")
        
        if not df.empty:
            actual_start = df['ts'].min()
            actual_end = df['ts'].max()
            requested_start = datetime.fromtimestamp(args.start_epoch, tz=timezone.utc)
            requested_end = datetime.fromtimestamp(args.end_epoch, tz=timezone.utc)
            
            logger.info(f"Requested time range: {requested_start} to {requested_end}")
            logger.info(f"Actual data time range: {actual_start} to {actual_end}")
            
            # Calculate coverage statistics
            requested_duration = (requested_end - requested_start).total_seconds() / 3600  # hours
            actual_duration = (actual_end - actual_start).total_seconds() / 3600  # hours
            coverage_pct = (actual_duration / requested_duration) * 100 if requested_duration > 0 else 0
            
            logger.info(f"Requested duration: {requested_duration:.1f} hours")
            logger.info(f"Actual data duration: {actual_duration:.1f} hours")
            logger.info(f"Time coverage: {coverage_pct:.1f}%")
            
            # Data density analysis
            if args.data_type == "trades":
                total_seconds = actual_duration * 3600
                trades_per_second = len(df) / total_seconds if total_seconds > 0 else 0
                logger.info(f"Data density: {trades_per_second:.2f} trades per second across all tickers")
                
                # Per-ticker breakdown
                for ticker in tickers:
                    ticker_data = df[df['symbol'] == ticker]
                    if not ticker_data.empty:
                        ticker_trades_per_second = len(ticker_data) / total_seconds if total_seconds > 0 else 0
                        logger.info(f"  {ticker}: {len(ticker_data)} trades ({ticker_trades_per_second:.2f} trades/sec)")
            
            # Warn if coverage is poor
            if coverage_pct < 90:
                logger.warning(f"⚠️  POOR TIME COVERAGE: Only {coverage_pct:.1f}% of requested time range covered!")
                logger.warning("This may indicate API limits, data availability issues, or pagination problems.")
            elif coverage_pct < 100:
                logger.warning(f"⚠️  PARTIAL TIME COVERAGE: {coverage_pct:.1f}% of requested time range covered")
            else:
                logger.info("✅ FULL TIME COVERAGE: All requested time range data collected successfully")
        
        logger.info(f"Output location: {args.out}")
        logger.info("=" * 60)
        
        print(f"✓ Successfully wrote {data_description} to {args.out}")
        print(f"  - Processed {len(df)} rows for {len(tickers)} tickers")
        if args.data_type == "trades":
            print(f"  - Per-second precision with microsecond timestamps")
            if not df.empty:
                actual_start = df['ts'].min()
                actual_end = df['ts'].max()
                print(f"  - Data covers: {actual_start.strftime('%Y-%m-%d %H:%M:%S')} to {actual_end.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"  - Note: Alpaca only provides minute-level bars (finest granularity available)")
        
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    main()


