#!/usr/bin/env python3
"""
Memory-optimized Alpaca market data collector with incremental saving and resume capability.

This script addresses memory issues by:
1. Processing one ticker at a time
2. Saving data incrementally (per day)
3. Allowing resumption from failures
4. Implementing memory management and garbage collection

Based on Scott Wilber's expertise in financial data collection and GCP systems.
"""
import os
import argparse
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import time
import gc
import json

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
    """Setup logging configuration with appropriate level and format."""
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


def get_progress_file(out_dir: str) -> str:
    """Get the path to the progress tracking file."""
    return os.path.join(out_dir, ".collection_progress.json")


def load_progress(out_dir: str) -> Dict:
    """Load progress from the progress file."""
    progress_file = get_progress_file(out_dir)
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load progress file: {e}")
    return {"completed_tickers": [], "failed_tickers": [], "current_ticker": None, "current_date": None}


def save_progress(out_dir: str, progress: Dict) -> None:
    """Save progress to the progress file."""
    progress_file = get_progress_file(out_dir)
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logging.error(f"Could not save progress file: {e}")


def check_existing_data(out_dir: str, ticker: str, date_str: str) -> bool:
    """Check if data already exists for a ticker and date."""
    target_dir = os.path.join(out_dir, f"symbol={ticker}", f"date={date_str}")
    data_file = os.path.join(target_dir, "data.parquet")
    return os.path.exists(data_file)


def save_ticker_data(df: pd.DataFrame, out_dir: str, ticker: str, date_str: str, logger: logging.Logger) -> None:
    """Save data for a specific ticker and date."""
    if df.empty:
        logger.warning(f"No data to save for {ticker} on {date_str}")
        return
    
    target_dir = os.path.join(out_dir, f"symbol={ticker}", f"date={date_str}")
    os.makedirs(target_dir, exist_ok=True)
    
    output_file = os.path.join(target_dir, "data.parquet")
    
    # Add RTH flag (09:30-16:00 America/New_York)
    logger.info(f"Computing RTH flags for {ticker} {date_str}...")
    ny = ZoneInfo("America/New_York")
    ts_ny = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ny)
    is_rth = (ts_ny.dt.weekday < 5) & (
        ((ts_ny.dt.hour > 9) | ((ts_ny.dt.hour == 9) & (ts_ny.dt.minute >= 30))) &
        ((ts_ny.dt.hour < 16) | ((ts_ny.dt.hour == 16) & (ts_ny.dt.minute == 0)))
    )
    df["is_rth"] = is_rth.astype("int8")
    
    df.to_parquet(output_file, engine="pyarrow", compression=None, index=False)
    logger.info(f"Saved {len(df)} rows for {ticker} {date_str} to {output_file}")


def fetch_ticker_trades_incremental(
    ticker: str, 
    start_epoch: int, 
    end_epoch: int, 
    out_dir: str,
    logger: logging.Logger,
    progress: Dict
) -> bool:
    """Fetch trade data for a single ticker with incremental saving."""
    if not ALPACA_V2:
        raise RuntimeError("alpaca-py not available; please install and configure.")

    logger.info(f"Fetching trades for {ticker}")
    logger.info(f"Time range: {datetime.fromtimestamp(start_epoch, tz=timezone.utc)} to {datetime.fromtimestamp(end_epoch, tz=timezone.utc)}")
    
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_epoch, tz=timezone.utc)
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(calls_per_second=8)
    
    # Process data day by day to manage memory
    current_date = start_dt.date()
    end_date = end_dt.date()
    
    total_trades = 0
    days_processed = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Check if we already have this date
        if check_existing_data(out_dir, ticker, date_str):
            logger.info(f"Data already exists for {ticker} {date_str}, skipping")
            current_date += timedelta(days=1)
            continue
        
        # Update progress
        progress["current_ticker"] = ticker
        progress["current_date"] = date_str
        save_progress(out_dir, progress)
        
        # Set time boundaries for this day
        day_start = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end = datetime.combine(current_date, datetime.max.time(), tzinfo=timezone.utc)
        
        # Adjust for actual start/end times
        if current_date == start_dt.date():
            day_start = start_dt
        if current_date == end_dt.date():
            day_end = end_dt
        
        logger.info(f"Processing {ticker} for {date_str} ({day_start} to {day_end})")
        
        # Collect trades for this day
        day_trades = []
        current_start = day_start
        page_count = 0
        consecutive_empty_pages = 0
        max_consecutive_empty = 3
        
        while current_start < day_end and consecutive_empty_pages < max_consecutive_empty:
            page_count += 1
            
            # Rate limit before making API call
            rate_limiter.wait_if_needed()
            
            # Request trades for this page
            req = StockTradesRequest(
                symbol_or_symbols=ticker,
                start=current_start,
                end=day_end,
                limit=10000,
                feed="sip"
            )
            
            try:
                trades = client.get_stock_trades(req)
                
                if ticker in trades.data and trades.data[ticker]:
                    ticker_trades = trades.data[ticker]
                    page_trades_count = len(ticker_trades)
                    consecutive_empty_pages = 0
                    
                    logger.info(f"  Page {page_count}: Received {page_trades_count} trades for {ticker} {date_str}")
                    
                    # Convert trades to list of dicts
                    for trade in ticker_trades:
                        day_trades.append({
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
                        logger.info(f"  Received partial page ({page_trades_count} < 10000), no more data available")
                        break
                    
                    # Update start time for next page
                    last_trade_time = ticker_trades[-1].timestamp
                    if last_trade_time <= current_start:
                        logger.warning(f"  Last trade time ({last_trade_time}) not advancing, stopping pagination")
                        break
                    
                    current_start = last_trade_time + pd.Timedelta(microseconds=1)
                    
                else:
                    consecutive_empty_pages += 1
                    logger.info(f"  Page {page_count}: No trades found (empty page {consecutive_empty_pages}/{max_consecutive_empty})")
                    
                    if consecutive_empty_pages == 1:
                        current_start += timedelta(hours=1)
                    elif consecutive_empty_pages == 2:
                        current_start += timedelta(days=1)
                    else:
                        logger.warning(f"  Too many consecutive empty pages, stopping pagination")
                        break
                    
            except Exception as e:
                logger.error(f"  Error fetching page {page_count} for {ticker} {date_str}: {str(e)}")
                time.sleep(1)
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    logger.error(f"  Too many consecutive errors, stopping pagination")
                    break
        
        # Save data for this day
        if day_trades:
            df = pd.DataFrame(day_trades)
            save_ticker_data(df, out_dir, ticker, date_str, logger)
            total_trades += len(day_trades)
            days_processed += 1
            
            # Clear memory
            del df, day_trades
            gc.collect()
        
        # Move to next day
        current_date += timedelta(days=1)
    
    logger.info(f"Completed {ticker}: {total_trades} trades across {days_processed} days")
    return True


def fetch_ticker_bars_incremental(
    ticker: str, 
    start_epoch: int, 
    end_epoch: int, 
    out_dir: str,
    logger: logging.Logger,
    progress: Dict
) -> bool:
    """Fetch minute bars for a single ticker with incremental saving."""
    if not ALPACA_V2:
        raise RuntimeError("alpaca-py not available; please install and configure.")

    logger.info(f"Fetching minute bars for {ticker}")
    
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment")

    client = StockHistoricalDataClient(api_key, api_secret)
    start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_epoch, tz=timezone.utc)
    
    # Process data day by day
    current_date = start_dt.date()
    end_date = end_dt.date()
    
    total_records = 0
    days_processed = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Check if we already have this date
        if check_existing_data(out_dir, ticker, date_str):
            logger.info(f"Data already exists for {ticker} {date_str}, skipping")
            current_date += timedelta(days=1)
            continue
        
        # Update progress
        progress["current_ticker"] = ticker
        progress["current_date"] = date_str
        save_progress(out_dir, progress)
        
        # Set time boundaries for this day
        day_start = datetime.combine(current_date, datetime.min.time(), tzinfo=timezone.utc)
        day_end = datetime.combine(current_date, datetime.max.time(), tzinfo=timezone.utc)
        
        # Adjust for actual start/end times
        if current_date == start_dt.date():
            day_start = start_dt
        if current_date == end_dt.date():
            day_end = end_dt
        
        logger.info(f"Processing {ticker} for {date_str} ({day_start} to {day_end})")
        
        try:
            # Request minute bars for this day
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=day_start,
                end=day_end,
                adjustment=None,
                feed="sip"
            )
            
            bars = client.get_stock_bars(req)
            
            if ticker in bars.data and bars.data[ticker]:
                # Convert to DataFrame
                df = bars.data[ticker].df.reset_index()
                
                if not df.empty:
                    # Standardize columns
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
                    
                    # Save data for this day
                    save_ticker_data(df, out_dir, ticker, date_str, logger)
                    total_records += len(df)
                    days_processed += 1
                    
                    # Clear memory
                    del df
                    gc.collect()
                else:
                    logger.info(f"No bars data for {ticker} on {date_str}")
            else:
                logger.info(f"No bars data for {ticker} on {date_str}")
                
        except Exception as e:
            logger.error(f"Error fetching bars for {ticker} {date_str}: {str(e)}")
            # Continue with next day
        
        # Move to next day
        current_date += timedelta(days=1)
    
    logger.info(f"Completed {ticker}: {total_records} records across {days_processed} days")
    return True


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Memory-optimized Alpaca market data collector with resume capability")
    parser.add_argument("--tickers", required=True, help="Comma-separated (e.g., IVV,VOO,VXX,SPY,UVXY)")
    parser.add_argument("--start-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--end-epoch", required=True, type=_parse_epoch)
    parser.add_argument("--out", default="parquet_out/market")
    parser.add_argument("--data-type", choices=["bars", "trades"], default="trades", 
                       help="Data type: 'trades' for per-second trade data (default), 'bars' for minute-level OHLCV")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("=" * 60)
    logger.info("Memory-Optimized Alpaca Market Data Collection Script Started")
    logger.info("=" * 60)
    
    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    logger.info(f"Processing tickers: {tickers}")
    logger.info(f"Data type: {args.data_type}")
    logger.info(f"Output directory: {args.out}")
    logger.info(f"Resume mode: {args.resume}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)
    logger.info(f"Ensured output directory exists: {args.out}")
    
    # Load progress if resuming
    progress = load_progress(args.out) if args.resume else {"completed_tickers": [], "failed_tickers": [], "current_ticker": None, "current_date": None}
    
    if args.resume:
        logger.info(f"Resuming from previous run. Completed: {progress['completed_tickers']}")
        logger.info(f"Failed: {progress['failed_tickers']}")
        if progress['current_ticker'] and progress['current_date']:
            logger.info(f"Current position: {progress['current_ticker']} at {progress['current_date']}")
    
    # Filter out completed tickers if resuming
    if args.resume and progress['completed_tickers']:
        remaining_tickers = [t for t in tickers if t not in progress['completed_tickers']]
        logger.info(f"Remaining tickers to process: {remaining_tickers}")
        tickers = remaining_tickers
    
    if not tickers:
        logger.info("All tickers already completed!")
        return
    
    try:
        # Process each ticker individually
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing ticker {i}/{len(tickers)}: {ticker}")
            
            try:
                if args.data_type == "trades":
                    success = fetch_ticker_trades_incremental(
                        ticker, args.start_epoch, args.end_epoch, args.out, logger, progress
                    )
                else:
                    success = fetch_ticker_bars_incremental(
                        ticker, args.start_epoch, args.end_epoch, args.out, logger, progress
                    )
                
                if success:
                    progress['completed_tickers'].append(ticker)
                    if ticker in progress['failed_tickers']:
                        progress['failed_tickers'].remove(ticker)
                    logger.info(f"✅ Successfully completed {ticker}")
                else:
                    progress['failed_tickers'].append(ticker)
                    logger.error(f"❌ Failed to complete {ticker}")
                
                # Clear progress tracking
                progress['current_ticker'] = None
                progress['current_date'] = None
                save_progress(args.out, progress)
                
                # Force garbage collection between tickers
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                progress['failed_tickers'].append(ticker)
                progress['current_ticker'] = ticker
                save_progress(args.out, progress)
                logger.error("Stack trace:", exc_info=True)
                continue
        
        # Final summary
        logger.info("=" * 60)
        logger.info("SCRIPT COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Successfully completed: {progress['completed_tickers']}")
        if progress['failed_tickers']:
            logger.warning(f"Failed tickers: {progress['failed_tickers']}")
            logger.warning("You can resume with --resume flag to retry failed tickers")
        
        logger.info(f"Output location: {args.out}")
        
    except Exception as e:
        logger.error(f"Script failed with error: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise


if __name__ == "__main__":
    main()
