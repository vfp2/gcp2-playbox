#!/usr/bin/env python3
import os
import argparse
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    z_window: int = 3600  # seconds
    lag_min: int = -3600
    lag_max: int = 3600
    lag_step: int = 300
    neutral_mode: str = "hold"  # hold|flat


def load_gcp2_series(root: str) -> Dict[int, pd.DataFrame]:
    """
    Load GCP2 data from partitioned parquet files.
    
    This function handles the hierarchical structure: date=YYYY-MM-DD/group_id=NNN/gcp2.parquet
    """
    logger.info(f"Loading GCP2 data from {root}")
    out: Dict[int, List[pd.DataFrame]] = {}
    
    # Walk partition folders: date=YYYY-MM-DD/group_id=NNN/gcp2.parquet
    for date_dir in sorted(os.listdir(root)):
        if not date_dir.startswith("date="):
            continue
        date_path = os.path.join(root, date_dir)
        logger.debug(f"Processing date directory: {date_dir}")
        
        for gid_dir in sorted(os.listdir(date_path)):
            if not gid_dir.startswith("group_id="):
                continue
            gid = int(gid_dir.split("=", 1)[1])
            pq = os.path.join(date_path, gid_dir, "gcp2.parquet")
            if not os.path.exists(pq):
                logger.warning(f"GCP2 file not found: {pq}")
                continue
                
            logger.debug(f"Loading GCP2 data for group_id={gid} from {date_dir}")
            df = pd.read_parquet(pq)
            out.setdefault(gid, []).append(df)
    
    # Merge data for each group_id
    merged: Dict[int, pd.DataFrame] = {}
    for gid, parts in out.items():
        logger.info(f"Merging {len(parts)} data parts for group_id={gid}")
        df = pd.concat(parts, axis=0, ignore_index=True).sort_values("time_epoch")
        df["ts"] = pd.to_datetime(df["time_epoch"], unit="s", utc=True)
        df.set_index("ts", inplace=True)
        merged[gid] = df
        logger.info(f"Group {gid}: {len(df)} records from {df.index.min()} to {df.index.max()}")
    
    logger.info(f"Loaded GCP2 data for {len(merged)} groups")
    return merged


def load_price_data(root: str, symbol: str) -> pd.DataFrame:
    """
    Load per-second price data from partitioned parquet files.
    
    This function handles the hierarchical structure: symbol=SYM/date=YYYY-MM-DD/data.parquet
    """
    logger.info(f"Loading price data for {symbol} from {root}")
    parts: List[pd.DataFrame] = []
    sym_dir = os.path.join(root, f"symbol={symbol}")
    
    if not os.path.isdir(sym_dir):
        logger.error(f"Symbol directory not found: {sym_dir}")
        return pd.DataFrame()
    
    # Count total files for progress tracking
    total_files = 0
    for date_dir in sorted(os.listdir(sym_dir)):
        if date_dir.startswith("date="):
            total_files += 1
    
    logger.info(f"Found {total_files} date directories for {symbol}")
    
    processed_files = 0
    for date_dir in sorted(os.listdir(sym_dir)):
        if not date_dir.startswith("date="):
            continue
            
        data_file = os.path.join(sym_dir, date_dir, "data.parquet")
        if os.path.exists(data_file):
            logger.debug(f"Loading price data from {date_dir}")
            df = pd.read_parquet(data_file)
            parts.append(df)
            processed_files += 1
            
            if processed_files % 10 == 0:
                logger.info(f"Processed {processed_files}/{total_files} date directories for {symbol}")
    
    if not parts:
        logger.error(f"No price data found for {symbol}")
        return pd.DataFrame()
    
    logger.info(f"Concatenating {len(parts)} price data parts for {symbol}")
    df = pd.concat(parts, axis=0, ignore_index=True)
    
    # Convert timestamp and set index
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.set_index("ts", inplace=True)
    
    # Sort by timestamp and remove duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"{symbol}: {len(df)} price records from {df.index.min()} to {df.index.max()}")
    return df


def compute_signal(gcp_df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """
    Compute trading signal based on GCP2 netvar z-score.
    
    This function calculates rolling z-scores over the specified window.
    """
    logger.info(f"Computing signal with z_window={cfg.z_window}s")
    
    # z-score over window, 1 Hz
    x = gcp_df["netvar_count_xor_alt"].astype(float)
    roll = x.rolling(cfg.z_window, min_periods=max(10, cfg.z_window // 10))
    z = (x - roll.mean()) / (roll.std(ddof=0) + 1e-9)
    signal = z.fillna(0.0)
    
    logger.info(f"Signal computed: {len(signal)} points, range [{signal.min():.3f}, {signal.max():.3f}]")
    return signal


def align_and_trade(signal: pd.Series, price_data: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Align GCP2 signals with price data and compute trading results.
    
    This function ensures proper temporal synchronization between signals and prices.
    """
    logger.info("Aligning signals with price data")
    
    # Use price column from per-second data
    px = price_data["price"].copy()
    
    # Ensure both datasets have the same frequency (1 second)
    px = px.asfreq("1s").ffill()
    sig = signal.reindex(px.index, method="ffill").fillna(0.0)
    
    logger.info(f"Aligned data: {len(px)} price points, {len(sig)} signal points")
    logger.info(f"Price range: [{px.min():.2f}, {px.max():.2f}]")
    logger.info(f"Signal range: [{sig.min():.3f}, {sig.max():.3f}]")
    
    # Simple long/short from sign of signal; neutral holds prior by default
    pos = np.sign(sig)
    if cfg.neutral_mode == "flat":
        pos = pos.where(sig != 0.0, 0.0)
    
    # PnL: position shift times returns
    ret = px.pct_change().fillna(0.0)
    pnl = (pos.shift(1).fillna(0.0) * ret)
    
    df = pd.DataFrame({
        "price": px,
        "signal": sig,
        "position": pos,
        "ret": ret,
        "pnl": pnl
    })
    
    # Calculate summary statistics
    total_pnl = pnl.sum()
    sharpe = pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252 * 24 * 3600)  # Annualized
    max_drawdown = (pnl.cumsum() - pnl.cumsum().cummax()).min()
    
    logger.info(f"Trading results: Total PnL={total_pnl:.4f}, Sharpe={sharpe:.3f}, MaxDD={max_drawdown:.4f}")
    
    return df


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Holmberg-inspired backtest scaffold over Nautilus-ready Parquet")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g., IVV,VOO,VXX,SPY,UVXY")
    parser.add_argument("--gcp2", default="parquet_out/gcp2")
    parser.add_argument("--market", default="parquet_out/market")
    parser.add_argument("--z-window", type=int, default=3600)
    parser.add_argument("--lag-min", type=int, default=-3600)
    parser.add_argument("--lag-max", type=int, default=3600)
    parser.add_argument("--lag-step", type=int, default=300)
    parser.add_argument("--neutral", choices=["hold","flat"], default="hold")
    args = parser.parse_args()

    logger.info("Starting GCP2 backtest")
    logger.info(f"Arguments: {vars(args)}")

    cfg = StrategyConfig(
        z_window=args.z_window, 
        lag_min=args.lag_min, 
        lag_max=args.lag_max, 
        lag_step=args.lag_step, 
        neutral_mode=args.neutral
    )

    # Load GCP2 data
    gcp2 = load_gcp2_series(args.gcp2)
    if not gcp2:
        logger.error("No GCP2 data found.")
        return

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    logger.info(f"Processing {len(tickers)} tickers: {tickers}")
    
    # Process each ticker
    for i, sym in enumerate(tickers, 1):
        logger.info(f"Processing ticker {i}/{len(tickers)}: {sym}")
        
        # Load price data
        price_data = load_price_data(args.market, sym)
        if price_data.empty:
            logger.error(f"No price data for {sym}, skipping")
            continue
        
        # Process each GCP2 group
        for j, (gid, gcp_df) in enumerate(gcp2.items(), 1):
            logger.info(f"Processing group {j}/{len(gcp2)}: group_id={gid}")
            
            # Compute signal
            sig = compute_signal(gcp_df, cfg)
            
            # Run backtest
            bt = align_and_trade(sig, price_data, cfg)
            
            # Save results
            out_dir = os.path.join("parquet_out", "backtests", f"symbol={sym}", f"group_id={gid}")
            os.makedirs(out_dir, exist_ok=True)
            
            output_file = os.path.join(out_dir, "backtest.parquet")
            bt.to_parquet(output_file, engine="pyarrow", compression=None)
            
            logger.info(f"Backtest completed: {sym} (group_id {gid}) -> {output_file}")
            
            # Save summary statistics
            summary = {
                "symbol": sym,
                "group_id": gid,
                "start_date": bt.index.min().isoformat(),
                "end_date": bt.index.max().isoformat(),
                "total_pnl": float(bt["pnl"].sum()),
                "sharpe_ratio": float(bt["pnl"].mean() / (bt["pnl"].std() + 1e-9) * np.sqrt(252 * 24 * 3600)),
                "max_drawdown": float((bt["pnl"].cumsum() - bt["pnl"].cumsum().cummax()).min()),
                "total_trades": int((bt["position"].diff() != 0).sum()),
                "win_rate": float((bt["pnl"] > 0).mean())
            }
            
            summary_file = os.path.join(out_dir, "summary.json")
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Summary saved: {summary_file}")
    
    logger.info("Backtest completed successfully!")


if __name__ == "__main__":
    main()


