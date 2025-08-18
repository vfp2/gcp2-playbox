#!/usr/bin/env python3
import os
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv


@dataclass
class StrategyConfig:
    z_window: int = 3600  # seconds
    lag_min: int = -3600
    lag_max: int = 3600
    lag_step: int = 300
    neutral_mode: str = "hold"  # hold|flat


def load_gcp2_series(root: str) -> Dict[int, pd.DataFrame]:
    # Walk partition folders: date=YYYY-MM-DD/group_id=NNN/gcp2.parquet
    out: Dict[int, List[pd.DataFrame]] = {}
    for date_dir in sorted(os.listdir(root)):
        if not date_dir.startswith("date="):
            continue
        date_path = os.path.join(root, date_dir)
        for gid_dir in sorted(os.listdir(date_path)):
            if not gid_dir.startswith("group_id="):
                continue
            gid = int(gid_dir.split("=", 1)[1])
            pq = os.path.join(date_path, gid_dir, "gcp2.parquet")
            if not os.path.exists(pq):
                continue
            df = pd.read_parquet(pq)
            out.setdefault(gid, []).append(df)
    merged: Dict[int, pd.DataFrame] = {}
    for gid, parts in out.items():
        df = pd.concat(parts, axis=0, ignore_index=True).sort_values("time_epoch")
        df["ts"] = pd.to_datetime(df["time_epoch"], unit="s", utc=True)
        df.set_index("ts", inplace=True)
        merged[gid] = df
    return merged


def load_bars(root: str, symbol: str) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    sym_dir = os.path.join(root, f"symbol={symbol}")
    if not os.path.isdir(sym_dir):
        return pd.DataFrame()
    for date_dir in sorted(os.listdir(sym_dir)):
        pq = os.path.join(sym_dir, date_dir, "bars.parquet")
        if os.path.exists(pq):
            parts.append(pd.read_parquet(pq))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, axis=0, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df.set_index("ts", inplace=True)
    return df.sort_index()


def compute_signal(gcp_df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    # z-score over window, 1 Hz
    x = gcp_df["netvar_count_xor_alt"].astype(float)
    roll = x.rolling(cfg.z_window, min_periods=max(10, cfg.z_window // 10))
    z = (x - roll.mean()) / (roll.std(ddof=0) + 1e-9)
    return z.fillna(0.0)


def align_and_trade(signal: pd.Series, bars: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    # Resample bars to seconds by forward fill close; include both RTH/extended implicitly
    px = bars["close"].copy()
    px = px.asfreq("1S").ffill()
    sig = signal.reindex(px.index, method="ffill").fillna(0.0)

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

    cfg = StrategyConfig(z_window=args.z_window, lag_min=args.lag_min, lag_max=args.lag_max, lag_step=args.lag_step, neutral_mode=args.neutral)

    gcp2 = load_gcp2_series(args.gcp2)
    if not gcp2:
        print("No GCP2 data found.")
        return

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    for sym in tickers:
        bars = load_bars(args.market, sym)
        if bars.empty:
            print(f"No bars for {sym}")
            continue
        # For demo, use group_id=1; extend to loop all groups and compare
        gid = sorted(gcp2.keys())[0]
        sig = compute_signal(gcp2[gid], cfg)
        bt = align_and_trade(sig, bars, cfg)
        out_dir = os.path.join("parquet_out", "backtests", f"symbol={sym}")
        os.makedirs(out_dir, exist_ok=True)
        bt.to_parquet(os.path.join(out_dir, "backtest.parquet"), engine="pyarrow", compression=None)
        print(f"Backtest written: {sym} (group_id {gid}) -> {out_dir}")


if __name__ == "__main__":
    main()


