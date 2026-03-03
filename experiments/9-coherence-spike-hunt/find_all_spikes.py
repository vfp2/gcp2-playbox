#!/usr/bin/env python3
"""
Find all significant coherence spikes in the entire Global Network history.
Looks for sustained positive coherence episodes using multiple detection methods.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "gcp2.net-rng-data-downloaded"
NETWORK_DIR = DATA_DIR / "network"


def load_all_global_network() -> pd.DataFrame:
    """Load ALL Global Network data from local CSVs."""
    folder = NETWORK_DIR / "global_network"
    frames = []

    for year_dir in sorted(folder.iterdir()):
        if not year_dir.is_dir():
            continue
        for csv_file in sorted(year_dir.glob("*.csv")):
            if ".csv.zip" in csv_file.name:
                continue
            try:
                df = pd.read_csv(csv_file)
                frames.append(df)
                first = datetime.fromtimestamp(df["epoch_time_utc"].iloc[0], tz=timezone.utc)
                last = datetime.fromtimestamp(df["epoch_time_utc"].iloc[-1], tz=timezone.utc)
                print(f"  {csv_file.name}: {len(df):>10,} records  ({first.date()} to {last.date()})")
            except Exception as e:
                print(f"  {csv_file.name}: ERROR - {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["epoch_time_utc"])
    return combined.sort_values("epoch_time_utc").reset_index(drop=True)


def detect_spikes_rolling_mean(nc, timestamps, window_sec=900, sigma=3.0, min_duration_sec=300):
    """Detect sustained positive coherence episodes using rolling mean threshold."""
    expected_std = np.sqrt(2.0 / window_sec)
    threshold = sigma * expected_std

    rolling_mean = pd.Series(nc).rolling(window=window_sec, min_periods=window_sec // 2).mean()
    above = rolling_mean > threshold

    runs = []
    in_run = False
    run_start = 0

    for i in range(len(nc)):
        if above.iloc[i] and not in_run:
            in_run = True
            run_start = i
        elif not above.iloc[i] and in_run:
            in_run = False
            run_end = i - 1
            dur = timestamps[run_end] - timestamps[run_start]
            if dur >= min_duration_sec:
                peak = float(rolling_mean.iloc[run_start:run_end + 1].max())
                cs = float(nc[run_start:run_end + 1].sum())
                mean_nc = float(nc[run_start:run_end + 1].mean())
                runs.append({
                    "start_idx": run_start,
                    "end_idx": run_end,
                    "start_ts": timestamps[run_start],
                    "end_ts": timestamps[run_end],
                    "duration_sec": dur,
                    "peak_rolling_mean": peak,
                    "cumsum": cs,
                    "mean_nc": mean_nc,
                })
    if in_run:
        run_end = len(nc) - 1
        dur = timestamps[run_end] - timestamps[run_start]
        if dur >= min_duration_sec:
            peak = float(rolling_mean.iloc[run_start:run_end + 1].max())
            cs = float(nc[run_start:run_end + 1].sum())
            mean_nc = float(nc[run_start:run_end + 1].mean())
            runs.append({
                "start_idx": run_start,
                "end_idx": run_end,
                "start_ts": timestamps[run_start],
                "end_ts": timestamps[run_end],
                "duration_sec": dur,
                "peak_rolling_mean": peak,
                "cumsum": cs,
                "mean_nc": mean_nc,
            })

    return sorted(runs, key=lambda x: -x["duration_sec"])


def detect_steepest_climbs(nc, timestamps, window_sec=7200):
    """Find the steepest cumulative sum climbs over a fixed window."""
    cumsum = np.cumsum(nc)
    if len(cumsum) <= window_sec:
        return []

    # Compute climb over window
    climb = cumsum[window_sec:] - cumsum[:-window_sec]

    # Get top climbs (non-overlapping)
    results = []
    used = np.zeros(len(climb), dtype=bool)

    for _ in range(50):  # Find up to 50
        masked = np.where(used, -np.inf, climb)
        best_end_offset = np.argmax(masked)
        if masked[best_end_offset] <= 0:
            break

        best_end_idx = best_end_offset + window_sec
        best_start_idx = best_end_offset

        climb_val = climb[best_end_offset]
        mean_nc = nc[best_start_idx:best_end_idx].mean()

        results.append({
            "start_ts": timestamps[best_start_idx],
            "end_ts": timestamps[best_end_idx],
            "start_idx": best_start_idx,
            "end_idx": best_end_idx,
            "climb": float(climb_val),
            "mean_nc": float(mean_nc),
            "z_equiv": float(climb_val / np.sqrt(2.0 * window_sec)),
        })

        # Mark overlapping region as used
        overlap_start = max(0, best_end_offset - window_sec)
        overlap_end = min(len(used), best_end_offset + window_sec)
        used[overlap_start:overlap_end] = True

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("GLOBAL NETWORK COHERENCE SPIKE ANALYSIS")
    print("=" * 80)

    print("\nLoading all Global Network data...")
    df = load_all_global_network()

    if df.empty:
        print("No data found!")
        sys.exit(1)

    nc = df["network_coherence"].values
    timestamps = df["epoch_time_utc"].values

    first_dt = datetime.fromtimestamp(timestamps[0], tz=timezone.utc)
    last_dt = datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)
    total_days = (last_dt - first_dt).total_seconds() / 86400

    print(f"\nTotal: {len(df):,} records")
    print(f"Range: {first_dt} to {last_dt} ({total_days:.0f} days)")
    print(f"Mean NC: {nc.mean():.6f}")
    print(f"Std NC: {nc.std():.4f}")

    # ─── Method 1: Sustained positive coherence (rolling mean > 3-sigma) ─────
    print(f"\n{'='*80}")
    print("METHOD 1: SUSTAINED POSITIVE COHERENCE (15-min rolling mean > 3-sigma)")
    print("=" * 80)

    runs = detect_spikes_rolling_mean(nc, timestamps, window_sec=900, sigma=3.0, min_duration_sec=300)

    print(f"\nFound {len(runs)} episodes (>5 min duration)")

    # Show all >= 1 hour
    long_runs = [r for r in runs if r["duration_sec"] >= 3600]
    print(f"Of which {len(long_runs)} are >= 1 hour")

    print(f"\n--- Top 30 longest episodes ---")
    print(f"{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'Duration':>10s}  {'Peak Mean':>10s}  {'Avg NC':>8s}  {'CumSum':>10s}")
    print("-" * 100)

    for rank, r in enumerate(runs[:30], 1):
        start_dt = datetime.fromtimestamp(r["start_ts"], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(r["end_ts"], tz=timezone.utc)
        if r["duration_sec"] >= 3600:
            dur_str = f"{r['duration_sec'] / 3600:.1f}h"
        else:
            dur_str = f"{r['duration_sec'] / 60:.0f}m"
        print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {dur_str:>10s}  {r['peak_rolling_mean']:>10.5f}  {r['mean_nc']:>8.4f}  {r['cumsum']:>10.1f}")

    # ─── Method 2: Steepest 2-hour cumsum climbs ─────────────────────────────
    print(f"\n{'='*80}")
    print("METHOD 2: STEEPEST 2-HOUR CUMULATIVE SUM CLIMBS")
    print("=" * 80)

    climbs = detect_steepest_climbs(nc, timestamps, window_sec=7200)

    print(f"\n--- Top 20 steepest 2-hour climbs (non-overlapping) ---")
    print(f"{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'CumSum Climb':>13s}  {'Mean NC':>8s}  {'Z-equiv':>8s}")
    print("-" * 85)

    for rank, c in enumerate(climbs[:20], 1):
        start_dt = datetime.fromtimestamp(c["start_ts"], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(c["end_ts"], tz=timezone.utc)
        print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {c['climb']:>13.1f}  {c['mean_nc']:>8.4f}  {c['z_equiv']:>8.3f}")

    # ─── Method 3: Steepest 1-hour climbs ────────────────────────────────────
    print(f"\n{'='*80}")
    print("METHOD 3: STEEPEST 1-HOUR CUMULATIVE SUM CLIMBS")
    print("=" * 80)

    climbs_1h = detect_steepest_climbs(nc, timestamps, window_sec=3600)

    print(f"\n--- Top 20 steepest 1-hour climbs (non-overlapping) ---")
    print(f"{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'CumSum Climb':>13s}  {'Mean NC':>8s}  {'Z-equiv':>8s}")
    print("-" * 85)

    for rank, c in enumerate(climbs_1h[:20], 1):
        start_dt = datetime.fromtimestamp(c["start_ts"], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(c["end_ts"], tz=timezone.utc)
        print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {c['climb']:>13.1f}  {c['mean_nc']:>8.4f}  {c['z_equiv']:>8.3f}")

    # ─── Focus on last 2 weeks (Feb 13-26) ──────────────────────────────────
    print(f"\n{'='*80}")
    print("FOCUS: LAST 2 WEEKS (Feb 13 - Feb 26, 2026)")
    print("=" * 80)

    feb13 = datetime(2026, 2, 13, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    feb27 = datetime(2026, 2, 27, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    mask = (timestamps >= feb13) & (timestamps < feb27)
    recent_nc = nc[mask]
    recent_ts = timestamps[mask]

    if len(recent_nc) > 0:
        recent_runs = detect_spikes_rolling_mean(recent_nc, recent_ts, window_sec=900, sigma=3.0, min_duration_sec=300)
        print(f"\nFound {len(recent_runs)} episodes in last 2 weeks")
        print(f"\n--- All episodes (longest first) ---")
        print(f"{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'Duration':>10s}  {'Peak Mean':>10s}  {'Avg NC':>8s}  {'CumSum':>10s}")
        print("-" * 100)
        for rank, r in enumerate(recent_runs[:20], 1):
            start_dt = datetime.fromtimestamp(r["start_ts"], tz=timezone.utc)
            end_dt = datetime.fromtimestamp(r["end_ts"], tz=timezone.utc)
            if r["duration_sec"] >= 3600:
                dur_str = f"{r['duration_sec'] / 3600:.1f}h"
            else:
                dur_str = f"{r['duration_sec'] / 60:.0f}m"
            print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {dur_str:>10s}  {r['peak_rolling_mean']:>10.5f}  {r['mean_nc']:>8.4f}  {r['cumsum']:>10.1f}")

        # Also show steepest 2h climbs in recent period
        recent_climbs = detect_steepest_climbs(recent_nc, recent_ts, window_sec=7200)
        print(f"\n--- Top 10 steepest 2-hour climbs in last 2 weeks ---")
        print(f"{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'CumSum Climb':>13s}  {'Mean NC':>8s}  {'Z-equiv':>8s}")
        print("-" * 85)
        for rank, c in enumerate(recent_climbs[:10], 1):
            start_dt = datetime.fromtimestamp(c["start_ts"], tz=timezone.utc)
            end_dt = datetime.fromtimestamp(c["end_ts"], tz=timezone.utc)
            print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {c['climb']:>13.1f}  {c['mean_nc']:>8.4f}  {c['z_equiv']:>8.3f}")
