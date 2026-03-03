#!/usr/bin/env python3
"""
Experiment 9: Coherence Spike Hunt

1. Fetch all missing GCP2 Global Network data from the API
2. Find the ~2-hour network coherence spike between Feb 28 - Mar 2, 2026
3. Search all historical data for spikes of similar or longer duration
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add data directory to path for imports
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "gcp2.net-rng-data-downloaded"
sys.path.insert(0, str(DATA_DIR))

from gcp2_api_client import GCP2APIClient, CLUSTER_NAME_TO_FOLDER

OUTPUT_DIR = Path(__file__).resolve().parent
NETWORK_DIR = DATA_DIR / "network"

# ─── Step 1: Fetch missing data from API ─────────────────────────────────────

def get_local_latest_ts(folder_name: str) -> float:
    """Get the latest epoch timestamp from local CSV files."""
    cluster_dir = NETWORK_DIR / folder_name
    if not cluster_dir.exists():
        return 0.0

    latest_ts = 0.0
    for year_dir in sorted(cluster_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for csv_file in sorted(year_dir.glob("*.csv")):
            if ".csv.zip" in csv_file.name:
                continue
            try:
                # Read only last line efficiently
                df = pd.read_csv(csv_file, usecols=["epoch_time_utc"])
                if not df.empty:
                    file_max = df["epoch_time_utc"].max()
                    latest_ts = max(latest_ts, file_max)
            except Exception as e:
                print(f"  Warning: could not read {csv_file.name}: {e}")
    return latest_ts


def fetch_missing_data(cluster_name: str, folder_name: str) -> pd.DataFrame:
    """Fetch data from API that isn't already saved locally."""
    local_latest = get_local_latest_ts(folder_name)
    if local_latest == 0.0:
        print(f"  No local data found for {cluster_name}")
        return pd.DataFrame()

    local_dt = datetime.fromtimestamp(local_latest, tz=timezone.utc)
    print(f"  Local data ends at: {local_dt.isoformat()}")

    start_ts = local_latest + 1  # one second after last local record
    end_ts = time.time()  # now
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    gap_hours = (end_ts - start_ts) / 3600
    print(f"  Gap to fill: {gap_hours:.1f} hours ({local_dt} → {end_dt})")

    if gap_hours < 0.1:
        print(f"  Already up to date!")
        return pd.DataFrame()

    client = GCP2APIClient()

    def progress(count, msg):
        print(f"    [{count:,} records] {msg}")

    df = client.get_cluster_history(
        cluster_name,
        start_ts=start_ts,
        end_ts=end_ts,
        progress_callback=progress
    )

    if not df.empty:
        first_dt = datetime.fromtimestamp(df["epoch_time_utc"].min(), tz=timezone.utc)
        last_dt = datetime.fromtimestamp(df["epoch_time_utc"].max(), tz=timezone.utc)
        print(f"  Fetched {len(df):,} records: {first_dt} → {last_dt}")
    else:
        print(f"  No new data returned from API")

    return df


def save_fetched_data(df: pd.DataFrame, folder_name: str):
    """Save fetched data to monthly CSV files, matching existing format."""
    if df.empty:
        return

    df["datetime"] = pd.to_datetime(df["epoch_time_utc"], unit="s", utc=True)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month

    for (year, month), group in df.groupby(["year", "month"]):
        year_dir = NETWORK_DIR / folder_name / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)

        # Build filename matching existing pattern
        display_name = folder_name.replace("_", " ").title().replace(" ", "_")
        # Find existing file to match naming
        existing = list(year_dir.glob(f"*_{year}_{month:02d}.csv"))
        if existing:
            csv_path = existing[0]
            # Append to existing file
            existing_df = pd.read_csv(csv_path)
            save_cols = ["epoch_time_utc", "network_coherence", "active_devices"]
            combined = pd.concat([existing_df, group[save_cols]], ignore_index=True)
            combined = combined.drop_duplicates(subset=["epoch_time_utc"], keep="last")
            combined = combined.sort_values("epoch_time_utc").reset_index(drop=True)
            combined.to_csv(csv_path, index=False)
            print(f"  Updated {csv_path.name}: {len(combined):,} total records")
        else:
            # Create new file - derive name from cluster folder
            # Map folder names to the naming convention used in existing files
            name_parts = folder_name.split("_")
            if folder_name == "global_network":
                file_display = "Global_Network"
            elif folder_name.startswith("cluster_"):
                file_display = "_".join(w.title() for w in name_parts)
            elif folder_name.startswith("research_tower_"):
                file_display = "_".join(w.title() for w in name_parts)
            else:
                file_display = "_".join(w.title() for w in name_parts)

            filename = f"GCP2_Network_Coherence_{file_display}_{year}_{month:02d}.csv"
            csv_path = year_dir / filename
            save_cols = ["epoch_time_utc", "network_coherence", "active_devices"]
            group[save_cols].sort_values("epoch_time_utc").to_csv(csv_path, index=False)
            print(f"  Created {csv_path.name}: {len(group):,} records")


def fetch_all_clusters():
    """Fetch missing data for all clusters."""
    client = GCP2APIClient()
    clusters = client.list_clusters()

    print(f"\n{'='*70}")
    print(f"STEP 1: FETCHING MISSING DATA FOR ALL CLUSTERS")
    print(f"{'='*70}")
    print(f"Found {len(clusters)} clusters from API\n")

    for cluster in sorted(clusters, key=lambda c: c["name"]):
        name = cluster["name"]
        folder = CLUSTER_NAME_TO_FOLDER.get(name)
        if not folder:
            # Try to derive folder name
            folder = name.lower().replace(" ", "_").replace("-", "_")
        print(f"\n--- {name} (folder: {folder}) ---")
        df = fetch_missing_data(name, folder)
        if not df.empty:
            save_fetched_data(df, folder)


# ─── Step 2: Find the spike between Feb 28 - Mar 2 ──────────────────────────

def load_global_network(start_ts: float, end_ts: float) -> pd.DataFrame:
    """Load Global Network data from local CSVs for a time range."""
    folder = NETWORK_DIR / "global_network"
    frames = []

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    for year_dir in sorted(folder.iterdir()):
        if not year_dir.is_dir():
            continue
        for csv_file in sorted(year_dir.glob("*.csv")):
            if ".csv.zip" in csv_file.name:
                continue
            try:
                df = pd.read_csv(csv_file)
                mask = (df["epoch_time_utc"] >= start_ts) & (df["epoch_time_utc"] <= end_ts)
                filtered = df[mask]
                if not filtered.empty:
                    frames.append(filtered)
            except Exception as e:
                print(f"  Warning: {csv_file.name}: {e}")

    if not frames:
        return pd.DataFrame(columns=["epoch_time_utc", "network_coherence", "active_devices"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["epoch_time_utc"])
    return combined.sort_values("epoch_time_utc").reset_index(drop=True)


def find_spike_in_range():
    """Find the ~2 hour coherence spike between Feb 28 - Mar 2, 2026."""
    print(f"\n{'='*70}")
    print(f"STEP 2: FINDING THE COHERENCE SPIKE (Feb 28 - Mar 2, 2026)")
    print(f"{'='*70}")

    start = datetime(2026, 2, 28, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    end = datetime(2026, 3, 2, 23, 59, 59, tzinfo=timezone.utc).timestamp()

    print(f"\nLoading Global Network data for Feb 28 - Mar 2, 2026...")
    df = load_global_network(start, end)
    print(f"Loaded {len(df):,} records")

    if df.empty:
        print("ERROR: No data loaded!")
        return None

    df["datetime"] = pd.to_datetime(df["epoch_time_utc"], unit="s", utc=True)

    # Method: Use rolling cumulative sum to detect sustained positive coherence
    # A "spike" = period where cumsum is climbing steeply = sustained positive NC values
    # We'll use multiple window sizes to be thorough

    nc = df["network_coherence"].values
    timestamps = df["epoch_time_utc"].values

    # Rolling sum over various windows (in seconds)
    for window_label, window_sec in [("5 min", 300), ("15 min", 900), ("30 min", 1800), ("1 hour", 3600)]:
        rolling_sum = pd.Series(nc).rolling(window=window_sec, min_periods=window_sec//2).sum()
        # Normalize by sqrt of window size for comparable Z-scores
        rolling_z = rolling_sum / np.sqrt(window_sec)
        max_idx = rolling_z.idxmax()
        max_z = rolling_z.iloc[max_idx]
        max_ts = timestamps[max_idx]
        max_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc)
        print(f"  {window_label:>8s} window: max rolling Z = {max_z:.3f} at {max_dt}")

    # Primary detection: Use 15-minute rolling mean to find sustained positive regions
    print(f"\n--- Detecting sustained positive coherence episodes ---")

    # 15-min rolling mean of NC
    window = 900  # 15 minutes
    rolling_mean = pd.Series(nc).rolling(window=window, min_periods=window//2).mean()

    # A "spike" is where rolling mean exceeds a threshold
    # NC has mean 0 and variance ~2, so rolling mean of 900 samples has std = sqrt(2/900) ≈ 0.047
    # A 3-sigma threshold would be ~0.14
    expected_std = np.sqrt(2.0 / window)
    threshold = 3.0 * expected_std  # 3-sigma

    above = rolling_mean > threshold
    df["above_threshold"] = above.values
    df["rolling_mean"] = rolling_mean.values

    # Find contiguous runs of above-threshold
    runs = []
    in_run = False
    run_start = 0
    for i in range(len(df)):
        if above.iloc[i] and not in_run:
            in_run = True
            run_start = i
        elif not above.iloc[i] and in_run:
            in_run = False
            run_end = i - 1
            duration_sec = timestamps[run_end] - timestamps[run_start]
            runs.append((run_start, run_end, duration_sec))
    if in_run:
        run_end = len(df) - 1
        duration_sec = timestamps[run_end] - timestamps[run_start]
        runs.append((run_start, run_end, duration_sec))

    # Sort by duration, find ones near 2 hours
    runs.sort(key=lambda x: -x[2])

    print(f"\nFound {len(runs)} episodes above 3-sigma threshold ({threshold:.4f})")
    print(f"\nTop 10 longest sustained positive coherence episodes:")
    print(f"{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'Duration':>12s}  {'Peak 15m Mean':>14s}  {'CumSum':>10s}")
    print("-" * 95)

    spike_details = []
    for rank, (rs, re, dur) in enumerate(runs[:10], 1):
        start_dt = datetime.fromtimestamp(timestamps[rs], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(timestamps[re], tz=timezone.utc)
        dur_str = f"{dur/3600:.1f}h" if dur >= 3600 else f"{dur/60:.0f}m"
        peak_mean = rolling_mean.iloc[rs:re+1].max()
        cumsum_run = nc[rs:re+1].sum()
        print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {dur_str:>12s}  {peak_mean:>14.4f}  {cumsum_run:>10.1f}")
        spike_details.append({
            "rank": rank,
            "start_ts": timestamps[rs],
            "end_ts": timestamps[re],
            "start_utc": start_dt.isoformat(),
            "end_utc": end_dt.isoformat(),
            "duration_sec": dur,
            "peak_15m_mean": peak_mean,
            "cumsum": cumsum_run
        })

    # Also look at it from a cumsum perspective: steepest 2-hour climb
    print(f"\n--- Steepest cumulative sum climbs (2-hour windows) ---")
    window_2h = 7200
    cumsum = np.cumsum(nc)
    if len(cumsum) > window_2h:
        climb_2h = cumsum[window_2h:] - cumsum[:-window_2h]
        best_idx = np.argmax(climb_2h) + window_2h  # end of window
        best_start_idx = best_idx - window_2h
        best_climb = climb_2h[best_idx - window_2h]
        best_start_dt = datetime.fromtimestamp(timestamps[best_start_idx], tz=timezone.utc)
        best_end_dt = datetime.fromtimestamp(timestamps[best_idx], tz=timezone.utc)
        print(f"  Steepest 2h climb: {best_climb:.1f} cumulative units")
        print(f"  From: {best_start_dt}")
        print(f"  To:   {best_end_dt}")
        print(f"  Mean NC during this window: {nc[best_start_idx:best_idx].mean():.4f}")
        print(f"  Z-equivalent: {best_climb / np.sqrt(2.0 * window_2h):.3f}")

    return spike_details


# ─── Step 3: Search all history for spikes of similar+ duration ──────────────

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
                print(f"  Loaded {csv_file.name}: {len(df):,} records")
            except Exception as e:
                print(f"  Warning: {csv_file.name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["epoch_time_utc"])
    return combined.sort_values("epoch_time_utc").reset_index(drop=True)


def search_all_history(target_duration_sec: float):
    """Search all Global Network history for spikes of target_duration or longer."""
    print(f"\n{'='*70}")
    print(f"STEP 3: SEARCHING ALL HISTORY FOR SPIKES >= {target_duration_sec/3600:.1f} HOURS")
    print(f"{'='*70}")

    print(f"\nLoading all Global Network data...")
    df = load_all_global_network()
    print(f"\nTotal: {len(df):,} records")

    if df.empty:
        print("No data!")
        return

    first_dt = datetime.fromtimestamp(df["epoch_time_utc"].min(), tz=timezone.utc)
    last_dt = datetime.fromtimestamp(df["epoch_time_utc"].max(), tz=timezone.utc)
    total_days = (last_dt - first_dt).total_seconds() / 86400
    print(f"Time range: {first_dt.date()} to {last_dt.date()} ({total_days:.0f} days)")

    nc = df["network_coherence"].values
    timestamps = df["epoch_time_utc"].values

    # Use same detection method: 15-min rolling mean > 3-sigma
    window = 900
    expected_std = np.sqrt(2.0 / window)
    threshold = 3.0 * expected_std

    print(f"\nComputing 15-min rolling mean across {len(nc):,} data points...")
    rolling_mean = pd.Series(nc).rolling(window=window, min_periods=window//2).mean()

    above = rolling_mean > threshold

    # Find contiguous runs
    print("Finding contiguous above-threshold episodes...")
    runs = []
    in_run = False
    run_start = 0
    for i in range(len(df)):
        if above.iloc[i] and not in_run:
            in_run = True
            run_start = i
        elif not above.iloc[i] and in_run:
            in_run = False
            run_end = i - 1
            duration_sec = timestamps[run_end] - timestamps[run_start]
            if duration_sec >= target_duration_sec:
                peak_mean = rolling_mean.iloc[run_start:run_end+1].max()
                cumsum_run = nc[run_start:run_end+1].sum()
                runs.append((run_start, run_end, duration_sec, peak_mean, cumsum_run))
    if in_run:
        run_end = len(df) - 1
        duration_sec = timestamps[run_end] - timestamps[run_start]
        if duration_sec >= target_duration_sec:
            peak_mean = rolling_mean.iloc[run_start:run_end+1].max()
            cumsum_run = nc[run_start:run_end+1].sum()
            runs.append((run_start, run_end, duration_sec, peak_mean, cumsum_run))

    runs.sort(key=lambda x: -x[2])

    print(f"\nFound {len(runs)} episodes >= {target_duration_sec/3600:.1f} hours in all history")
    print(f"\n{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'Duration':>10s}  {'Peak 15m Mean':>14s}  {'CumSum':>10s}")
    print("-" * 90)

    for rank, (rs, re, dur, peak, cs) in enumerate(runs, 1):
        start_dt = datetime.fromtimestamp(timestamps[rs], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(timestamps[re], tz=timezone.utc)
        dur_str = f"{dur/3600:.1f}h"
        print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {dur_str:>10s}  {peak:>14.4f}  {cs:>10.1f}")

    if not runs:
        print("  (none found)")
        # Try lower threshold
        print(f"\nRetrying with 2.5-sigma threshold...")
        threshold_lower = 2.5 * expected_std
        above_lower = rolling_mean > threshold_lower
        runs_lower = []
        in_run = False
        for i in range(len(df)):
            if above_lower.iloc[i] and not in_run:
                in_run = True
                run_start = i
            elif not above_lower.iloc[i] and in_run:
                in_run = False
                run_end = i - 1
                dur = timestamps[run_end] - timestamps[run_start]
                if dur >= target_duration_sec:
                    peak = rolling_mean.iloc[run_start:run_end+1].max()
                    cs = nc[run_start:run_end+1].sum()
                    runs_lower.append((run_start, run_end, dur, peak, cs))
        if in_run:
            run_end = len(df) - 1
            dur = timestamps[run_end] - timestamps[run_start]
            if dur >= target_duration_sec:
                peak = rolling_mean.iloc[run_start:run_end+1].max()
                cs = nc[run_start:run_end+1].sum()
                runs_lower.append((run_start, run_end, dur, peak, cs))

        runs_lower.sort(key=lambda x: -x[2])
        print(f"Found {len(runs_lower)} episodes at 2.5-sigma")
        for rank, (rs, re, dur, peak, cs) in enumerate(runs_lower[:20], 1):
            start_dt = datetime.fromtimestamp(timestamps[rs], tz=timezone.utc)
            end_dt = datetime.fromtimestamp(timestamps[re], tz=timezone.utc)
            dur_str = f"{dur/3600:.1f}h"
            print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {dur_str:>10s}  {peak:>14.4f}  {cs:>10.1f}")

    # Also show top 20 longest at any threshold for context
    print(f"\n--- For context: Top 20 longest positive coherence episodes at 3-sigma (any duration) ---")
    all_runs = []
    in_run = False
    for i in range(len(df)):
        if above.iloc[i] and not in_run:
            in_run = True
            run_start = i
        elif not above.iloc[i] and in_run:
            in_run = False
            run_end = i - 1
            dur = timestamps[run_end] - timestamps[run_start]
            if dur >= 300:  # at least 5 minutes
                peak = rolling_mean.iloc[run_start:run_end+1].max()
                cs = nc[run_start:run_end+1].sum()
                all_runs.append((run_start, run_end, dur, peak, cs))
    if in_run:
        run_end = len(df) - 1
        dur = timestamps[run_end] - timestamps[run_start]
        if dur >= 300:
            peak = rolling_mean.iloc[run_start:run_end+1].max()
            cs = nc[run_start:run_end+1].sum()
            all_runs.append((run_start, run_end, dur, peak, cs))

    all_runs.sort(key=lambda x: -x[2])

    print(f"\n{'#':>3s}  {'Start (UTC)':>22s}  {'End (UTC)':>22s}  {'Duration':>10s}  {'Peak 15m Mean':>14s}  {'CumSum':>10s}")
    print("-" * 90)
    for rank, (rs, re, dur, peak, cs) in enumerate(all_runs[:20], 1):
        start_dt = datetime.fromtimestamp(timestamps[rs], tz=timezone.utc)
        end_dt = datetime.fromtimestamp(timestamps[re], tz=timezone.utc)
        if dur >= 3600:
            dur_str = f"{dur/3600:.1f}h"
        else:
            dur_str = f"{dur/60:.0f}m"
        print(f"{rank:3d}  {start_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {end_dt.strftime('%Y-%m-%d %H:%M:%S'):>22s}  {dur_str:>10s}  {peak:>14.4f}  {cs:>10.1f}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 9: Coherence Spike Hunt")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching new data")
    parser.add_argument("--skip-search", action="store_true", help="Skip historical search")
    args = parser.parse_args()

    # Step 1: Fetch missing data
    if not args.skip_fetch:
        fetch_all_clusters()

    # Step 2: Find the spike
    spike_details = find_spike_in_range()

    # Step 3: Search all history
    if not args.skip_search and spike_details:
        # Use the duration of the top spike as target
        target = spike_details[0]["duration_sec"]
        search_all_history(target)
    elif not args.skip_search:
        # Default to 2 hours
        search_all_history(7200)
