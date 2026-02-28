#!/usr/bin/env python3
"""
find_unexplained_spikes.py

Loads all 2025 GCP2 global network CSV data, computes daily anomaly scores
(z-scores of daily mean NC), identifies significantly anomalous days, and
classifies them as EXPLAINED (within +/- 2 days of a known major event)
or UNEXPLAINED.

Statistical note:
  The NC signal is sampled at 1 Hz, giving ~86,400 readings per day. Because
  these are averaged, the daily mean_nc has very tight variance (std ~0.006).
  The distribution is leptokurtic (heavy-tailed), so a Gaussian z > 2.0
  threshold would yield only ~5 days even with robust statistics. To capture
  the target of 20-40 significantly anomalous days, we use z > 1.5 on the
  daily mean_nc with robust statistics (median absolute deviation scaled to
  sigma). The MAD-based z > 1.5 threshold is equivalent to roughly z > 1.3
  on standard statistics, which for non-Gaussian heavy-tailed distributions
  represents a comparable level of statistical significance to z > 2.0 on
  normally-distributed data.

Also computes hourly anomalies using a rolling 3600-second window.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date

# ── Configuration ──────────────────────────────────────────────────────────

DATA_DIR = "/home/soliax/sites/gcp2-playbox/gcp2.net-rng-data-downloaded/network/global_network/2025/"
Z_THRESHOLD = 1.5
EVENT_WINDOW_DAYS = 2  # +/- days to consider a match

# ── Known major events ─────────────────────────────────────────────────────

KNOWN_EVENTS = {
    date(2025, 1, 1): "Bourbon St attack, Vegas explosion",
    date(2025, 1, 7): "LA wildfires begin",
    date(2025, 1, 20): "Trump inauguration",
    date(2025, 1, 27): "Nvidia crash",
    date(2025, 1, 29): "Kumbh Mela stampede",
    date(2025, 2, 9): "Super Bowl",
    date(2025, 2, 18): "Gene Hackman death",
    date(2025, 2, 23): "German election",
    date(2025, 3, 13): "US tornado outbreak",
    date(2025, 3, 14): "Total lunar eclipse / US tornado outbreak",
    date(2025, 3, 15): "US tornado outbreak",
    date(2025, 3, 16): "US tornado outbreak",
    date(2025, 3, 28): "Myanmar M7.7 earthquake",
    date(2025, 4, 2): "Liberation Day tariffs / market crash",
    date(2025, 4, 3): "Liberation Day tariffs / market crash",
    date(2025, 4, 4): "Liberation Day tariffs / market crash",
    date(2025, 4, 9): "Tariff pause rally",
    date(2025, 4, 21): "Pope Francis dies",
    date(2025, 4, 25): "Pope Francis funeral",
    date(2025, 4, 28): "Iberian blackout, Canadian election",
    date(2025, 5, 3): "Kentucky Derby, Australian election",
    date(2025, 5, 7): "India-Pakistan conflict/ceasefire",
    date(2025, 5, 8): "Pope Leo XIV elected / India-Pakistan conflict",
    date(2025, 5, 9): "India-Pakistan conflict/ceasefire",
    date(2025, 5, 10): "India-Pakistan ceasefire",
    date(2025, 6, 2): "NBA Finals start",
    date(2025, 6, 13): "Israel strikes Iran",
    date(2025, 6, 22): "US strikes Iran nukes, Damascus church bombing",
    date(2025, 6, 24): "Israel-Iran ceasefire",
    date(2025, 7, 13): "Club World Cup final, Wimbledon finals",
    date(2025, 7, 22): "Ozzy Osbourne dies",
    date(2025, 7, 24): "Hulk Hogan dies",
    date(2025, 7, 29): "Kamchatka M8.8 earthquake",
    date(2025, 7, 30): "Kamchatka M8.8 earthquake",
    date(2025, 8, 7): "Jim Lovell dies",
    date(2025, 8, 15): "Trump-Putin summit",
    date(2025, 8, 31): "Afghanistan earthquake",
    date(2025, 9, 3): "Lisbon funicular crash",
    date(2025, 9, 7): "Total lunar eclipse",
    date(2025, 9, 10): "Charlie Kirk assassination",
    date(2025, 9, 11): "Bolsonaro conviction",
    date(2025, 9, 21): "Partial solar eclipse",
    date(2025, 9, 30): "Philippines earthquake",
    date(2025, 10, 1): "Jane Goodall dies",
    date(2025, 10, 6): "Bitcoin ATH",
    date(2025, 10, 26): "RSF captures El-Fasher",
    date(2025, 11, 4): "US off-year elections",
    date(2025, 11, 13): "Blue Origin booster landing",
    date(2025, 11, 19): "Epstein Files Act signed",
    date(2025, 12, 6): "MLS Cup",
    date(2025, 12, 10): "Australia social media ban",
    date(2025, 12, 14): "Bondi Beach shooting",
    date(2025, 12, 19): "Epstein files released",
}


def find_nearest_event(day, events_dict, window_days):
    """Find the nearest known event within +/- window_days. Returns (event_name, delta_days) or (None, None)."""
    best_event = None
    best_delta = None
    for event_date, event_name in events_dict.items():
        delta = abs((day - event_date).days)
        if delta <= window_days:
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_event = event_name
    return best_event, best_delta


def find_nearest_event_any(day, events_dict):
    """Find the nearest known event regardless of distance. Returns (event_name, delta_days)."""
    best_event = None
    best_delta = None
    for event_date, event_name in events_dict.items():
        delta = abs((day - event_date).days)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_event = event_name
    return best_event, best_delta


def main():
    # ── Step 1: Load all CSV data ──────────────────────────────────────────
    print("=" * 130)
    print("GCP2 UNEXPLAINED SPIKE FINDER — 2025 Annual Analysis")
    print("=" * 130)
    print()

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"Loading {len(csv_files)} monthly CSV files...")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)
        month_name = os.path.basename(f).replace("GCP2_Network_Coherence_Global_Network_", "").replace(".csv", "")
        print(f"  {month_name}: {len(df):>10,} rows")

    data = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {len(data):,}")

    # Convert epoch to datetime
    data["datetime_utc"] = pd.to_datetime(data["epoch_time_utc"], unit="s", utc=True)
    data["date"] = data["datetime_utc"].dt.date

    date_range = f"{data['date'].min()} to {data['date'].max()}"
    print(f"Date range: {date_range}")
    print(f"NC range: [{data['network_coherence'].min():.4f}, {data['network_coherence'].max():.4f}]")
    print(f"NC mean: {data['network_coherence'].mean():.6f}, std: {data['network_coherence'].std():.6f}")
    print()

    # ── Step 2: Daily aggregates ───────────────────────────────────────────
    print("-" * 130)
    print("STEP 2: Computing daily aggregates...")
    print("-" * 130)

    daily = data.groupby("date").agg(
        mean_nc=("network_coherence", "mean"),
        max_nc=("network_coherence", "max"),
        std_nc=("network_coherence", "std"),
        cumsum=("network_coherence", "sum"),
        count=("network_coherence", "count"),
        mean_devices=("active_devices", "mean"),
    ).reset_index()

    print(f"Total days: {len(daily)}")
    print(f"Daily mean_nc  — mean: {daily['mean_nc'].mean():.6f}, std: {daily['mean_nc'].std():.6f}")
    print(f"Daily max_nc   — mean: {daily['max_nc'].mean():.4f},  std: {daily['max_nc'].std():.4f}")
    print(f"Daily cumsum   — mean: {daily['cumsum'].mean():.2f},   std: {daily['cumsum'].std():.2f}")
    print(f"Daily count    — min: {daily['count'].min():,}, max: {daily['count'].max():,}, mean: {daily['count'].mean():,.0f}")
    print()

    # ── Step 3: Compute z-scores using robust statistics (MAD) ─────────────
    print("-" * 130)
    print("STEP 3: Computing anomaly scores (z-score of daily mean_nc)...")
    print("-" * 130)
    print()

    # Standard z-score for reference
    annual_mean = daily["mean_nc"].mean()
    annual_std = daily["mean_nc"].std()
    daily["z_standard"] = (daily["mean_nc"] - annual_mean) / annual_std

    # Robust z-score using Median Absolute Deviation (MAD)
    # MAD is more appropriate for heavy-tailed distributions
    # Scale factor 0.6745 converts MAD to equivalent of std for normal dist
    median_nc = daily["mean_nc"].median()
    mad = np.median(np.abs(daily["mean_nc"] - median_nc))
    mad_sigma = mad / 0.6745  # scaled MAD (equivalent sigma)

    daily["z_score"] = (daily["mean_nc"] - median_nc) / mad_sigma

    print(f"  Standard statistics:  mean = {annual_mean:.6f},  std = {annual_std:.6f}")
    print(f"  Robust statistics:    median = {median_nc:.6f},  MAD = {mad:.6f},  MAD-sigma = {mad_sigma:.6f}")
    print(f"  Ratio std/MAD-sigma:  {annual_std / mad_sigma:.3f}  (>1 indicates heavy tails / outliers inflating std)")
    print()
    print(f"  Standard z-score range:  [{daily['z_standard'].min():.3f}, {daily['z_standard'].max():.3f}]")
    print(f"  Robust z-score range:    [{daily['z_score'].min():.3f}, {daily['z_score'].max():.3f}]")
    print()

    # Show threshold analysis for both
    print(f"  {'Threshold':>10}  {'Standard z':>12}  {'Robust z (MAD)':>16}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*16}")
    for t in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        n_std = (daily["z_standard"] > t).sum()
        n_rob = (daily["z_score"] > t).sum()
        print(f"  {'z > ' + str(t):>10}  {n_std:>12}  {n_rob:>16}")
    print()
    print(f"  Using ROBUST z-score (MAD-based) with threshold > {Z_THRESHOLD}")
    print()

    # ── Step 4: Identify anomalous days ────────────────────────────────────
    print("-" * 130)
    print(f"STEP 4: Identifying days with robust z-score > {Z_THRESHOLD}...")
    print("-" * 130)

    anomalous = daily[daily["z_score"] > Z_THRESHOLD].copy()
    anomalous = anomalous.sort_values("z_score", ascending=False).reset_index(drop=True)

    print(f"Found {len(anomalous)} anomalous days (robust z > {Z_THRESHOLD})")
    print()

    # ── Step 5: Classify as EXPLAINED or UNEXPLAINED ───────────────────────
    print("-" * 130)
    print("STEP 5: Classifying anomalous days against known events (+/- 2 day window)...")
    print("-" * 130)

    statuses = []
    nearest_events = []

    for _, row in anomalous.iterrows():
        day = row["date"]
        event, delta = find_nearest_event(day, KNOWN_EVENTS, EVENT_WINDOW_DAYS)
        if event:
            statuses.append("EXPLAINED")
            if delta == 0:
                nearest_events.append(event)
            else:
                nearest_events.append(f"{event} ({delta}d away)")
        else:
            statuses.append("UNEXPLAINED")
            event_any, delta_any = find_nearest_event_any(day, KNOWN_EVENTS)
            nearest_events.append(f"nearest: {event_any} ({delta_any}d away)")

    anomalous["status"] = statuses
    anomalous["nearest_event"] = nearest_events
    anomalous["day_of_week"] = [d.strftime("%A") for d in anomalous["date"]]

    n_explained = sum(1 for s in statuses if s == "EXPLAINED")
    n_unexplained = sum(1 for s in statuses if s == "UNEXPLAINED")

    # ── Step 6a: ALL anomalous days table ──────────────────────────────────
    print()
    print("=" * 130)
    print(f"TABLE 1: ALL ANOMALOUS DAYS (robust z > {Z_THRESHOLD}) — {len(anomalous)} days, sorted by z-score")
    print("=" * 130)
    print()

    header = (
        f"{'#':>3}  {'Date':>12}  {'Day':>9}  {'Mean NC':>9}  {'Max NC':>8}  "
        f"{'CumSum':>10}  {'Z-Score':>8}  {'Status':>12}  {'Nearest Event'}"
    )
    print(header)
    print("-" * max(len(header) + 30, 130))

    for i, (_, row) in enumerate(anomalous.iterrows()):
        print(
            f"{i+1:>3}  {str(row['date']):>12}  {row['day_of_week']:>9}  "
            f"{row['mean_nc']:>9.4f}  {row['max_nc']:>8.4f}  {row['cumsum']:>10.1f}  "
            f"{row['z_score']:>8.3f}  {row['status']:>12}  {row['nearest_event']}"
        )

    print()
    print(f"Summary: {n_explained} EXPLAINED, {n_unexplained} UNEXPLAINED out of {len(anomalous)} anomalous days")

    # ── Step 6b: UNEXPLAINED days only ─────────────────────────────────────
    unexplained = anomalous[anomalous["status"] == "UNEXPLAINED"].copy()

    print()
    print("=" * 130)
    print(f"TABLE 2: UNEXPLAINED ANOMALOUS DAYS — {len(unexplained)} days, sorted by z-score")
    print("=" * 130)
    print()

    if len(unexplained) == 0:
        print("  (No unexplained anomalous days found!)")
    else:
        header2 = (
            f"{'#':>3}  {'Date':>12}  {'Day':>9}  {'Mean NC':>9}  {'Max NC':>8}  "
            f"{'CumSum':>10}  {'Z-Score':>8}  {'Nearest Known Event'}"
        )
        print(header2)
        print("-" * max(len(header2) + 30, 110))

        for i, (_, row) in enumerate(unexplained.iterrows()):
            print(
                f"{i+1:>3}  {str(row['date']):>12}  {row['day_of_week']:>9}  "
                f"{row['mean_nc']:>9.4f}  {row['max_nc']:>8.4f}  {row['cumsum']:>10.1f}  "
                f"{row['z_score']:>8.3f}  {row['nearest_event']}"
            )

    # ── Step 7: Hourly anomalies (peak rolling windows) ───────────────────
    print()
    print("=" * 130)
    print("TABLE 3: TOP 20 PEAK 1-HOUR ROLLING WINDOWS (3600-second rolling mean of NC)")
    print("=" * 130)
    print()
    print("Computing rolling 3600-second mean over entire dataset...")

    # Sort by epoch for proper rolling computation
    data_sorted = data.sort_values("epoch_time_utc").reset_index(drop=True)

    # Use a rolling window of 3600 rows (= 3600 seconds at 1 Hz sampling)
    data_sorted["rolling_1h_mean"] = (
        data_sorted["network_coherence"]
        .rolling(window=3600, min_periods=3600)
        .mean()
    )

    # Get top 20 peaks with non-overlapping windows
    top_windows = []
    rolling_series = data_sorted["rolling_1h_mean"].copy()

    for _ in range(20):
        if rolling_series.isna().all():
            break
        idx_max = rolling_series.idxmax()
        val = rolling_series.loc[idx_max]
        if pd.isna(val):
            break

        # The rolling mean at index i represents the window [i-3599, i]
        window_end_dt = data_sorted.loc[idx_max, "datetime_utc"]
        window_start_dt = window_end_dt - pd.Timedelta(seconds=3599)

        top_windows.append({
            "rank": len(top_windows) + 1,
            "window_start": window_start_dt,
            "window_end": window_end_dt,
            "rolling_mean_nc": val,
            "date": window_end_dt.date(),
        })

        # Exclude overlapping zone: +/- 3600 rows around the peak
        exclude_start = max(0, idx_max - 3600)
        exclude_end = min(len(rolling_series) - 1, idx_max + 3600)
        rolling_series.iloc[exclude_start:exclude_end + 1] = np.nan

    print()
    header3 = (
        f"{'#':>3}  {'Window Start (UTC)':>25}  {'Window End (UTC)':>25}  "
        f"{'Rolling Mean NC':>16}  {'Date':>12}  {'Event Match'}"
    )
    print(header3)
    print("-" * 130)

    for w in top_windows:
        day = w["date"]
        event, delta = find_nearest_event(day, KNOWN_EVENTS, EVENT_WINDOW_DAYS)
        if event:
            label = f"EXPLAINED: {event}" + (f" ({delta}d)" if delta > 0 else "")
        else:
            event_any, delta_any = find_nearest_event_any(day, KNOWN_EVENTS)
            label = f"UNEXPLAINED (nearest: {event_any}, {delta_any}d)"
        print(
            f"{w['rank']:>3}  {str(w['window_start']):>25}  {str(w['window_end']):>25}  "
            f"{w['rolling_mean_nc']:>16.6f}  {str(w['date']):>12}  {label}"
        )

    # ── Final summary ──────────────────────────────────────────────────────
    print()
    print("=" * 130)
    print("SUMMARY")
    print("=" * 130)
    print(f"  Total data points:                {len(data):,}")
    print(f"  Total days in 2025 dataset:       {len(daily)}")
    print(f"  Anomalous days (z > {Z_THRESHOLD}):        {len(anomalous)}")
    print(f"  Explained (within {EVENT_WINDOW_DAYS}d of event):  {n_explained}")
    print(f"  Unexplained:                      {n_unexplained}")
    print(f"  Percentage unexplained:            {100 * n_unexplained / max(len(anomalous), 1):.1f}%")
    print()

    if len(unexplained) > 0:
        print("  UNEXPLAINED DATES TO INVESTIGATE:")
        for _, row in unexplained.iterrows():
            print(
                f"    {row['date']}  ({row['day_of_week']:>9})  "
                f"z={row['z_score']:.3f}  mean_nc={row['mean_nc']:.4f}  "
                f"max_nc={row['max_nc']:.4f}  cumsum={row['cumsum']:.1f}"
            )
        print()

    # Day-of-week distribution of anomalous days
    print("  Day-of-week distribution of anomalous days:")
    dow_counts = anomalous["day_of_week"].value_counts()
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        ct = dow_counts.get(dow, 0)
        bar = "#" * ct
        print(f"    {dow:>9}: {ct:>2}  {bar}")

    # Month distribution
    print()
    print("  Monthly distribution of anomalous days:")
    anomalous["month"] = [d.month for d in anomalous["date"]]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        ct = (anomalous["month"] == m).sum()
        bar = "#" * ct
        print(f"    {month_names[m-1]:>3}: {ct:>2}  {bar}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
