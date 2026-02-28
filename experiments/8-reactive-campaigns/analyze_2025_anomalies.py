"""
GCP2 Global Network 2025 -- Top 30 Most Anomalous Days
=======================================================
Processes ~31M rows of 1-second network_coherence data, month by month,
computing daily aggregates and ranking days by anomaly strength.
"""

import glob
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

DATA_DIR = "/home/soliax/sites/gcp2-playbox/gcp2.net-rng-data-downloaded/network/global_network/2025"
PATTERN = f"{DATA_DIR}/GCP2_Network_Coherence_Global_Network_2025_*.csv"
TOP_N = 30

# Rolling window size for 1-hour Z-score (3600 seconds at 1 Hz)
ROLLING_WINDOW = 3600

t0 = time.time()

# ---------------------------------------------------------------------------
# Pass 1: stream month-by-month, compute per-day statistics
# ---------------------------------------------------------------------------
# We will also need the global mean/std of network_coherence for the Z-score
# calculation. Do a two-pass approach:
#   Pass 1a: collect running sums to get global mean/std
#   Pass 1b: compute daily rolling-Z max using the global stats
# Since daily max rolling Z depends on a rolling window *within* each day,
# we can approximate it by processing each day individually with the global
# mean/std once known.

files = sorted(glob.glob(PATTERN))
print(f"Found {len(files)} monthly CSV files.\n")

# ---- Pass 1a: gather daily aggregates + running sums for global stats ----
daily_records = []          # list of dicts, one per day
global_sum = 0.0
global_sum_sq = 0.0
global_n = 0

for fpath in files:
    fname = fpath.split("/")[-1]
    print(f"  Loading {fname} ...", end=" ", flush=True)
    t1 = time.time()

    df = pd.read_csv(fpath, usecols=["epoch_time_utc", "network_coherence"])
    print(f"{len(df):,} rows in {time.time()-t1:.1f}s", flush=True)

    nc = df["network_coherence"].values.astype(np.float64)
    global_sum += nc.sum()
    global_sum_sq += (nc ** 2).sum()
    global_n += len(nc)

    # Assign each row to a UTC date
    df["date"] = pd.to_datetime(df["epoch_time_utc"], unit="s", utc=True).dt.date

    for date, grp in df.groupby("date"):
        vals = grp["network_coherence"].values.astype(np.float64)
        daily_records.append({
            "date": date,
            "mean_nc": vals.mean(),
            "max_nc": vals.max(),
            "min_nc": vals.min(),
            "std_nc": vals.std(),
            "cumsum": vals.sum(),
            "n_samples": len(vals),
        })

    del df  # free memory

global_mean = global_sum / global_n
global_std = np.sqrt(global_sum_sq / global_n - global_mean ** 2)
print(f"\nGlobal stats  --  mean: {global_mean:.6f}   std: {global_std:.6f}   N: {global_n:,}")

# ---- Pass 1b: compute daily max |rolling Z| (1-hour window) ----
# We re-read each month, but only compute the rolling Z per day.
print("\nPass 2: computing daily max rolling Z (1-hour window) ...")

daily_max_rolling_z = {}  # date -> max |Z|

for fpath in files:
    fname = fpath.split("/")[-1]
    print(f"  Processing {fname} ...", end=" ", flush=True)
    t1 = time.time()

    df = pd.read_csv(fpath, usecols=["epoch_time_utc", "network_coherence"])
    df["date"] = pd.to_datetime(df["epoch_time_utc"], unit="s", utc=True).dt.date

    # For each day, compute rolling mean over ROLLING_WINDOW samples,
    # then convert to Z-score (how many sigma is the rolling mean from
    # the expected mean=0 given N=ROLLING_WINDOW independent samples).
    # Under the null, std of rolling mean = global_std / sqrt(ROLLING_WINDOW).
    rolling_std_of_mean = global_std / np.sqrt(ROLLING_WINDOW)

    for date, grp in df.groupby("date"):
        vals = grp["network_coherence"]
        if len(vals) < ROLLING_WINDOW:
            # Partial day -- still compute with available data
            rm = vals.rolling(window=min(len(vals), ROLLING_WINDOW), min_periods=1).mean()
        else:
            rm = vals.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        # Z-score of the rolling mean
        z = (rm - global_mean) / rolling_std_of_mean
        daily_max_rolling_z[date] = float(np.nanmax(np.abs(z.values)))

    del df
    print(f"done in {time.time()-t1:.1f}s", flush=True)

# ---------------------------------------------------------------------------
# Build master daily DataFrame
# ---------------------------------------------------------------------------
daily = pd.DataFrame(daily_records)
daily["max_rolling_z"] = daily["date"].map(daily_max_rolling_z)

# Composite anomaly score: combine multiple signals via rank-averaging
# Higher rank = more anomalous for each metric.
n_days = len(daily)

# Rank each metric (ascending rank, so highest value = highest rank number)
daily["rank_max_nc"] = daily["max_nc"].rank(method="min")
daily["rank_cumsum"] = daily["cumsum"].rank(method="min")        # positive cumsum = coherence
daily["rank_mean_nc"] = daily["mean_nc"].rank(method="min")      # higher mean = more anomalous
daily["rank_max_rz"] = daily["max_rolling_z"].rank(method="min")

# Composite: average of all four ranks
daily["composite_rank"] = (
    daily["rank_max_nc"]
    + daily["rank_cumsum"]
    + daily["rank_mean_nc"]
    + daily["rank_max_rz"]
) / 4.0

daily = daily.sort_values("composite_rank", ascending=False)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
top = daily.head(TOP_N).copy()
top["date_str"] = top["date"].apply(lambda d: d.strftime("%Y-%m-%d (%A)"))

# Also compute a "daily mean Z" = mean_nc / (global_std / sqrt(n_samples))
# This tells us how many sigma the daily mean is above/below zero.
top["daily_mean_z"] = top.apply(
    lambda r: r["mean_nc"] / (global_std / np.sqrt(r["n_samples"])), axis=1
)

print("\n" + "=" * 120)
print(f"  TOP {TOP_N} MOST ANOMALOUS DAYS IN 2025  --  GCP2 Global Network Coherence")
print("=" * 120)
print(f"  Global baseline:  mean = {global_mean:.6f},  std = {global_std:.6f},  N = {global_n:,} seconds")
print(f"  Rolling Z window: {ROLLING_WINDOW} seconds (1 hour)")
print(f"  Composite rank = average of ranks on: max_nc, cumsum, mean_nc, max_rolling_z")
print("=" * 120)

header = (
    f"{'Rank':>4}  {'Date (UTC)':>22}  "
    f"{'Mean NC':>9}  {'Max NC':>8}  {'Std NC':>8}  "
    f"{'CumSum':>12}  {'MaxRollingZ':>11}  "
    f"{'DailyMeanZ':>10}  {'Samples':>8}  {'CompRank':>8}"
)
print(header)
print("-" * 120)

for i, (_, row) in enumerate(top.iterrows(), 1):
    print(
        f"{i:>4}  {row['date_str']:>22}  "
        f"{row['mean_nc']:>9.4f}  {row['max_nc']:>8.4f}  {row['std_nc']:>8.4f}  "
        f"{row['cumsum']:>12.2f}  {row['max_rolling_z']:>11.2f}  "
        f"{row['daily_mean_z']:>10.2f}  {row['n_samples']:>8,}  {row['composite_rank']:>8.1f}"
    )

print("-" * 120)

# ---------------------------------------------------------------------------
# Also show the top 10 by each individual metric for deeper insight
# ---------------------------------------------------------------------------
for metric, label, ascending in [
    ("max_nc", "Highest Single-Second Coherence Spike", False),
    ("cumsum", "Highest Cumulative Sum (sustained positive coherence)", False),
    ("mean_nc", "Highest Daily Mean Coherence", False),
    ("max_rolling_z", "Highest 1-Hour Rolling Z-Score", False),
]:
    print(f"\n--- Top 10 by: {label} ---")
    sub = daily.nlargest(10, metric) if not ascending else daily.nsmallest(10, metric)
    for j, (_, r) in enumerate(sub.iterrows(), 1):
        d = r["date"].strftime("%Y-%m-%d")
        print(f"  {j:>2}. {d}   {metric} = {r[metric]:>12.4f}")

elapsed = time.time() - t0
print(f"\nTotal elapsed time: {elapsed:.1f}s")
