#!/usr/bin/env python3
"""
GCP 2.0 Holmberg Replication Analysis

Applies Ulf Holmberg's Max[Z]-market correlation methodology to GCP 2.0
Global Network data, adapted for GCP2's network coherence metric.

Holmberg's original approach (GCP1, 2020-2024):
  1. Raw RNG data → Z-scores per egg → Stouffer Z = Sum(z)/sqrt(N)
  2. Max[Z] = daily max of |Stouffer Z|
  3. Correlate Max[Z] with SPY returns, VIX levels, VIX changes

GCP2 adaptation:
  GCP2 provides `network_coherence` per second — a pre-whitened coherence
  metric, NOT a Z-score. It is bounded at -1.0 (baseline) with a long right
  tail (coherence events). Variance ~2.0, skew ~2.8, kurtosis ~12.

  To create a comparable daily anomaly metric:
  1. Rolling Z-score of nc over 1-hour window: z = (nc - rolling_mean) / rolling_std
     → daily max |rolling_z| = standardized peak anomaly (the Holmberg analog)
  2. Peak NC = daily max(network_coherence) = raw peak coherence event
  3. Daily NetVar = mean(nc²) = network variance

Usage:
  python3 scripts/gcp2_holmberg_analysis.py --data-dir gcp_data --months 3
  python3 scripts/gcp2_holmberg_analysis.py --data-dir gcp_data --months 12

Requires: pip install pandas numpy scipy yfinance
"""

import argparse
import sys
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── Configuration ─────────────────────────────────────────────────────────

ROLLING_WINDOW = 3600  # 1-hour rolling window for Z-score computation (seconds)
MIN_ROLLING_PERIODS = 360  # minimum 6 minutes of data for rolling stats


# ── GCP2 Data Processing ─────────────────────────────────────────────────

def load_network_zip(zip_path: Path) -> pd.DataFrame:
    """Load a single GCP2 network coherence ZIP file into a DataFrame."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    return df


def load_gcp2_data(data_dir: Path, months: int) -> pd.DataFrame:
    """Load GCP2 Global Network data for the specified number of recent months."""
    network_dir = data_dir / "network" / "global_network"

    all_zips = sorted(network_dir.rglob("*.zip"))
    if not all_zips:
        print(f"ERROR: No ZIP files found in {network_dir}")
        sys.exit(1)

    zip_info = []
    for zp in all_zips:
        name = zp.stem
        parts = name.split("_")
        try:
            year = int(parts[-2])
            month = int(parts[-1].replace(".csv", ""))
            zip_info.append((year, month, zp))
        except (ValueError, IndexError):
            continue

    zip_info.sort(key=lambda x: (x[0], x[1]))
    selected = zip_info[-months:]

    print(f"Loading GCP2 Global Network data for {len(selected)} months:")
    frames = []
    for year, month, zp in selected:
        print(f"  {year}-{month:02d}: {zp.name}...", end=" ", flush=True)
        df = load_network_zip(zp)
        print(f"{len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("epoch_time_utc", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined["datetime_utc"] = pd.to_datetime(combined["epoch_time_utc"], unit="s", utc=True)
    combined["date"] = combined["datetime_utc"].dt.date
    combined["hour_utc"] = combined["datetime_utc"].dt.hour

    print(f"\nTotal: {len(combined):,} rows "
          f"({combined['date'].min()} to {combined['date'].max()})")
    return combined


def compute_rolling_z(gcp_df: pd.DataFrame) -> pd.Series:
    """Compute rolling Z-score of network_coherence over 1-hour window.

    This normalizes GCP2's non-standard coherence metric into a proper
    Z-score, making it comparable to Holmberg's GCP1 Stouffer Z.

    z_t = (nc_t - rolling_mean) / rolling_std
    """
    nc = gcp_df["network_coherence"]
    roll = nc.rolling(ROLLING_WINDOW, min_periods=MIN_ROLLING_PERIODS)
    rolling_z = (nc - roll.mean()) / (roll.std(ddof=0) + 1e-9)
    return rolling_z


def compute_daily_gcp_metrics(gcp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily metrics from per-second GCP2 data.

    Metrics:
      max_rolling_z:  max(|rolling_z|) per day — Holmberg Max[Z] analog
      peak_nc:        max(network_coherence) per day — raw peak coherence
      mean_nc:        mean(network_coherence) per day
      netvar:         mean(network_coherence²) per day — network variance
      nc_sum:         sum(network_coherence) per day — cumulative coherence

    Note: adds 'rolling_z' column to gcp_df in-place for use by market-hours.
    """
    print("  Computing rolling Z-scores (1-hour window)...", end=" ", flush=True)
    gcp_df["rolling_z"] = compute_rolling_z(gcp_df)
    n_valid = gcp_df["rolling_z"].notna().sum()
    print(f"{n_valid:,}/{len(gcp_df):,} valid")

    daily = gcp_df.groupby("date").agg(
        max_rolling_z=("rolling_z", lambda x: x.dropna().abs().max()
                        if x.notna().any() else np.nan),
        peak_nc=("network_coherence", "max"),
        mean_nc=("network_coherence", "mean"),
        std_nc=("network_coherence", "std"),
        netvar=("network_coherence", lambda x: (x ** 2).mean()),
        nc_sum=("network_coherence", "sum"),
        median_devices=("active_devices", "median"),
        n_seconds=("network_coherence", "count"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def compute_market_hours_metrics(gcp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics using only US market hours (14:30-21:00 UTC)."""
    # Filter for 14:30-21:00 UTC (9:30 AM - 4:00 PM ET)
    market = gcp_df[
        ((gcp_df["hour_utc"] == 14) & (gcp_df["datetime_utc"].dt.minute >= 30)) |
        ((gcp_df["hour_utc"] >= 15) & (gcp_df["hour_utc"] < 21))
    ].copy()

    if market.empty:
        return pd.DataFrame()

    daily_mkt = market.groupby("date").agg(
        peak_nc_market=("network_coherence", "max"),
        max_rolling_z_market=("rolling_z", lambda x: x.dropna().abs().max()
                               if x.notna().any() else np.nan),
        n_seconds_market=("network_coherence", "count"),
    ).reset_index()

    daily_mkt["date"] = pd.to_datetime(daily_mkt["date"])
    return daily_mkt


# ── Market Data ───────────────────────────────────────────────────────────

def fetch_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch SPY and VIX daily data via yfinance."""
    import yfinance as yf

    print(f"\nFetching market data ({start_date} to {end_date})...")

    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

    if spy.empty:
        print("ERROR: No SPY data returned from yfinance")
        sys.exit(1)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    market = pd.DataFrame()
    market["date"] = spy.index
    market["spy_close"] = spy["Close"].values
    market["spy_return"] = spy["Close"].pct_change().values
    market["spy_volume"] = spy["Volume"].values

    if not vix.empty:
        vix_aligned = vix["Close"].reindex(spy.index)
        market["vix_close"] = vix_aligned.values
        market["vix_change"] = vix_aligned.diff().values
        market["vix_pct_change"] = vix_aligned.pct_change().values
        print(f"  SPY: {len(spy)} days, VIX: {len(vix)} days")
    else:
        print(f"  SPY: {len(spy)} days, VIX: unavailable")
        market["vix_close"] = np.nan
        market["vix_change"] = np.nan
        market["vix_pct_change"] = np.nan

    market["date"] = pd.to_datetime(market["date"]).dt.tz_localize(None)
    return market


# ── Correlation Analysis ──────────────────────────────────────────────────

def pearson_with_pvalue(x, y):
    """Pearson correlation with p-value, handling NaN."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p


def permutation_test(x, y, n_perms=10000):
    """Permutation test for correlation significance."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]
    if len(x_clean) < 5:
        return np.nan, np.nan, np.nan

    observed_r, _ = stats.pearsonr(x_clean, y_clean)
    rng = np.random.default_rng(42)
    perm_rs = np.zeros(n_perms)
    for i in range(n_perms):
        perm_rs[i] = stats.pearsonr(rng.permutation(x_clean), y_clean)[0]

    p_value = np.mean(np.abs(perm_rs) >= np.abs(observed_r))
    return observed_r, p_value, perm_rs.std()


def lag_analysis(gcp_series, market_series, max_lag=5):
    """Test correlations at different lags (GCP leads market by N days)."""
    results = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = gcp_series.iloc[:len(gcp_series) - lag].values if lag > 0 else gcp_series.values
            y = market_series.iloc[lag:].values if lag > 0 else market_series.values
        else:
            x = gcp_series.iloc[-lag:].values
            y = market_series.iloc[:len(market_series) + lag].values

        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        r, p = pearson_with_pvalue(x, y)
        results.append({"lag": lag, "r": r, "p_value": p})

    return pd.DataFrame(results)


def percentile_threshold_analysis(merged_df, metric_col, percentiles=[50, 75, 90, 95]):
    """Analyze correlations for extreme days using percentile thresholds."""
    results = []
    for pct in percentiles:
        thresh = merged_df[metric_col].quantile(pct / 100)
        subset = merged_df[merged_df[metric_col] >= thresh]
        n = len(subset)
        if n < 5:
            results.append({"percentile": pct, "threshold": thresh, "n": n,
                            "r_vix_change": np.nan, "p_vix_change": np.nan,
                            "r_spy_return": np.nan, "p_spy_return": np.nan})
            continue

        r_vix, p_vix = pearson_with_pvalue(
            subset[metric_col].values, subset["vix_change"].values)
        r_spy, p_spy = pearson_with_pvalue(
            subset[metric_col].values, subset["spy_return"].values)
        results.append({
            "percentile": pct, "threshold": thresh, "n": n,
            "r_vix_change": r_vix, "p_vix_change": p_vix,
            "r_spy_return": r_spy, "p_spy_return": p_spy,
        })

    return pd.DataFrame(results)


# ── Reporting ─────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def run_analysis(merged: pd.DataFrame, gcp_daily: pd.DataFrame):
    """Run the full Holmberg-style analysis and print results."""

    # ── Network Coherence Distribution ──
    print_section("GCP2 NETWORK COHERENCE: RAW METRIC PROFILE")
    print(f"""
  GCP2's network_coherence is NOT a Z-score. It is a pre-whitened
  coherence metric bounded at -1.0 (baseline) with right-skewed tail.

  To create a Holmberg-comparable metric, we compute a rolling Z-score
  of nc over a {ROLLING_WINDOW}s (1-hour) window, then take daily max |z|.

  Metric              | Value
  --------------------|----------""")
    print(f"  Mean NetVar (nc^2)  | {gcp_daily['netvar'].mean():.4f}  (expected 1.0 under null)")
    print(f"  Median devices      | {gcp_daily['median_devices'].median():.0f}")
    print(f"  Days                | {len(gcp_daily)}")

    # ── Daily Metrics ──
    print_section("DAILY METRICS: DESCRIPTIVE STATISTICS")

    metrics = [
        ("Max Rolling-Z", "max_rolling_z", "Holmberg Max[Z] analog (normalized anomaly)"),
        ("Peak NC", "peak_nc", "Raw peak coherence per day"),
        ("NetVar", "netvar", "Daily mean of nc^2 (network variance)"),
        ("NC Sum", "nc_sum", "Daily sum of nc (cumulative coherence)"),
    ]

    print(f"\n  {'Metric':<18} | {'Mean':>10} | {'Std':>10} | {'Median':>10} | {'Min':>10} | {'Max':>10}")
    print(f"  {'-'*18}-|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*12}")
    for label, col, _ in metrics:
        if col not in gcp_daily.columns:
            continue
        s = gcp_daily[col].dropna()
        print(f"  {label:<18} | {s.mean():>10.4f} | {s.std():>10.4f} | "
              f"{s.median():>10.4f} | {s.min():>10.4f} | {s.max():>10.4f}")

    # ── Basic Correlations ──
    print_section("BASIC CORRELATIONS")

    correlations = [
        ("Rolling-Z vs VIX Level", "max_rolling_z", "vix_close"),
        ("Rolling-Z vs VIX Change", "max_rolling_z", "vix_change"),
        ("Rolling-Z vs SPY Return", "max_rolling_z", "spy_return"),
        ("Peak NC vs VIX Level", "peak_nc", "vix_close"),
        ("Peak NC vs VIX Change", "peak_nc", "vix_change"),
        ("Peak NC vs SPY Return", "peak_nc", "spy_return"),
        ("NetVar vs VIX Level", "netvar", "vix_close"),
        ("NetVar vs SPY Return", "netvar", "spy_return"),
        ("NC Sum vs VIX Change", "nc_sum", "vix_change"),
        ("NC Sum vs SPY Return", "nc_sum", "spy_return"),
    ]

    print(f"\n  {'Correlation':<30} | {'r':>8} | {'p-value':>8} | Sig")
    print(f"  {'-'*30}-|{'-'*10}|{'-'*10}|{'-'*6}")

    for label, col_x, col_y in correlations:
        if col_x not in merged.columns or col_y not in merged.columns:
            continue
        r, p = pearson_with_pvalue(merged[col_x].values, merged[col_y].values)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label:<30} | {r:>8.4f} | {p:>8.4f} | {sig}")

    # ── Permutation Tests ──
    print_section("PERMUTATION TESTS (10,000 permutations)")

    perm_tests = [
        ("Rolling-Z vs VIX Change", "max_rolling_z", "vix_change"),
        ("Rolling-Z vs SPY Return", "max_rolling_z", "spy_return"),
        ("Peak NC vs VIX Change", "peak_nc", "vix_change"),
        ("Peak NC vs SPY Return", "peak_nc", "spy_return"),
    ]

    for label, col_x, col_y in perm_tests:
        if col_x not in merged.columns:
            continue
        obs_r, perm_p, null_std = permutation_test(
            merged[col_x].values, merged[col_y].values)
        conclusion = "SIGNIFICANT" if perm_p < 0.05 else "Not significant"
        print(f"\n  {label}:")
        print(f"    Observed r:  {obs_r:>8.4f}")
        print(f"    Null std:    {null_std:>8.4f}")
        print(f"    Perm p:      {perm_p:>8.4f}  ({conclusion})")

    # ── Lag Analysis ──
    print_section("LAG ANALYSIS (GCP leads market by N days)")

    lag_tests = [
        ("Rolling-Z -> VIX Change", "max_rolling_z", "vix_change"),
        ("Rolling-Z -> SPY Return", "max_rolling_z", "spy_return"),
        ("Peak NC -> VIX Change", "peak_nc", "vix_change"),
        ("Peak NC -> SPY Return", "peak_nc", "spy_return"),
    ]

    for label, gcp_col, market_col in lag_tests:
        if gcp_col not in merged.columns or market_col not in merged.columns:
            continue
        lags = lag_analysis(merged[gcp_col], merged[market_col])
        print(f"\n  {label}:")
        print(f"  {'Lag':>5} | {'r':>8} | {'p-value':>8}")
        print(f"  {'-'*5}-|{'-'*10}|{'-'*10}")
        for _, row in lags.iterrows():
            lag_label = f"+{int(row['lag'])}" if row['lag'] >= 0 else str(int(row['lag']))
            sig = "*" if row["p_value"] < 0.05 else ""
            print(f"  {lag_label:>5} | {row['r']:>8.4f} | {row['p_value']:>8.4f} {sig}")

    # ── Percentile Threshold Analysis ──
    print_section("THRESHOLD ANALYSIS (Percentile-Based)")

    for label, col in [("Max Rolling-Z", "max_rolling_z"), ("Peak NC", "peak_nc")]:
        if col not in merged.columns:
            continue
        thresholds = percentile_threshold_analysis(merged, col)
        print(f"\n  {label}:")
        print(f"  {'Pctl':>6} | {'Thresh':>8} | {'n':>5} | {'r(dVIX)':>8} | "
              f"{'p(dVIX)':>8} | {'r(SPY)':>8} | {'p(SPY)':>8}")
        print(f"  {'-'*6}-|{'-'*10}|{'-'*7}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}")
        for _, row in thresholds.iterrows():
            def fmt(v):
                return f"{v:.4f}" if not np.isnan(v) else "   N/A"
            print(f"  P{int(row['percentile']):>4} | {row['threshold']:>8.2f} | "
                  f"{int(row['n']):>5} | {fmt(row['r_vix_change']):>8} | "
                  f"{fmt(row['p_vix_change']):>8} | {fmt(row['r_spy_return']):>8} | "
                  f"{fmt(row['p_spy_return']):>8}")

    # ── Extreme Days ──
    print_section("TOP 10 HIGHEST PEAK COHERENCE DAYS")

    top10_days = merged.nlargest(10, "peak_nc")
    print(f"\n  {'Date':<12} | {'PeakNC':>8} | {'RollZ':>8} | {'VIX':>6} | "
          f"{'dVIX':>7} | {'SPY%':>8} | {'Dev':>5}")
    print(f"  {'-'*12}-|{'-'*10}|{'-'*10}|{'-'*8}|{'-'*9}|{'-'*10}|{'-'*7}")
    for _, row in top10_days.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d")
        rz = f"{row['max_rolling_z']:.2f}" if not np.isnan(row.get('max_rolling_z', np.nan)) else "  N/A"
        vix = f"{row['vix_close']:.1f}" if not np.isnan(row.get('vix_close', np.nan)) else " N/A"
        dvix = f"{row['vix_change']:.2f}" if not np.isnan(row.get('vix_change', np.nan)) else "  N/A"
        spy = f"{row['spy_return']*100:.2f}%" if not np.isnan(row.get('spy_return', np.nan)) else "  N/A"
        dev = f"{row['median_devices']:.0f}" if not np.isnan(row.get('median_devices', np.nan)) else "N/A"
        print(f"  {date_str:<12} | {row['peak_nc']:>8.2f} | {rz:>8} | "
              f"{vix:>6} | {dvix:>7} | {spy:>8} | {dev:>5}")

    if len(top10_days) > 0:
        print(f"\n  Extreme days avg: dVIX = {top10_days['vix_change'].mean():.2f}, "
              f"SPY = {top10_days['spy_return'].mean()*100:.2f}%")
        print(f"  All days avg:     dVIX = {merged['vix_change'].mean():.2f}, "
              f"SPY = {merged['spy_return'].mean()*100:.2f}%")

    # ── Market Hours Comparison ──
    if "max_rolling_z_market" in merged.columns:
        print_section("MARKET HOURS vs FULL DAY COMPARISON")
        full = merged["max_rolling_z"].dropna()
        mkt = merged["max_rolling_z_market"].dropna()
        print(f"\n  {'Window':<20} | {'Mean':>8} | {'Std':>8} | {'Median':>8}")
        print(f"  {'-'*20}-|{'-'*10}|{'-'*10}|{'-'*10}")
        print(f"  {'Full day (24h)':<20} | {full.mean():>8.2f} | {full.std():>8.2f} | {full.median():>8.2f}")
        print(f"  {'Market hours':<20} | {mkt.mean():>8.2f} | {mkt.std():>8.2f} | {mkt.median():>8.2f}")

        both = merged[["max_rolling_z", "max_rolling_z_market"]].dropna()
        if len(both) >= 5:
            r, p = stats.pearsonr(both["max_rolling_z"], both["max_rolling_z_market"])
            print(f"\n  Full vs market-hours correlation: r = {r:.4f}, p = {p:.4f}")

    # ── VIX Regime Analysis ──
    print_section("VIX REGIME ANALYSIS")

    vix_valid = merged.dropna(subset=["vix_close"])
    vix_median = vix_valid["vix_close"].median()
    high_vix = vix_valid[vix_valid["vix_close"] > vix_median]
    low_vix = vix_valid[vix_valid["vix_close"] <= vix_median]

    print(f"\n  VIX median: {vix_median:.1f}")

    for label, col in [("Max Rolling-Z", "max_rolling_z"), ("Peak NC", "peak_nc")]:
        if col not in merged.columns:
            continue
        print(f"\n  {label}:")
        print(f"  {'Regime':<20} | {'n':>5} | {'Mean':>10} | {'Std':>8}")
        print(f"  {'-'*20}-|{'-'*7}|{'-'*12}|{'-'*10}")
        print(f"  {'High VIX (>' + f'{vix_median:.0f})':<20} | {len(high_vix):>5} | "
              f"{high_vix[col].mean():>10.4f} | {high_vix[col].std():>8.4f}")
        print(f"  {'Low VIX (<=' + f'{vix_median:.0f})':<20} | {len(low_vix):>5} | "
              f"{low_vix[col].mean():>10.4f} | {low_vix[col].std():>8.4f}")

        h_vals = high_vix[col].dropna()
        l_vals = low_vix[col].dropna()
        if len(h_vals) >= 5 and len(l_vals) >= 5:
            t_stat, t_p = stats.ttest_ind(h_vals, l_vals)
            sig = "Significant" if t_p < 0.05 else "Not significant"
            print(f"  t-test: t = {t_stat:.2f}, p = {t_p:.4f} ({sig})")

    # ── Summary ──
    print_section("SUMMARY")

    rz_vix_r, rz_vix_p = pearson_with_pvalue(
        merged["max_rolling_z"].values, merged["vix_close"].values)
    rz_dvix_r, rz_dvix_p = pearson_with_pvalue(
        merged["max_rolling_z"].values, merged["vix_change"].values)
    rz_spy_r, rz_spy_p = pearson_with_pvalue(
        merged["max_rolling_z"].values, merged["spy_return"].values)
    pnc_vix_r, pnc_vix_p = pearson_with_pvalue(
        merged["peak_nc"].values, merged["vix_close"].values)
    pnc_dvix_r, pnc_dvix_p = pearson_with_pvalue(
        merged["peak_nc"].values, merged["vix_change"].values)
    pnc_spy_r, pnc_spy_p = pearson_with_pvalue(
        merged["peak_nc"].values, merged["spy_return"].values)

    print(f"""
  Dataset:      GCP2 Global Network
  Period:       {merged['date'].min().strftime('%Y-%m-%d')} to {merged['date'].max().strftime('%Y-%m-%d')}
  Trading days: {len(merged)}
  Devices:      ~{gcp_daily['median_devices'].median():.0f}
  NetVar:       {gcp_daily['netvar'].mean():.4f} (expected 1.0 under null)

  GCP1 replication (2016-2022, Stouffer Z, 1,695 days):
    Max[Z] vs VIX:    r = -0.049, p = 0.04
    Max[Z] vs SPY:    r = -0.001, p = 0.98
    Permutation:      p = 0.56

  GCP2 analysis (Rolling Z-score of network coherence):
    Rolling-Z vs VIX:    r = {rz_vix_r:.4f}, p = {rz_vix_p:.4f}
    Rolling-Z vs dVIX:   r = {rz_dvix_r:.4f}, p = {rz_dvix_p:.4f}
    Rolling-Z vs SPY:    r = {rz_spy_r:.4f}, p = {rz_spy_p:.4f}

  GCP2 analysis (Raw peak network coherence):
    Peak NC vs VIX:      r = {pnc_vix_r:.4f}, p = {pnc_vix_p:.4f}
    Peak NC vs dVIX:     r = {pnc_dvix_r:.4f}, p = {pnc_dvix_p:.4f}
    Peak NC vs SPY:      r = {pnc_spy_r:.4f}, p = {pnc_spy_p:.4f}""")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GCP2 Holmberg Replication Analysis")
    parser.add_argument("--data-dir", type=str, default="gcp_data",
                        help="GCP2 data directory")
    parser.add_argument("--months", type=int, default=3,
                        help="Number of recent months to analyze")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("  GCP 2.0 HOLMBERG REPLICATION ANALYSIS")
    print("  Adapted for GCP2 network coherence metric")
    print("=" * 70)

    # Step 1: Load GCP2 data
    gcp_df = load_gcp2_data(data_dir, args.months)

    # Step 2: Compute daily GCP metrics (includes rolling Z computation)
    print("\nComputing daily GCP2 metrics...")
    gcp_daily = compute_daily_gcp_metrics(gcp_df)
    print(f"  {len(gcp_daily)} days computed")

    # Market-hours metrics (needs rolling_z already on gcp_df)
    gcp_market_hours = compute_market_hours_metrics(gcp_df)
    if not gcp_market_hours.empty:
        gcp_daily = gcp_daily.merge(gcp_market_hours, on="date", how="left")
        print(f"  Market-hours metrics for {len(gcp_market_hours)} days")

    # Step 3: Fetch market data
    start_date = gcp_daily["date"].min().strftime("%Y-%m-%d")
    end_date = (gcp_daily["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
    market = fetch_market_data(start_date, end_date)

    # Step 4: Merge on date (trading days only)
    merged = gcp_daily.merge(market, on="date", how="inner")
    print(f"\nMerged dataset: {len(merged)} trading days "
          f"({merged['date'].min().strftime('%Y-%m-%d')} to "
          f"{merged['date'].max().strftime('%Y-%m-%d')})")

    if len(merged) < 10:
        print("ERROR: Too few overlapping days for analysis.")
        sys.exit(1)

    # Step 5: Run analysis
    run_analysis(merged, gcp_daily)

    # Save merged data
    output_csv = data_dir / "holmberg_analysis_merged.csv"
    merged.to_csv(output_csv, index=False)
    print(f"\nMerged data saved to: {output_csv}")


if __name__ == "__main__":
    main()
