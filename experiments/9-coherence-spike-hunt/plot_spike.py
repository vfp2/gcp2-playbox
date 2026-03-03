#!/usr/bin/env python3
"""
Experiment 9: Visualize the March 1, 2026 coherence spike
and compare it against all historical spikes in GCP2 Global Network history.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "gcp2.net-rng-data-downloaded"
NETWORK_DIR = DATA_DIR / "network" / "global_network"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_global_network_range(start_ts: float, end_ts: float) -> pd.DataFrame:
    frames = []
    for year_dir in sorted(NETWORK_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for csv_file in sorted(year_dir.glob("*.csv")):
            if ".csv.zip" in csv_file.name:
                continue
            df = pd.read_csv(csv_file)
            filtered = df[(df["epoch_time_utc"] >= start_ts) & (df["epoch_time_utc"] <= end_ts)]
            if not filtered.empty:
                frames.append(filtered)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["epoch_time_utc"])
    return combined.sort_values("epoch_time_utc").reset_index(drop=True)


def load_all_global_network() -> pd.DataFrame:
    frames = []
    for year_dir in sorted(NETWORK_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for csv_file in sorted(year_dir.glob("*.csv")):
            if ".csv.zip" in csv_file.name:
                continue
            frames.append(pd.read_csv(csv_file))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["epoch_time_utc"])
    return combined.sort_values("epoch_time_utc").reset_index(drop=True)


def detect_episodes(nc, timestamps, window=900, sigma=3.0, min_dur=300):
    expected_std = np.sqrt(2.0 / window)
    threshold = sigma * expected_std
    rolling_mean = pd.Series(nc).rolling(window=window, min_periods=window // 2).mean()
    above = rolling_mean > threshold

    runs = []
    in_run = False
    rs = 0
    for i in range(len(nc)):
        if above.iloc[i] and not in_run:
            in_run = True
            rs = i
        elif not above.iloc[i] and in_run:
            in_run = False
            re = i - 1
            dur = timestamps[re] - timestamps[rs]
            if dur >= min_dur:
                runs.append({
                    "start_ts": timestamps[rs], "end_ts": timestamps[re],
                    "start_idx": rs, "end_idx": re,
                    "duration_sec": dur,
                    "peak": float(rolling_mean.iloc[rs:re + 1].max()),
                    "cumsum": float(nc[rs:re + 1].sum()),
                })
    if in_run:
        re = len(nc) - 1
        dur = timestamps[re] - timestamps[rs]
        if dur >= min_dur:
            runs.append({
                "start_ts": timestamps[rs], "end_ts": timestamps[re],
                "start_idx": rs, "end_idx": re,
                "duration_sec": dur,
                "peak": float(rolling_mean.iloc[rs:re + 1].max()),
                "cumsum": float(nc[rs:re + 1].sum()),
            })
    return sorted(runs, key=lambda x: -x["duration_sec"]), rolling_mean


# ─── Style ────────────────────────────────────────────────────────────────────

BG = "#0d1117"
PANEL_BG = "#161b22"
GRID = "#21262d"
TEXT = "#c9d1d9"
ACCENT = "#ff6b6b"
ACCENT2 = "#4ecdc4"
ACCENT3 = "#ffe66d"
SIGMA_LINE = "#ff4444"
ENVELOPE = "#1a3a2a"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "grid.color": GRID,
    "grid.alpha": 0.4,
    "font.family": "monospace",
    "font.size": 9,
})


def main():
    print("Loading data...")

    # ── Panel 1 data: Mar 1 spike close-up ────────────────────────────────
    spike_start = datetime(2026, 3, 1, 2, 0, 0, tzinfo=timezone.utc).timestamp()
    spike_end = datetime(2026, 3, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp()
    df_spike = load_global_network_range(spike_start, spike_end)
    print(f"  Spike window: {len(df_spike):,} records")

    # ── Panel 2 data: Dec 11 2025 comparison ──────────────────────────────
    dec_start = datetime(2025, 12, 11, 15, 0, 0, tzinfo=timezone.utc).timestamp()
    dec_end = datetime(2025, 12, 12, 1, 0, 0, tzinfo=timezone.utc).timestamp()
    df_dec = load_global_network_range(dec_start, dec_end)
    print(f"  Dec 11 window: {len(df_dec):,} records")

    # ── Panel 3 data: all history episode durations ───────────────────────
    print("  Loading full history for episode scan...")
    df_all = load_all_global_network()
    print(f"  Full history: {len(df_all):,} records")

    nc_all = df_all["network_coherence"].values
    ts_all = df_all["epoch_time_utc"].values
    all_episodes, _ = detect_episodes(nc_all, ts_all, window=900, sigma=3.0, min_dur=60)
    # Exclude the Mar 2024 startup artifact (>100h)
    genuine = [e for e in all_episodes if e["duration_sec"] < 360000]
    print(f"  Found {len(genuine)} genuine episodes (excl. startup)")

    # ── Build the figure ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 2.5], hspace=0.32, wspace=0.28,
                          left=0.07, right=0.97, top=0.93, bottom=0.06)

    fig.suptitle("GCP2 GLOBAL NETWORK COHERENCE — SPIKE HUNT",
                 fontsize=16, fontweight="bold", color=ACCENT, y=0.97)
    fig.text(0.5, 0.945,
             "Experiment 9  |  731 days of per-second data  |  59.9 million records",
             ha="center", fontsize=9, color="#8b949e")

    window = 900
    expected_std = np.sqrt(2.0 / window)

    # ── Panel 1: Mar 1 2026 spike (left) ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    nc1 = df_spike["network_coherence"].values
    ts1 = df_spike["epoch_time_utc"].values
    rm1 = pd.Series(nc1).rolling(window=window, min_periods=window // 2).mean()
    sigma_vals1 = rm1 / expected_std
    dt1 = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts1]

    ax1.fill_between(dt1, sigma_vals1, 0, where=sigma_vals1 > 0,
                     color=ACCENT, alpha=0.3, linewidth=0)
    ax1.fill_between(dt1, sigma_vals1, 0, where=sigma_vals1 < 0,
                     color="#4488ff", alpha=0.15, linewidth=0)
    ax1.plot(dt1, sigma_vals1, color=ACCENT, linewidth=0.7, alpha=0.9)
    ax1.axhline(3.0, color=SIGMA_LINE, linewidth=1, linestyle="--", alpha=0.7, label="3σ threshold")
    ax1.axhline(-3.0, color=SIGMA_LINE, linewidth=1, linestyle="--", alpha=0.7)

    # Mark the sustained episode
    ep_start = datetime(2026, 3, 1, 4, 13, 44, tzinfo=timezone.utc)
    ep_end = datetime(2026, 3, 1, 5, 40, 6, tzinfo=timezone.utc)
    ax1.axvspan(ep_start, ep_end, color=ACCENT3, alpha=0.08)
    ax1.annotate("1h 26m above 3σ", xy=(ep_start, 8.2), fontsize=8,
                 color=ACCENT3, fontweight="bold")

    peak_time = datetime(2026, 3, 1, 5, 30, 0, tzinfo=timezone.utc)
    ax1.annotate("PEAK 8.0σ", xy=(peak_time, 8.0), xytext=(peak_time, 9.5),
                 fontsize=8, color=ACCENT, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.2),
                 ha="center")

    ax1.set_title("MARCH 1, 2026 — THE SPIKE", fontsize=11, fontweight="bold",
                  color=ACCENT, pad=8)
    ax1.set_ylabel("15-min rolling mean (σ units)")
    ax1.set_ylim(-4, 11)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax1.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # ── Panel 1b: Mar 1 cumulative sum (right) ───────────────────────────
    ax1b = fig.add_subplot(gs[0, 1])

    cumsum1 = np.cumsum(nc1)
    # Expected random walk envelope (95%)
    n_points = np.arange(1, len(nc1) + 1)
    envelope = 1.96 * np.sqrt(2.0 * n_points)

    ax1b.fill_between(dt1, envelope, -envelope, color=ENVELOPE, alpha=0.5, label="95% envelope")
    ax1b.plot(dt1, cumsum1, color=ACCENT, linewidth=1.0)
    ax1b.axvspan(ep_start, ep_end, color=ACCENT3, alpha=0.08)

    ax1b.set_title("CUMULATIVE SUM — MAR 1, 2026", fontsize=11, fontweight="bold",
                   color=ACCENT, pad=8)
    ax1b.set_ylabel("Cumulative NC")
    ax1b.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax1b.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax1b.grid(True, alpha=0.3)

    # ── Panel 2: Dec 11 2025 comparison (left) ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    nc2 = df_dec["network_coherence"].values
    ts2 = df_dec["epoch_time_utc"].values
    rm2 = pd.Series(nc2).rolling(window=window, min_periods=window // 2).mean()
    sigma_vals2 = rm2 / expected_std
    dt2 = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts2]

    ax2.fill_between(dt2, sigma_vals2, 0, where=sigma_vals2 > 0,
                     color=ACCENT2, alpha=0.3, linewidth=0)
    ax2.fill_between(dt2, sigma_vals2, 0, where=sigma_vals2 < 0,
                     color="#4488ff", alpha=0.15, linewidth=0)
    ax2.plot(dt2, sigma_vals2, color=ACCENT2, linewidth=0.7, alpha=0.9)
    ax2.axhline(3.0, color=SIGMA_LINE, linewidth=1, linestyle="--", alpha=0.7, label="3σ threshold")
    ax2.axhline(-3.0, color=SIGMA_LINE, linewidth=1, linestyle="--", alpha=0.7)

    dec_ep_start = datetime(2025, 12, 11, 17, 49, 0, tzinfo=timezone.utc)
    dec_ep_end = datetime(2025, 12, 11, 21, 13, 15, tzinfo=timezone.utc)
    ax2.axvspan(dec_ep_start, dec_ep_end, color=ACCENT3, alpha=0.08)
    ax2.annotate("3h 24m above 3σ", xy=(dec_ep_start, 8.5), fontsize=8,
                 color=ACCENT3, fontweight="bold")

    ax2.set_title("DEC 11, 2025 — LONGEST GENUINE SPIKE (comparison)", fontsize=11,
                  fontweight="bold", color=ACCENT2, pad=8)
    ax2.set_ylabel("15-min rolling mean (σ units)")
    ax2.set_ylim(-4, 11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax2.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax2.grid(True, alpha=0.3)

    # ── Panel 2b: Dec 11 cumulative sum (right) ──────────────────────────
    ax2b = fig.add_subplot(gs[1, 1])

    cumsum2 = np.cumsum(nc2)
    n_points2 = np.arange(1, len(nc2) + 1)
    envelope2 = 1.96 * np.sqrt(2.0 * n_points2)

    ax2b.fill_between(dt2, envelope2, -envelope2, color=ENVELOPE, alpha=0.5, label="95% envelope")
    ax2b.plot(dt2, cumsum2, color=ACCENT2, linewidth=1.0)
    ax2b.axvspan(dec_ep_start, dec_ep_end, color=ACCENT3, alpha=0.08)

    ax2b.set_title("CUMULATIVE SUM — DEC 11, 2025", fontsize=11, fontweight="bold",
                   color=ACCENT2, pad=8)
    ax2b.set_ylabel("Cumulative NC")
    ax2b.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
    ax2b.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax2b.grid(True, alpha=0.3)

    # ── Panel 3: Historical episode duration ranking ─────────────────────
    ax3 = fig.add_subplot(gs[2, :])

    # Sort by duration descending, show top 25
    top = genuine[:25]
    durations_min = [e["duration_sec"] / 60 for e in top]
    labels = []
    colors = []
    for e in top:
        dt = datetime.fromtimestamp(e["start_ts"], tz=timezone.utc)
        labels.append(dt.strftime("%Y-%m-%d\n%H:%M"))
        if dt.year == 2026 and dt.month == 3 and dt.day == 1:
            colors.append(ACCENT)
        elif dt.year == 2025 and dt.month == 12 and dt.day == 11:
            colors.append(ACCENT2)
        else:
            colors.append("#8b949e")

    x = np.arange(len(top))
    bars = ax3.bar(x, durations_min, color=colors, width=0.7, edgecolor="none", alpha=0.85)

    # Add duration labels on bars
    for i, (bar, dur) in enumerate(zip(bars, durations_min)):
        if dur >= 60:
            label = f"{dur / 60:.1f}h"
        else:
            label = f"{dur:.0f}m"
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 label, ha="center", va="bottom", fontsize=7, color=TEXT)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=6.5, rotation=0)
    ax3.set_ylabel("Duration (minutes)")
    ax3.set_title("ALL SUSTAINED POSITIVE COHERENCE EPISODES (3σ, >1 min) — RANKED BY DURATION",
                  fontsize=11, fontweight="bold", color=TEXT, pad=8)
    ax3.axhline(60, color=ACCENT3, linewidth=1, linestyle=":", alpha=0.5, label="1 hour")
    ax3.legend(loc="upper right", fontsize=7, framealpha=0.3)
    ax3.grid(True, axis="y", alpha=0.3)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ACCENT, label="Mar 1, 2026 (1h 26m)"),
        Patch(facecolor=ACCENT2, label="Dec 11, 2025 (3h 24m)"),
        Patch(facecolor="#8b949e", label="Other episodes"),
    ]
    ax3.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.3)

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "spike_hunt_mar1_2026.png"
    fig.savefig(out_path, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
