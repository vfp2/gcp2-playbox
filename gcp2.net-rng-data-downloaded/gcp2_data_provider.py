#!/usr/bin/env python3
"""
GCP 2.0 Data Provider - Unified interface for CSV and API data

This module provides a unified interface for accessing GCP2 network coherence data,
seamlessly combining historical CSV files with live API data.

Usage:
    from gcp2_data_provider import get_cluster_coherence, get_available_clusters

    # Get data for a time range (automatically uses CSV or API as needed)
    df = get_cluster_coherence("global_network", start_ts, end_ts)

    # List available clusters
    clusters = get_available_clusters()
"""

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from gcp2_api_client import (
    GCP2APIClient,
    cluster_name_to_folder,
    folder_to_cluster_name,
)

# Configuration
DATA_DIR = Path(__file__).parent
NETWORK_DIR = DATA_DIR / "network"

# Cutoff timestamp: data before this comes from CSV, after from API
# Set to the last complete day of CSV data (Jan 27, 2026 23:59:59 UTC)
CSV_CUTOFF_TS = datetime(2026, 1, 27, 23, 59, 59, tzinfo=timezone.utc).timestamp()

# Cache for API client
_api_client: Optional[GCP2APIClient] = None


def get_api_client() -> GCP2APIClient:
    """Get or create the API client singleton."""
    global _api_client
    if _api_client is None:
        _api_client = GCP2APIClient()
    return _api_client


@lru_cache(maxsize=32)
def get_csv_available_months(folder_name: str) -> list[tuple[int, int, Path]]:
    """Get list of (year, month, csv_path) tuples for available CSV data."""
    network_path = NETWORK_DIR / folder_name
    if not network_path.exists():
        return []

    available = []
    for year_dir in network_path.iterdir():
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        for csv_file in year_dir.glob("*.csv"):
            if ".csv.zip" in csv_file.name:
                continue
            # Parse month from filename like "GCP2_Network_Coherence_Global_Network_2025_01.csv"
            parts = csv_file.stem.split("_")
            try:
                month = int(parts[-1])
                available.append((year, month, csv_file))
            except (ValueError, IndexError):
                continue

    return sorted(available, key=lambda x: (x[0], x[1]))


def get_csv_latest_timestamp(folder_name: str) -> Optional[float]:
    """Get the latest timestamp available in CSV files for a cluster."""
    months = get_csv_available_months(folder_name)
    if not months:
        return None

    # Get the latest month's file
    _, _, latest_file = months[-1]

    try:
        # Read just the last few rows to get the max timestamp
        df = pd.read_csv(latest_file)
        if df.empty:
            return None
        return float(df["epoch_time_utc"].max())
    except Exception:
        return None


def load_csv_data(folder_name: str, start_ts: float, end_ts: float) -> pd.DataFrame:
    """Load data from CSV files for the specified time range."""
    months = get_csv_available_months(folder_name)
    if not months:
        return pd.DataFrame(columns=["epoch_time_utc", "network_coherence", "active_devices"])

    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    # Find relevant CSV files
    relevant_files = []
    for year, month, csv_path in months:
        file_start = datetime(year, month, 1, tzinfo=timezone.utc)
        # Approximate end of month
        if month == 12:
            file_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            file_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        # Check if file overlaps with requested range
        if file_start <= end_dt and file_end > start_dt:
            relevant_files.append(csv_path)

    if not relevant_files:
        return pd.DataFrame(columns=["epoch_time_utc", "network_coherence", "active_devices"])

    # Load and concatenate
    frames = []
    for csv_path in relevant_files:
        try:
            df = pd.read_csv(csv_path)
            frames.append(df)
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            continue

    if not frames:
        return pd.DataFrame(columns=["epoch_time_utc", "network_coherence", "active_devices"])

    combined = pd.concat(frames, ignore_index=True)

    # Filter to exact time range
    mask = (combined["epoch_time_utc"] >= start_ts) & (combined["epoch_time_utc"] <= end_ts)
    result = combined[mask].copy()

    return result.sort_values("epoch_time_utc").reset_index(drop=True)


def load_api_data(folder_name: str, start_ts: float, end_ts: float) -> pd.DataFrame:
    """Load data from API for the specified time range."""
    cluster_name = folder_to_cluster_name(folder_name)
    client = get_api_client()

    try:
        df = client.get_cluster_history(cluster_name, start_ts=start_ts, end_ts=end_ts)
        return df
    except Exception as e:
        print(f"Error fetching API data for {cluster_name}: {e}")
        return pd.DataFrame(columns=["epoch_time_utc", "network_coherence", "active_devices"])


def get_cluster_coherence(
    folder_name: str,
    start_ts: float,
    end_ts: float,
    prefer_api: bool = False
) -> pd.DataFrame:
    """Get cluster coherence data for the specified time range.

    Automatically uses CSV files for historical data and API for recent data.

    Args:
        folder_name: Cluster folder name (e.g., "global_network", "cluster_london")
        start_ts: Start timestamp (epoch seconds)
        end_ts: End timestamp (epoch seconds)
        prefer_api: If True, always use API even for historical data

    Returns:
        DataFrame with columns: epoch_time_utc, network_coherence, active_devices
    """
    if prefer_api:
        return load_api_data(folder_name, start_ts, end_ts)

    # Determine which data sources to use
    csv_cutoff = get_csv_latest_timestamp(folder_name) or CSV_CUTOFF_TS

    frames = []

    # Load CSV data for historical portion
    if start_ts < csv_cutoff:
        csv_end = min(end_ts, csv_cutoff)
        csv_df = load_csv_data(folder_name, start_ts, csv_end)
        if not csv_df.empty:
            frames.append(csv_df)

    # Load API data for recent portion
    if end_ts > csv_cutoff:
        api_start = max(start_ts, csv_cutoff + 1)
        api_df = load_api_data(folder_name, api_start, end_ts)
        if not api_df.empty:
            frames.append(api_df)

    if not frames:
        return pd.DataFrame(columns=["epoch_time_utc", "network_coherence", "active_devices"])

    # Combine and deduplicate
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["epoch_time_utc"], keep="last")
    combined = combined.sort_values("epoch_time_utc").reset_index(drop=True)

    return combined


def get_available_clusters() -> list[dict]:
    """Get list of available clusters from both CSV and API.

    Returns:
        List of dicts with 'folder_name', 'display_name', 'source' keys
    """
    clusters = {}

    # Add clusters from CSV directories
    if NETWORK_DIR.exists():
        for folder in NETWORK_DIR.iterdir():
            if folder.is_dir():
                folder_name = folder.name
                display_name = folder_to_cluster_name(folder_name)
                clusters[folder_name] = {
                    "folder_name": folder_name,
                    "display_name": display_name,
                    "source": "csv"
                }

    # Add clusters from API
    try:
        client = get_api_client()
        api_clusters = client.list_clusters()
        for c in api_clusters:
            folder_name = cluster_name_to_folder(c["name"])
            if folder_name in clusters:
                clusters[folder_name]["source"] = "csv+api"
            else:
                clusters[folder_name] = {
                    "folder_name": folder_name,
                    "display_name": c["name"],
                    "source": "api"
                }
    except Exception as e:
        print(f"Warning: Could not fetch clusters from API: {e}")

    return sorted(clusters.values(), key=lambda x: x["folder_name"])


if __name__ == "__main__":
    from datetime import timedelta

    print("=== GCP2 Data Provider Test ===\n")

    # List available clusters
    print("Available clusters:")
    for c in get_available_clusters():
        print(f"  - {c['folder_name']} ({c['source']})")

    # Test loading data spanning CSV and API
    print("\n--- Testing data loading ---")

    # Historical data (should use CSV)
    print("\n1. Historical data (CSV):")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    end = datetime(2025, 1, 1, 0, 5, 0, tzinfo=timezone.utc).timestamp()
    df = get_cluster_coherence("global_network", start, end)
    print(f"   Range: 2025-01-01 00:00 to 00:05")
    print(f"   Records: {len(df)}")
    if not df.empty:
        print(f"   Sample:\n{df.head()}")

    # Recent data (should use API)
    print("\n2. Recent data (API):")
    start = datetime(2026, 2, 10, 23, 0, 0, tzinfo=timezone.utc).timestamp()
    end = datetime(2026, 2, 10, 23, 5, 0, tzinfo=timezone.utc).timestamp()
    df = get_cluster_coherence("global_network", start, end)
    print(f"   Range: 2026-02-10 23:00 to 23:05")
    print(f"   Records: {len(df)}")
    if not df.empty:
        print(f"   Sample:\n{df.head()}")

    # Spanning data (should use both)
    print("\n3. Spanning data (CSV + API):")
    csv_cutoff = get_csv_latest_timestamp("global_network")
    if csv_cutoff:
        cutoff_dt = datetime.fromtimestamp(csv_cutoff, tz=timezone.utc)
        print(f"   CSV cutoff: {cutoff_dt}")
        start = csv_cutoff - 300  # 5 min before cutoff
        end = csv_cutoff + 86400  # 1 day after cutoff
        df = get_cluster_coherence("global_network", start, end)
        print(f"   Records: {len(df)}")
