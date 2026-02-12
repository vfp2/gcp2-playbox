#!/usr/bin/env python3
"""
GCP 2.0 API Client for graphql.rng.observer

A simple client for fetching RNG coherence data from the public API.
No authentication required.

Usage:
    from gcp2_api_client import GCP2APIClient

    client = GCP2APIClient()
    clusters = client.list_clusters()
    df = client.get_cluster_history("Global Network", start_ts, end_ts)
"""

import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

# API Configuration
BASE_URL = "https://graphql.rng.observer"
MAX_RECORDS_PER_REQUEST = 10080  # ~1 week of second-level data
REQUEST_DELAY = 0.5  # seconds between paginated requests


class GCP2APIClient:
    """Client for the GCP 2.0 RNG Observer API."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def list_clusters(self, query: str = "") -> list[dict]:
        """Fetch list of available clusters.

        Args:
            query: Search term to filter clusters (empty for all)

        Returns:
            List of cluster dicts with name, city, country, coordinates, lastUpdated
        """
        # Use different queries to get all clusters
        all_clusters = {}

        for search_term in ["global", "cluster", "research", "tower"]:
            url = f"{self.base_url}/api/rest/search/clusters"
            params = {"query": search_term, "limit": 100}
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for cluster in data.get("clusters", []):
                all_clusters[cluster["name"]] = cluster

        # If specific query provided, filter
        if query:
            return [c for c in all_clusters.values()
                    if query.lower() in c["name"].lower()]

        return list(all_clusters.values())

    def list_devices(self, query: str = "") -> list[dict]:
        """Fetch list of available devices.

        Args:
            query: Search term to filter devices

        Returns:
            List of device dicts with id, name, city, country, coordinates, lastUpdated
        """
        url = f"{self.base_url}/api/rest/search/devices"
        params = {"query": query, "limit": 1000}
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("devices", [])

    def get_cluster_history_raw(
        self,
        cluster_name: str,
        limit: int = MAX_RECORDS_PER_REQUEST,
        offset: int = 0
    ) -> dict:
        """Fetch raw cluster history from API (single request).

        Args:
            cluster_name: Name of the cluster (e.g., "Global Network")
            limit: Max records to fetch (max 10080)
            offset: Offset for pagination

        Returns:
            Raw API response dict
        """
        url = f"{self.base_url}/api/rest/cluster-history"
        params = {
            "clusterName": cluster_name,
            "limit": min(limit, MAX_RECORDS_PER_REQUEST),
            "offset": offset
        }
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def get_cluster_history(
        self,
        cluster_name: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ) -> pd.DataFrame:
        """Fetch cluster coherence history as a DataFrame.

        Handles pagination automatically. Returns data in CSV-compatible format.

        Args:
            cluster_name: Name of the cluster (e.g., "Global Network")
            start_ts: Start timestamp (epoch seconds), optional
            end_ts: End timestamp (epoch seconds), optional
            progress_callback: Optional callback(fetched_count, message) for progress

        Returns:
            DataFrame with columns: epoch_time_utc, network_coherence, active_devices
        """
        all_records = []
        offset = 0

        while True:
            if progress_callback:
                progress_callback(len(all_records), f"Fetching offset {offset}...")

            data = self.get_cluster_history_raw(cluster_name, MAX_RECORDS_PER_REQUEST, offset)
            history = data.get("cluster", {}).get("history", [])

            if not history:
                break

            # Filter by time range if specified
            for record in history:
                ts = record["timestamp"]
                if start_ts is not None and ts < start_ts:
                    # Data is in reverse chronological order, so we're past our range
                    # But we need to keep checking since offset might skip around
                    continue
                if end_ts is not None and ts > end_ts:
                    continue
                all_records.append(record)

            # Check if we've gone past start_ts (data is newest first)
            oldest_ts = min(r["timestamp"] for r in history)
            if start_ts is not None and oldest_ts < start_ts:
                break

            # If we got fewer than requested, we've hit the end
            if len(history) < MAX_RECORDS_PER_REQUEST:
                break

            offset += MAX_RECORDS_PER_REQUEST
            time.sleep(REQUEST_DELAY)

        if not all_records:
            return pd.DataFrame(columns=["epoch_time_utc", "network_coherence", "active_devices"])

        # Convert to DataFrame with CSV-compatible column names
        df = pd.DataFrame(all_records)
        df = df.rename(columns={
            "timestamp": "epoch_time_utc",
            "coherence": "network_coherence",
            "activeDevices": "active_devices"
        })

        # Sort by timestamp ascending (CSV files are in chronological order)
        df = df.sort_values("epoch_time_utc").reset_index(drop=True)

        return df

    def get_device_history_raw(
        self,
        limit: int = MAX_RECORDS_PER_REQUEST,
        offset: int = 0
    ) -> dict:
        """Fetch raw device history from API (single request).

        Note: This endpoint doesn't take a device ID parameter in the spec.

        Args:
            limit: Max records to fetch (max 10080)
            offset: Offset for pagination

        Returns:
            Raw API response dict
        """
        url = f"{self.base_url}/api/rest/device-history"
        params = {
            "limit": min(limit, MAX_RECORDS_PER_REQUEST),
            "offset": offset
        }
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()


# Cluster name to folder name mapping
CLUSTER_NAME_TO_FOLDER = {
    "Global Network": "global_network",
    "Cluster Cape Town ZA": "cluster_cape_town_za",
    "Cluster Edmonton": "cluster_edmonton",
    "Cluster Hong Kong": "cluster_hong_kong",
    "Cluster Hyderabad": "cluster_hyderabad",
    "Cluster Lagos": "cluster_lagos",
    "Cluster London": "cluster_london",
    "Cluster Los Angeles": "cluster_los_angeles",
    "Cluster Madrid": "cluster_madrid",
    "Cluster Mexico City": "cluster_mexico_city",
    "Cluster New York City": "cluster_new_york_city",
    "Cluster Puerto Rico": "cluster_puerto_rico",
    "Cluster Seoul Korea": "cluster_seoul_korea",
    "Cluster Stockolm": "cluster_stockolm",
    "Cluster São Paulo": "cluster_são_paulo",
    "Cluster Tel-Aviv": "cluster_tel_aviv",
    "Research Tower HMI N10A": "research_tower_hmi_n10a",
    "Research Tower Dispenza N10A": "research_tower_dispenza_n10a",
}

# Reverse mapping
FOLDER_TO_CLUSTER_NAME = {v: k for k, v in CLUSTER_NAME_TO_FOLDER.items()}


def folder_to_cluster_name(folder: str) -> str:
    """Convert folder name to API cluster name."""
    return FOLDER_TO_CLUSTER_NAME.get(folder, folder)


def cluster_name_to_folder(name: str) -> str:
    """Convert API cluster name to folder name."""
    if name in CLUSTER_NAME_TO_FOLDER:
        return CLUSTER_NAME_TO_FOLDER[name]
    # Fallback: convert to lowercase with underscores
    return name.lower().replace(" ", "_").replace("-", "_")


if __name__ == "__main__":
    # Test the client
    client = GCP2APIClient()

    print("=== Testing GCP2 API Client ===\n")

    # List clusters
    print("Clusters:")
    clusters = client.list_clusters()
    for c in sorted(clusters, key=lambda x: x["name"]):
        print(f"  - {c['name']}")

    print(f"\nTotal: {len(clusters)} clusters\n")

    # Fetch recent data from Global Network
    print("Fetching recent Global Network data (last 100 records)...")
    data = client.get_cluster_history_raw("Global Network", limit=100)
    history = data.get("cluster", {}).get("history", [])
    print(f"  Got {len(history)} records")
    if history:
        newest = datetime.fromtimestamp(history[0]["timestamp"], tz=timezone.utc)
        oldest = datetime.fromtimestamp(history[-1]["timestamp"], tz=timezone.utc)
        print(f"  Time range: {oldest} to {newest}")
        print(f"  Sample: {history[0]}")

    # Test DataFrame conversion
    print("\nTesting DataFrame conversion...")
    df = client.get_cluster_history("Global Network",
                                     start_ts=time.time() - 3600,  # last hour
                                     end_ts=time.time())
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    if not df.empty:
        print(f"  Sample:\n{df.head()}")
