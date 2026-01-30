"""
Dash web application to explore Global Consciousness Project (GCP) EGG data
stored in BigQuery and reproduce Nelson-style statistical analysis.

* Filtered egg list ({len(EGG_COLS_FILTERED)} columns, excludes {len(BROKEN_EGGS)} broken eggs)
* Implements correct GCP methodology:
  1. Stouffer Z across eggs: Z_t(s) = ΣZ_i/√N (dynamic N based on active eggs)
  2. χ² based on Stouffer Z: (Z_t(s))² (distributed as χ²(1) under null hypothesis)
  3. Cumulative deviation: Σ((Z_t(s))² - 1) to detect departure from randomness
* Uses published expected values: μ=100, σ=7.0712
* Handles missing egg data by dynamically adjusting N in Stouffer Z calculation
* Excludes broken eggs with constant output (zero variance) that cause systematic bias
* Guards against division-by-zero and empty windows
* Live slider read-outs
* Sliders fire the callback only on mouse-up
* Text input validation delayed until Enter key or focus loss to prevent interruption during typing
"""

import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime as _dt, timezone as _tz, timedelta as _td

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import diskcache as dc
from google.cloud import bigquery
from google.cloud.bigquery_storage_v1 import BigQueryReadClient
from pathlib import Path
from functools import lru_cache

# ───────────────────────────── configuration ──────────────────────────────
GCP_PROJECT  = os.getenv("GCP_PROJECT", "gcpingcp")
GCP_DATASET  = os.getenv("GCP_DATASET", "eggs_us")
GCP_TABLE    = os.getenv("GCP_TABLE", "basket")          # raw second-level table
BASELINE_TBL = os.getenv("BASELINE_TABLE", "baseline_individual_nozeros")

# Date range for sliders (3rd Aug 1998 to 31 Dec 2027)
DATE_MIN = _dt(1998, 8, 3, tzinfo=_tz.utc).date()
DATE_MAX = _dt(2027, 12, 31, tzinfo=_tz.utc).date()
# Default to start of 911 Nelson experiment (first plane hit WTC at 8:46 AM EDT = 12:46 PM UTC)
# DEFAULT_DATE = _dt(2011, 3, 11, tzinfo=_tz.utc).date()
# DEFAULT_TIME = _dt(2011, 3, 11, 5, 36, 0, tzinfo=_tz.utc).time()  # 2:46 PM JST = 05:36 AM UTC
DEFAULT_DATE = _dt(2001, 9, 11, tzinfo=_tz.utc).date()
DEFAULT_TIME = _dt(2001, 9, 11, 12, 35, 0, tzinfo=_tz.utc).time()  # 8:35 AM EDT = 12:35 PM UTC
# DEFAULT_WINDOW_LEN = 3600  # 1 hour default window
# DEFAULT_BINS = 3600        # 1 hour default bins
DEFAULT_WINDOW_LEN = 15000
DEFAULT_BINS = 15000

# Set default window length limits (will be updated by callback)
LEN_MIN_S, LEN_MAX_S = 60, 90 * 24 * 3600                     # Default: 1 min – 90 days (3 months)

BINS_MIN, BINS_MAX   = 1, 30000

CACHE = dc.Cache("./bq_cache", size_limit=2 * 1024**3)
# Track which parameter combinations have already been printed this runtime
_PRINTED_KEYS = set()

bq_client  = bigquery.Client(project=GCP_PROJECT)
bqs_client = BigQueryReadClient()

# ───────────────────────────── GCP2 configuration ──────────────────────────
GCP2_DATA_DIR = Path(__file__).parent.parent.parent / "gcp2.net-rng-data-downloaded"
GCP2_NETWORK_DIR = GCP2_DATA_DIR / "network"
GCP2_DEVICE_DIR = GCP2_DATA_DIR / "devices"
GCP2_DATE_MIN = _dt(2024, 3, 1, tzinfo=_tz.utc).date()
ROLLING_WINDOW_SECONDS = 3600  # 1 hour for rolling Z-score
MIN_ROLLING_PERIODS = 360      # 6 minutes minimum

# GCP2 network options (folder_name, display_name)
GCP2_NETWORKS = [
    ("global_network", "Global Network"),
    ("cluster_cape_town_za", "Cape Town, ZA"),
    ("cluster_edmonton", "Edmonton"),
    ("cluster_hong_kong", "Hong Kong"),
    ("cluster_hyderabad", "Hyderabad"),
    ("cluster_london", "London"),
    ("cluster_los_angeles", "Los Angeles"),
    ("cluster_madrid", "Madrid"),
    ("cluster_mexico_city", "Mexico City"),
    ("cluster_new_york_city", "New York City"),
    ("cluster_puerto_rico", "Puerto Rico"),
    ("cluster_são_paulo", "São Paulo"),
    ("cluster_seoul_korea", "Seoul, Korea"),
    ("cluster_stockolm", "Stockholm"),
    ("cluster_tel_aviv", "Tel Aviv"),
]

# GCP2 individual device options - dynamically populated + hardcoded examples
def get_gcp2_device_options():
    """Get list of available individual device IDs."""
    devices = []
    if GCP2_DEVICE_DIR.exists():
        for device_dir in sorted(GCP2_DEVICE_DIR.iterdir(), key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if device_dir.is_dir() and device_dir.name.isdigit():
                device_id = device_dir.name
                devices.append((f"device_{device_id}", f"Device {device_id}"))
    # Hardcoded device IDs (user-requested)
    hardcoded_devices = [
        ("device_498", "Device 498"),
    ]
    for folder, name in hardcoded_devices:
        if (folder, name) not in devices:
            devices.append((folder, name))
    # Sort devices by numeric ID
    devices.sort(key=lambda x: int(x[0].replace("device_", "")) if x[0].replace("device_", "").isdigit() else 999999)
    return devices

GCP2_DEVICES = get_gcp2_device_options()

def get_gcp2_latest_date():
    """Get the latest available GCP2 data date."""
    global_network_dir = GCP2_NETWORK_DIR / "global_network"
    if not global_network_dir.exists():
        return "2024-present"

    latest_year = 0
    latest_month = 0

    for year_dir in global_network_dir.iterdir():
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue

        for csv_file in year_dir.glob("*.csv"):
            if ".csv.zip" in csv_file.name:
                continue
            parts = csv_file.stem.split("_")
            try:
                month = int(parts[-1])
                if year > latest_year or (year == latest_year and month > latest_month):
                    latest_year = year
                    latest_month = month
            except (ValueError, IndexError):
                continue

    if latest_year > 0:
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        return f"{month_names[latest_month]} {latest_year}"
    return "2024-present"

GCP2_LATEST_DATE = get_gcp2_latest_date()

def get_gcp1_latest_date():
    """Query BigQuery for the actual latest GCP1 data date."""
    try:
        query = f"""
        SELECT MAX(recorded_at) as latest_date
        FROM `{GCP_PROJECT}.{GCP_DATASET}.{GCP_TABLE}`
        """
        result = bq_client.query(query).result()
        for row in result:
            if row.latest_date:
                return row.latest_date.strftime('%b %Y')
        return DATE_MAX.strftime('%b %Y')
    except Exception as e:
        print(f"Error querying GCP1 latest date: {e}")
        return DATE_MAX.strftime('%b %Y')

GCP1_LATEST_DATE = get_gcp1_latest_date()

# ───────────────────────────── egg column list ─────────────────────────────
EGG_COLS = [
  "egg_1","egg_28","egg_33","egg_34","egg_37","egg_100","egg_101","egg_102","egg_103","egg_104",
  "egg_105","egg_106","egg_107","egg_108","egg_109","egg_110","egg_111","egg_112","egg_114","egg_115",
  "egg_116","egg_117","egg_118","egg_119","egg_134","egg_142","egg_161","egg_223","egg_224","egg_226",
  "egg_227","egg_228","egg_230","egg_231","egg_233","egg_237","egg_1000","egg_1003","egg_1004","egg_1005",
  "egg_1013","egg_1021","egg_1022","egg_1023","egg_1024","egg_1025","egg_1026","egg_1027","egg_1029",
  "egg_1051","egg_1063","egg_1066","egg_1070","egg_1082","egg_1092","egg_1095","egg_1096","egg_1101",
  "egg_1113","egg_1223","egg_1237","egg_1245","egg_1251","egg_1295","egg_2000","egg_2001","egg_2002",
  "egg_2006","egg_2007","egg_2008","egg_2009","egg_2013","egg_2022","egg_2023","egg_2024","egg_2026",
  "egg_2027","egg_2028","egg_2040","egg_2041","egg_2042","egg_2043","egg_2044","egg_2045","egg_2046",
  "egg_2047","egg_2048","egg_2049","egg_2052","egg_2060","egg_2061","egg_2062","egg_2064","egg_2069",
  "egg_2070","egg_2073","egg_2080","egg_2083","egg_2084","egg_2088","egg_2091","egg_2093","egg_2094",
  "egg_2097","egg_2120","egg_2165","egg_2173","egg_2178","egg_2201","egg_2202","egg_2220","egg_2221",
  "egg_2222","egg_2225","egg_2230","egg_2231","egg_2232","egg_2234","egg_2235","egg_2236","egg_2239",
  "egg_2240","egg_2241","egg_2242","egg_2243","egg_2244","egg_2247","egg_2248","egg_2249","egg_2250",
  "egg_3005","egg_3023","egg_3043","egg_3045","egg_3066","egg_3101","egg_3103","egg_3104","egg_3106",
  "egg_3107","egg_3108","egg_3115","egg_3142","egg_3240","egg_3247","egg_4002","egg_4101","egg_4234",
  "egg_4251"
]

# Filtered egg columns excluding broken eggs (constant output = 0, Z-score = -14.1419)
# Based on analysis of 2011-03-11 data showing these eggs have zero variance
BROKEN_EGGS = [
    # "egg_2088", "egg_2249", "egg_2243", "egg_2236", 
    # "egg_2047", "egg_2024", "egg_2002", "egg_1237"
]

EGG_COLS_FILTERED = [egg for egg in EGG_COLS if egg not in BROKEN_EGGS]

# ───────────────────────────── SQL builder ────────────────────────────────
def build_sql(filter_broken_eggs: bool = True, use_pseudo_entropy: bool = False) -> str:
    if use_pseudo_entropy:
        # Replace all EGG values with binomial random sums (200 trials, p=0.5)
        # This simulates the actual EGG data structure: sums of 200 binary trials
        # Expected value = 200 * 0.5 = 100, variance = 200 * 0.5 * 0.5 = 50, std dev = sqrt(50) ≈ 7.07
        ascii_block = ",\n".join(f"    (SELECT SUM(CAST(RAND() > 0.5 AS INT64)) FROM UNNEST(GENERATE_ARRAY(1, 200)) AS trial) AS {c}" for c in EGG_COLS_FILTERED)
    else:
        ascii_block = ",\n".join(f"    ASCII({c}) AS {c}" for c in EGG_COLS_FILTERED)  # still used in raw CTE, keeps query readable
    
    # Use published expected values: μ=100, σ=7.0712
    # Keep ASCII conversion as requested
    # Only calculate Z-scores for eggs with non-NULL data
    # Optionally filter out 0 values (broken eggs)
    z_block_terms = []
    for c in EGG_COLS_FILTERED:
        if filter_broken_eggs:
            # Filter out 0 values when checkbox is enabled
            z_block_terms.append(f"    IF({c} IS NOT NULL AND {c} != 0, SAFE_DIVIDE(({c} - 100), 7.0712), NULL) AS z_{c}")
        else:
            # Include all non-NULL values (including 0)
            z_block_terms.append(f"    IF({c} IS NOT NULL, SAFE_DIVIDE(({c} - 100), 7.0712), NULL) AS z_{c}")
    z_block = ",\n".join(z_block_terms)
    
    # Count non-null eggs for dynamic N calculation
    null_count_block = " + ".join(f"IF(z_{c} IS NULL, 0, 1)" for c in EGG_COLS_FILTERED)
    
    # Calculate Stouffer Z with dynamic N (only include non-null eggs)
    stouffer_z_terms = []
    for c in EGG_COLS_FILTERED:
        stouffer_z_terms.append(f"IF(z_{c} IS NOT NULL, z_{c}, 0)")
    stouffer_z_sum = " + ".join(stouffer_z_terms)
    stouffer_z = f"SAFE_DIVIDE({stouffer_z_sum}, SQRT({null_count_block})) AS stouffer_z"
    
    # Calculate χ² based on Stouffer Z: (Stouffer Z)² - 1 (deviation from null hypothesis)
    chi2_stouffer = f"POW(SAFE_DIVIDE({stouffer_z_sum}, SQRT({null_count_block})), 2) - 1 AS chi2_stouffer"

    return f"""
DECLARE start_ts TIMESTAMP DEFAULT @start_ts;
DECLARE window_s INT64     DEFAULT @window_s;
DECLARE bins     INT64     DEFAULT @bins;
DECLARE sec_per_bin INT64 DEFAULT
    IF(bins >= window_s, 1, CAST(window_s / bins AS INT64));

WITH raw AS (
  SELECT recorded_at,
{ascii_block}
  FROM `{GCP_PROJECT}.{GCP_DATASET}.{GCP_TABLE}`
  WHERE recorded_at BETWEEN start_ts
                        AND TIMESTAMP_ADD(start_ts, INTERVAL window_s SECOND)
),
z AS (
  SELECT recorded_at,
{z_block}
  FROM raw
),
sec AS (
  SELECT recorded_at, 
         {chi2_stouffer},
         {stouffer_z},
         {null_count_block} AS active_eggs
  FROM z
),
bins AS (
  SELECT
    CAST(FLOOR(TIMESTAMP_DIFF(recorded_at, start_ts, SECOND)/sec_per_bin) AS INT64) AS bin_idx,
    SUM(chi2_stouffer) AS chi2_stouffer_sum,
    COUNT(*) AS seconds_in_bin,
    AVG(active_eggs) AS avg_active_eggs
  FROM sec
  GROUP BY bin_idx
)
SELECT bin_idx, chi2_stouffer_sum, seconds_in_bin, avg_active_eggs
FROM bins
ORDER BY bin_idx;"""

def render_sql(start_ts: float, window_s: int, bins: int, use_pseudo_entropy: bool = False) -> str:
    """Return a BigQuery query with ALL parameters inlined as literals.

    The DECLARE block is removed and any references to its variables are
    substituted so the output can be copy-pasted directly into the BigQuery
    console with no additional parameter binding.  (Requested by user.)
    """
    import re

    sql = build_sql(use_pseudo_entropy=use_pseudo_entropy)

    # We'll inline only the parameter placeholders (marked with '@').
    # This avoids touching identifiers like the CTE name `bins`.

    ts_literal = _dt.fromtimestamp(start_ts, _tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    sql = sql.replace("@start_ts", f"TIMESTAMP '{ts_literal}'")
    sql = sql.replace("@window_s", str(window_s))
    sql = sql.replace("@bins", str(bins))

    # Replace the computed sec_per_bin expression directly in the DECLARE block
    sec_per_bin = 1 if bins >= window_s else window_s // bins
    sql = re.sub(r"DECLARE sec_per_bin INT64 DEFAULT\s+[^;]+;",
                 f"DECLARE sec_per_bin INT64 DEFAULT {sec_per_bin};",
                 sql)

    return sql

# ───────────────────────────── significance envelope ───────────────────────
def compute_pointwise_p05_envelope(cumulative_seconds: pd.Series) -> np.ndarray:
    """Return the pointwise two-sided 95% (p=0.05) envelope for the cumulative
    sum S(T) = Σ((Stouffer Z)^2 − 1), evaluated at cumulative elapsed seconds T.

    Under the null, per-second increments X_t = (Stouffer Z_t)^2 − 1 have mean 0
    and variance Var(χ²(1) − 1) = 2. Therefore Var[S(T)] ≈ 2T, so a pointwise
    two-sided p = 0.05 band is ± z_{0.975} · sqrt(2T).
    """
    z_0_975 = 1.959963984540054
    cum_seconds_clipped = np.maximum(cumulative_seconds.to_numpy(dtype=float), 0.0)
    return z_0_975 * np.sqrt(2.0 * cum_seconds_clipped)

# ───────────────────────────── query helper ────────────────────────────────
CACHE = dc.Cache("./bq_cache", size_limit=2 * 1024**3)

def get_data_range_info() -> tuple[str, str, int]:
    """Get first recorded date, last recorded date, and total row count from the basket table.
    
    Returns:
        tuple: (first_date_str, last_date_str, total_count)
    """
    # Cache key for data range info (version 2 to avoid old format conflicts)
    cache_key = "data_range_info_v2"
    
    # Check if we have cached data range info
    cached_info = CACHE.get(cache_key)
    if cached_info is not None:
        # Verify the cached data is in the new format (should be strings, not SQL)
        if isinstance(cached_info[0], str) and not cached_info[0].startswith("SELECT"):
            return cached_info
        else:
            # Old format cached - clear it and regenerate
            CACHE.delete(cache_key)
    
    # SQL to find first recorded date
    first_date_sql = f"SELECT recorded_at FROM `{GCP_PROJECT}.{GCP_DATASET}.{GCP_TABLE}` ORDER BY recorded_at ASC LIMIT 1"
    
    # SQL to find last recorded date  
    last_date_sql = f"SELECT recorded_at FROM `{GCP_PROJECT}.{GCP_DATASET}.{GCP_TABLE}` ORDER BY recorded_at DESC LIMIT 1"
    
    # SQL to get total row count
    count_sql = f"SELECT COUNT(*) FROM `{GCP_PROJECT}.{GCP_DATASET}.{GCP_TABLE}`"
    
    try:
        # Execute queries to get actual values
        first_date_result = bq_client.query(first_date_sql).to_dataframe()
        last_date_result = bq_client.query(last_date_sql).to_dataframe()
        count_result = bq_client.query(count_sql).to_dataframe()
        
        # Format the dates nicely
        first_date = first_date_result.iloc[0, 0] if not first_date_result.empty else "Unknown"
        last_date = last_date_result.iloc[0, 0] if not last_date_result.empty else "Unknown"
        total_count = count_result.iloc[0, 0] if not count_result.empty else 0
        
        # Format dates as strings if they're datetime objects
        if hasattr(first_date, 'strftime'):
            first_date_str = first_date.strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            first_date_str = str(first_date)
            
        if hasattr(last_date, 'strftime'):
            last_date_str = last_date.strftime('%Y-%m-%d %H:%M:%S UTC')
        else:
            last_date_str = str(last_date)
        
        result = (first_date_str, last_date_str, total_count)
        # Cache for 1 hour since this data doesn't change frequently
        CACHE.set(cache_key, result, expire=3600)
        return result
    except Exception as e:
        print(f"Error getting data range info: {e}")
        result = ("Unknown", "Unknown", 0)
        # Cache the error result for a shorter time
        CACHE.set(cache_key, result, expire=300)
        return result

def query_bq(start_ts: float, window_s: int, bins: int, filter_broken_eggs: bool = True, use_pseudo_entropy: bool = False) -> pd.DataFrame:
    key = (start_ts, window_s, bins, filter_broken_eggs, use_pseudo_entropy)
    # fetch or compute dataframe
    df = CACHE.get(key)
    if df is None:
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_ts", "TIMESTAMP",
                                              _dt.fromtimestamp(start_ts, _tz.utc)),
                bigquery.ScalarQueryParameter("window_s", "INT64", window_s),
                bigquery.ScalarQueryParameter("bins",     "INT64", bins)
            ]
        )
        sql = build_sql(filter_broken_eggs, use_pseudo_entropy)
        df = (
            bq_client.query(sql, job_config=cfg)
            .to_dataframe(bqstorage_client=bqs_client, create_bqstorage_client=True)
        )
        df["cum_stouffer_z"] = df["chi2_stouffer_sum"].cumsum()
        CACHE.set(key, df, expire=3600)

    # Always print SQL and result for debugging and verification
    print("\n===== BigQuery SQL =====\n" + render_sql(start_ts, window_s, bins, use_pseudo_entropy) + "\n========================")
    print(df.head(10).to_string(index=False))
    print("========================\n")

    return df

# ───────────────────────────── GCP2 data loading ────────────────────────────
@lru_cache(maxsize=32)
def get_gcp2_available_months(network: str) -> list[tuple[int, int, Path]]:
    """Get list of (year, month, csv_path) tuples for available GCP2 data."""
    network_path = GCP2_NETWORK_DIR / network
    if not network_path.exists():
        return []

    months = []
    for year_dir in sorted(network_path.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        for csv_file in sorted(year_dir.glob("*.csv")):
            # Skip ZIP files
            if csv_file.suffix == ".zip" or ".csv.zip" in csv_file.name:
                continue
            # Parse month from filename: GCP2_Network_Coherence_{network}_{year}_{month}.csv
            parts = csv_file.stem.split("_")
            try:
                month = int(parts[-1])
                months.append((year, month, csv_file))
            except (ValueError, IndexError):
                continue
    return months


def get_gcp2_monthly_files(network: str, start_ts: float, end_ts: float) -> list[Path]:
    """Get list of monthly CSV files that cover the requested time range."""
    start_dt = _dt.fromtimestamp(start_ts, _tz.utc)
    end_dt = _dt.fromtimestamp(end_ts, _tz.utc)

    files = []
    for year, month, csv_path in get_gcp2_available_months(network):
        file_start = _dt(year, month, 1, tzinfo=_tz.utc)
        # Get first day of next month
        if month == 12:
            file_end = _dt(year + 1, 1, 1, tzinfo=_tz.utc)
        else:
            file_end = _dt(year, month + 1, 1, tzinfo=_tz.utc)

        # Check overlap with requested range
        if file_start < end_dt and file_end > start_dt:
            files.append(csv_path)

    return files


def load_device_data(device_id: str, start_ts: float, end_ts: float) -> pd.DataFrame:
    """Load individual device data from ZIP files.

    Device data format: device_number, epoch_time_utc, active_seconds, device_coherence, significance
    """
    import zipfile

    device_dir = GCP2_DEVICE_DIR / device_id
    if not device_dir.exists():
        print(f"Device directory not found: {device_dir}")
        return pd.DataFrame()

    frames = []

    # Load History and Latest ZIP files
    for zip_pattern in ["*_History.csv.zip", "*_Latest.csv.zip"]:
        for zip_path in device_dir.glob(zip_pattern):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f)
                        if not df.empty:
                            frames.append(df)
            except Exception as e:
                print(f"Error loading {zip_path}: {e}")
                continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['epoch_time_utc'])
    combined.sort_values("epoch_time_utc", inplace=True)

    # Filter to time range
    mask = (combined["epoch_time_utc"] >= start_ts) & (combined["epoch_time_utc"] < end_ts)
    data = combined[mask].copy()

    if data.empty:
        return pd.DataFrame()

    # Rename device_coherence to network_coherence for compatibility
    # and add active_devices = 1 for single device
    data = data.rename(columns={"device_coherence": "network_coherence"})
    data["active_devices"] = 1

    return data


def query_gcp2(start_ts: float, window_s: int, bins: int, network: str = "global_network") -> pd.DataFrame:
    """Query GCP2 network coherence data for the specified time window.

    Returns DataFrame with columns:
        - bin_idx: Bin index
        - chi2_rolling_z_sum: Sum of (rolling_z)^2 - 1 in bin
        - nc_sum: Sum of network_coherence in bin
        - seconds_in_bin: Count of seconds in bin
        - avg_active_devices: Average active devices
        - cum_rolling_z: Cumulative sum of (rolling_z)^2 - 1
        - cum_nc: Cumulative sum of network_coherence
    """
    cache_key = ("gcp2", start_ts, window_s, bins, network)
    cached = CACHE.get(cache_key)
    if cached is not None:
        print(f"\n===== GCP2 Data (cached) =====")
        print(f"Source: {network}, Rows: {len(cached)}")
        print("==============================\n")
        return cached

    end_ts = start_ts + window_s

    # Check if this is an individual device or a network/cluster
    is_device = network.startswith("device_")

    if is_device:
        # Load individual device data from ZIP files
        device_id = network.replace("device_", "")
        data = load_device_data(device_id, start_ts, end_ts)
        if data.empty:
            print(f"\n===== GCP2 Device Data =====")
            print(f"No data found for Device {device_id} in range {_dt.fromtimestamp(start_ts, _tz.utc)} to {_dt.fromtimestamp(end_ts, _tz.utc)}")
            print("============================\n")
            return pd.DataFrame()
        source_info = f"Device {device_id}"
    else:
        # Get relevant monthly CSV files for network/cluster
        csv_files = get_gcp2_monthly_files(network, start_ts, end_ts)
        if not csv_files:
            print(f"\n===== GCP2 Data =====")
            print(f"No data files found for {network} in range {_dt.fromtimestamp(start_ts, _tz.utc)} to {_dt.fromtimestamp(end_ts, _tz.utc)}")
            print("=====================\n")
            return pd.DataFrame()

        # Load and concatenate data
        frames = []
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                frames.append(df)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                continue

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values("epoch_time_utc", inplace=True)

        # Filter to exact time window
        mask = (combined["epoch_time_utc"] >= start_ts) & (combined["epoch_time_utc"] < end_ts)
        data = combined[mask].copy()

        if data.empty:
            print(f"\n===== GCP2 Data =====")
            print(f"No data in time range for {network}")
            print("=====================\n")
            return pd.DataFrame()

        source_info = f"{network}, Files: {len(csv_files)}"

    # Compute rolling Z-score of network_coherence
    nc = data["network_coherence"]
    roll = nc.rolling(ROLLING_WINDOW_SECONDS, min_periods=MIN_ROLLING_PERIODS)
    data["rolling_z"] = (nc - roll.mean()) / (roll.std(ddof=0) + 1e-9)

    # Compute chi-square style deviation: (rolling_z)^2 - 1
    data["chi2_rolling_z"] = data["rolling_z"] ** 2 - 1

    # Bin the data
    seconds_per_bin = max(1, window_s // bins)
    data["bin_idx"] = ((data["epoch_time_utc"] - start_ts) // seconds_per_bin).astype(int)

    # Aggregate by bin
    binned = data.groupby("bin_idx").agg(
        nc_sum=("network_coherence", "sum"),
        chi2_rolling_z_sum=("chi2_rolling_z", lambda x: x.dropna().sum()),
        seconds_in_bin=("network_coherence", "count"),
        avg_active_devices=("active_devices", "mean"),
    ).reset_index()

    # Compute cumulative sums
    binned["cum_nc"] = binned["nc_sum"].cumsum()
    binned["cum_rolling_z"] = binned["chi2_rolling_z_sum"].cumsum()

    print(f"\n===== GCP2 Data =====")
    print(f"Source: {source_info}, Rows: {len(binned)}")
    print(binned.head(10).to_string(index=False))
    print("=====================\n")

    CACHE.set(cache_key, binned, expire=3600)
    return binned


def validate_gcp2_date_range(start_ts: float, window_s: int) -> tuple[bool, str]:
    """Check if selected date range has GCP2 data available.

    Returns:
        (has_data, warning_message)
    """
    start_date = _dt.fromtimestamp(start_ts, _tz.utc).date()
    end_date = _dt.fromtimestamp(start_ts + window_s, _tz.utc).date()

    if start_date < GCP2_DATE_MIN:
        return False, f"⚠ GCP2 data starts March 2024 (selected: {start_date})"

    return True, ""

# ───────────────────────────── Dash layout ─────────────────────────────────
app = dash.Dash(__name__)
app.title = "Global Consciousness Project RNG Statistical Analysis Explorer"

# Add CSS for pulsing slider animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @keyframes dateSliderPulse {
                0% { background: linear-gradient(90deg, #00d4ff 0%, #4de6ff 25%, #99f2ff 50%, #4de6ff 75%, #00d4ff 100%); }
                20% { background: linear-gradient(90deg, #4de6ff 0%, #99f2ff 25%, #00d4ff 50%, #4de6ff 75%, #99f2ff 100%); }
                40% { background: linear-gradient(90deg, #99f2ff 0%, #00d4ff 25%, #4de6ff 50%, #99f2ff 75%, #00d4ff 100%); }
                60% { background: linear-gradient(90deg, #00d4ff 0%, #4de6ff 25%, #99f2ff 50%, #4de6ff 75%, #00d4ff 100%); }
                80% { background: linear-gradient(90deg, #4de6ff 0%, #99f2ff 25%, #00d4ff 50%, #4de6ff 75%, #99f2ff 100%); }
                100% { background: linear-gradient(90deg, #00d4ff 0%, #4de6ff 25%, #99f2ff 50%, #4de6ff 75%, #00d4ff 100%); }
            }
            
            @keyframes timeSliderPulse {
                0% { background: linear-gradient(90deg, #ff006e 0%, #ff6b9d 25%, #ff9ecd 50%, #ff6b9d 75%, #ff006e 100%); }
                20% { background: linear-gradient(90deg, #ff6b9d 0%, #ff9ecd 25%, #ff006e 50%, #ff6b9d 75%, #ff9ecd 100%); }
                40% { background: linear-gradient(90deg, #ff9ecd 0%, #ff006e 25%, #ff6b9d 50%, #ff9ecd 75%, #ff006e 100%); }
                60% { background: linear-gradient(90deg, #ff006e 0%, #ff6b9d 25%, #ff9ecd 50%, #ff6b9d 75%, #ff006e 100%); }
                80% { background: linear-gradient(90deg, #ff6b9d 0%, #ff9ecd 25%, #ff006e 50%, #ff6b9d 75%, #ff9ecd 100%); }
                100% { background: linear-gradient(90deg, #ff006e 0%, #ff6b9d 25%, #ff9ecd 50%, #ff6b9d 75%, #ff006e 100%); }
            }
            
            @keyframes lengthSliderPulse {
                0% { background: linear-gradient(90deg, #00ff88 0%, #4dffa3 25%, #99ffbe 50%, #4dffa3 75%, #00ff88 100%); }
                20% { background: linear-gradient(90deg, #4dffa3 0%, #99ffbe 25%, #00ff88 50%, #4dffa3 75%, #99ffbe 100%); }
                40% { background: linear-gradient(90deg, #99ffbe 0%, #00ff88 25%, #4dffa3 50%, #99ffbe 75%, #00ff88 100%); }
                60% { background: linear-gradient(90deg, #00ff88 0%, #4dffa3 25%, #99ffbe 50%, #4dffa3 75%, #00ff88 100%); }
                80% { background: linear-gradient(90deg, #4dffa3 0%, #99ffbe 25%, #00ff88 50%, #4dffa3 75%, #99ffbe 100%); }
                100% { background: linear-gradient(90deg, #00ff88 0%, #4dffa3 25%, #99ffbe 50%, #4dffa3 75%, #00ff88 100%); }
            }
            
            @keyframes binsSliderPulse {
                0% { background: linear-gradient(90deg, #9d4edd 0%, #b366e6 25%, #c980ef 50%, #b366e6 75%, #9d4edd 100%); }
                20% { background: linear-gradient(90deg, #b366e6 0%, #c980ef 25%, #9d4edd 50%, #b366e6 75%, #c980ef 100%); }
                40% { background: linear-gradient(90deg, #c980ef 0%, #9d4edd 25%, #b366e6 50%, #c980ef 75%, #9d4edd 100%); }
                60% { background: linear-gradient(90deg, #9d4edd 0%, #b366e6 25%, #c980ef 50%, #b366e6 75%, #9d4edd 100%); }
                80% { background: linear-gradient(90deg, #b366e6 0%, #c980ef 25%, #9d4edd 50%, #b366e6 75%, #c980ef 100%); }
                100% { background: linear-gradient(90deg, #9d4edd 0%, #b366e6 25%, #c980ef 50%, #b366e6 75%, #9d4edd 100%); }
            }
            
            .date-slider .rc-slider-track {
                animation: dateSliderPulse 3.2s linear infinite;
                box-shadow: 0 0 15px rgba(255, 0, 110, 0.6);
            }
            
            .time-slider .rc-slider-track {
                animation: timeSliderPulse 2.4s linear infinite;
                box-shadow: 0 0 15px rgba(0, 212, 255, 0.6);
            }
            
            .length-slider .rc-slider-track {
                animation: lengthSliderPulse 4s linear infinite;
                box-shadow: 0 0 15px rgba(157, 78, 221, 0.6);
            }
            
            .bins-slider .rc-slider-track {
                animation: binsSliderPulse 2s linear infinite;
                box-shadow: 0 0 15px rgba(0, 255, 136, 0.6);
            }
            
            .date-slider .rc-slider-handle {
                background: #00d4ff !important;
                border: 2px solid #00d4ff !important;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.8) !important;
            }
            
            .time-slider .rc-slider-handle {
                background: #ff006e !important;
                border: 2px solid #ff006e !important;
                box-shadow: 0 0 10px rgba(255, 0, 110, 0.8) !important;
            }
            
            .length-slider .rc-slider-handle {
                background: #00ff88 !important;
                border: 2px solid #00ff88 !important;
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.8) !important;
            }
            
            .bins-slider .rc-slider-handle {
                background: #9d4edd !important;
                border: 2px solid #9d4edd !important;
                box-shadow: 0 0 10px rgba(157, 78, 221, 0.8) !important;
            }
            
            .date-slider .rc-slider-handle:hover {
                transform: scale(1.2);
                box-shadow: 0 0 20px rgba(0, 212, 255, 1) !important;
            }
            
            .time-slider .rc-slider-handle:hover {
                transform: scale(1.2);
                box-shadow: 0 0 20px rgba(255, 0, 110, 1) !important;
            }
            
            .length-slider .rc-slider-handle:hover {
                transform: scale(1.2);
                box-shadow: 0 0 20px rgba(0, 255, 136, 1) !important;
            }
            
            .bins-slider .rc-slider-handle:hover {
                transform: scale(1.2);
            }
            
            /* Cache clear button hover effects */
            #clear-cache-btn:hover {
                background-color: #ffbe0b !important;
                color: #000000 !important;
                box-shadow: 0 0 20px rgba(255, 190, 11, 0.8) !important;
                transform: scale(1.05);
            }
            
            #clear-cache-btn:active {
                transform: scale(0.95);
                box-shadow: 0 0 15px rgba(255, 190, 11, 0.6) !important;
            }

            /* GCP2 Network Dropdown Styling */
            #gcp2-network-select {
                background-color: #1a1a2e !important;
            }
            #gcp2-network-select .Select-control {
                background-color: #1a1a2e !important;
                border: 1px solid #ffbe0b !important;
            }
            #gcp2-network-select .Select-value-label,
            #gcp2-network-select .Select-placeholder {
                color: #ffffff !important;
            }
            #gcp2-network-select .Select-menu-outer {
                background-color: #1a1a2e !important;
                border: 1px solid #ffbe0b !important;
            }
            #gcp2-network-select .VirtualizedSelectOption {
                background-color: #1a1a2e !important;
                color: #ffffff !important;
            }
            #gcp2-network-select .VirtualizedSelectFocusedOption {
                background-color: #ffbe0b !important;
                color: #0a0a0f !important;
            }
            /* React-Select v2+ styles used by Dash */
            #gcp2-network-select .Select-single-value,
            #gcp2-network-select input {
                color: #ffffff !important;
            }
            #gcp2-network-select .Select-arrow {
                border-color: #ffbe0b transparent transparent !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Cyberpunk psychedelic color scheme
CYBERPUNK_COLORS = {
    'bg_dark': '#0a0a0f',
    'bg_medium': '#1a1a2e', 
    'bg_light': '#16213e',
    'neon_pink': '#ff006e',
    'neon_cyan': '#00d4ff',
    'neon_purple': '#9d4edd',
    'neon_green': '#00ff88',
    'neon_yellow': '#ffbe0b',
    'text_primary': '#ffffff',
    'text_secondary': '#b8b8b8',
    'accent_gradient': 'linear-gradient(135deg, #ff006e 0%, #9d4edd 50%, #00d4ff 100%)'
}

# Game-like pulsing color animations for sliders
SLIDER_ANIMATIONS = {
    'date_slider': {
        'colors': ['#ff006e', '#ff6b9d', '#ff9ecd', '#ff006e'],  # Pink to light pink cycle
        'duration': '3s'
    },
    'time_slider': {
        'colors': ['#00d4ff', '#4de6ff', '#99f2ff', '#00d4ff'],  # Cyan to light cyan cycle
        'duration': '2.5s'
    },
    'length_slider': {
        'colors': ['#9d4edd', '#b366e6', '#c980ef', '#9d4edd'],  # Purple to light purple cycle
        'duration': '3.5s'
    },
    'bins_slider': {
        'colors': ['#00ff88', '#4dffa3', '#99ffbe', '#00ff88'],  # Green to light green cycle
        'duration': '2s'
    }
}

app.layout = html.Div([
    # Location component to capture URL parameters
    dcc.Location(id='url', refresh=False),
    
    # Main container with cyberpunk styling
    html.Div([
        # Header with glowing effect
        html.Div([
            html.H1("GLOBAL CONSCIOUSNESS PROJECT RNG STATISTICAL ANALYSIS EXPLORER", 
                   style={
                       "textAlign": "center",
                       "color": CYBERPUNK_COLORS['text_primary'],
                       "fontSize": "2.5rem",
                       "fontWeight": "900",
                       "textShadow": f"0 0 20px {CYBERPUNK_COLORS['neon_pink']}, 0 0 40px {CYBERPUNK_COLORS['neon_pink']}",
                       "marginBottom": "10px",
                       "fontFamily": "'Orbitron', 'Courier New', monospace",
                       "letterSpacing": "3px"
                   }),
            html.P([
                "NEURAL INTERFACE: Cumulative deviation of χ² based on Stouffer Z (detects departure from randomness)"
            ], style={
                "fontSize": "16px", 
                "color": CYBERPUNK_COLORS['neon_cyan'],
                "marginBottom": "20px",
                "textAlign": "center",
                "fontFamily": "'Courier New', monospace",
                "textShadow": f"0 0 10px {CYBERPUNK_COLORS['neon_cyan']}"
            }),
        ], style={
            "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
            "padding": "20px",
            "borderRadius": "15px",
            "border": f"2px solid {CYBERPUNK_COLORS['neon_pink']}",
            "boxShadow": f"0 0 30px {CYBERPUNK_COLORS['neon_pink']}40",
            "marginBottom": "30px"
        }),
        
        # Loading wrapper around the graph with cyberpunk styling
        html.Div([
            dcc.Loading(
                id="loading-graph",
                type="circle",
                color=CYBERPUNK_COLORS['neon_cyan'],
                children=[
                    dcc.Graph(
                        id="chi2-graph", 
                        style={
                            "height": "70vh",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "borderRadius": "10px",
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}",
                            "boxShadow": f"0 0 20px {CYBERPUNK_COLORS['neon_cyan']}30"
                        }
                    )
                ]
            )
        ], style={"marginBottom": "30px"}),

        # Basket Dataset Options - moved here, directly below graph
        html.Div([
            html.H4("Basket Dataset Options", style={
                "color": CYBERPUNK_COLORS['neon_cyan'],
                "marginBottom": "15px",
                "fontFamily": "'Orbitron', monospace",
                "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_cyan']}"
            }),
            html.Div([
                dcc.Checklist(
                    id="filter-broken-eggs",
                    options=[{"label": "Filter out broken EGGs with 0-trial sums", "value": "enabled"}],
                    value=["enabled"],  # Default to enabled
                    style={
                        "color": CYBERPUNK_COLORS['text_primary'],
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "14px"
                    }
                ),
                dcc.Checklist(
                    id="pseudo-entropy",
                    options=[{"label": "Overlay pseudo entropy (random values)", "value": "enabled"}],
                    value=[],  # Default to disabled
                    style={
                        "color": CYBERPUNK_COLORS['text_primary'],
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "14px",
                        "marginTop": "10px"
                    }
                ),
                dcc.Checklist(
                    id="show-parabolic-curve",
                    options=[{"label": "Show parabolic probability curve", "value": "enabled"}],
                    value=["enabled"],  # Default to enabled
                    style={
                        "color": CYBERPUNK_COLORS['text_primary'],
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "14px",
                        "marginTop": "10px"
                    }
                )
            ], style={"marginBottom": "15px"}),
            # Clear cache button on the right side
            html.Div([
                html.Button(
                    "CLEAR CACHE & REFRESH",
                    id="clear-cache-btn",
                    n_clicks=0,
                    style={
                        "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                        "color": CYBERPUNK_COLORS['neon_yellow'],
                        "border": f"2px solid {CYBERPUNK_COLORS['neon_yellow']}",
                        "borderRadius": "8px",
                        "padding": "10px 20px",
                        "fontSize": "14px",
                        "fontWeight": "bold",
                        "fontFamily": "'Courier New', monospace",
                        "cursor": "pointer",
                        "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_yellow']}",
                        "boxShadow": f"0 0 10px {CYBERPUNK_COLORS['neon_yellow']}30",
                        "transition": "all 0.3s ease"
                    }
                )
            ], style={"display": "flex", "alignItems": "center"})
        ], style={
            "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
            "padding": "20px",
            "borderRadius": "15px",
            "border": f"2px solid {CYBERPUNK_COLORS['neon_cyan']}",
            "boxShadow": f"0 0 30px {CYBERPUNK_COLORS['neon_cyan']}40",
            "marginBottom": "30px",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center"
        }),

        # Data Source Selection - GCP1 and/or GCP2
        html.Div([
            html.H4("DATA SOURCE SELECTION", style={
                "color": CYBERPUNK_COLORS['neon_yellow'],
                "marginBottom": "15px",
                "fontFamily": "'Orbitron', monospace",
                "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_yellow']}"
            }),
            html.Div([
                # Data source checkboxes
                dcc.Checklist(
                    id="data-source-select",
                    options=[
                        {"label": f" GCP1 (BigQuery RNG Data 1998-{GCP1_LATEST_DATE})", "value": "gcp1"},
                        {"label": f" GCP2 (Network Coherence Mar 2024-{GCP2_LATEST_DATE})", "value": "gcp2"},
                    ],
                    value=["gcp1"],  # Default to GCP1 only
                    inline=True,
                    style={
                        "color": CYBERPUNK_COLORS['text_primary'],
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "14px"
                    },
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"marginRight": "20px"}
                ),
            ], style={"marginBottom": "15px"}),

            # GCP2 Options (conditionally visible)
            html.Div(id="gcp2-options-container", children=[
                html.Div([
                    html.Label("GCP2 Source:", style={
                        "color": CYBERPUNK_COLORS['text_primary'],
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "13px",
                        "marginRight": "10px"
                    }),
                    dcc.Dropdown(
                        id="gcp2-network-select",
                        options=(
                            # Networks/Clusters section
                            [{"label": "── Networks ──", "value": "_header_networks", "disabled": True}] +
                            [{"label": name, "value": folder} for folder, name in GCP2_NETWORKS] +
                            # Individual Devices section (all 473 devices)
                            [{"label": "── Individual Devices ──", "value": "_header_devices", "disabled": True}] +
                            [{"label": name, "value": folder} for folder, name in GCP2_DEVICES]
                        ),
                        value="global_network",
                        clearable=False,
                        searchable=True,  # Enable search for easy device lookup
                        style={
                            "width": "250px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                        }
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginRight": "30px"}),

                html.Div([
                    html.Label("Display Mode:", style={
                        "color": CYBERPUNK_COLORS['text_primary'],
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "13px",
                        "marginRight": "10px"
                    }),
                    dcc.RadioItems(
                        id="gcp2-display-mode",
                        options=[
                            {"label": " Rolling Z-normalized (comparable)", "value": "rolling_z"},
                            {"label": " Raw cumsum(nc)", "value": "raw"},
                        ],
                        value="rolling_z",
                        inline=True,
                        style={
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "fontFamily": "'Courier New', monospace",
                            "fontSize": "13px"
                        },
                        inputStyle={"marginRight": "5px"},
                        labelStyle={"marginRight": "15px"}
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"display": "none", "flexWrap": "wrap", "gap": "10px"}),

            # Date range warning for GCP2
            html.Div(id="gcp2-date-warning", style={
                "color": CYBERPUNK_COLORS['neon_yellow'],
                "fontFamily": "'Courier New', monospace",
                "fontSize": "13px",
                "marginTop": "10px"
            }),
        ], style={
            "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
            "padding": "20px",
            "borderRadius": "15px",
            "border": f"2px solid {CYBERPUNK_COLORS['neon_yellow']}",
            "boxShadow": f"0 0 30px {CYBERPUNK_COLORS['neon_yellow']}40",
            "marginBottom": "30px"
        }),

        # Controls section with cyberpunk styling
        html.Div([
            # Date controls
            html.Div([
                html.Label("WINDOW START DATE (UTC)", 
                          style={
                              "color": CYBERPUNK_COLORS['neon_green'],
                              "fontSize": "14px",
                              "fontWeight": "bold",
                              "fontFamily": "'Courier New', monospace",
                              "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_green']}",
                              "marginBottom": "10px"
                          }),
                html.Div([
                    dcc.Input(
                        id="date-input",
                        type="text",
                        value=DEFAULT_DATE.strftime("%Y-%m-%d"),
                        placeholder="YYYY-MM-DD",
                        style={
                            "marginRight": "10px",
                            "width": "120px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_green']}",
                            "borderRadius": "5px",
                            "fontFamily": "'Courier New', monospace"
                        },
                        debounce=True
                    ),
                    html.Span(" OR USE SLIDER BELOW", 
                             style={
                                 "fontSize": "12px", 
                                 "color": CYBERPUNK_COLORS['text_secondary'],
                                 "fontFamily": "'Courier New', monospace"
                             }),
                    html.Div("VALIDATION: Press Enter or click outside to apply", 
                            style={
                                "fontSize": "10px", 
                                "color": CYBERPUNK_COLORS['neon_green'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace",
                                "fontStyle": "italic"
                            })
                ], style={"marginBottom": "10px"}),
                
                dcc.Slider(
                    id="start-date", 
                    min=0, 
                    max=(DATE_MAX - DATE_MIN).days, 
                    step=1,
                    value=(DEFAULT_DATE - DATE_MIN).days, 
                    marks={
                        0: "1998",
                        (_dt(1999, 1, 1).date() - DATE_MIN).days: "1999",
                        (_dt(2000, 1, 1).date() - DATE_MIN).days: "2000",
                        (_dt(2001, 1, 1).date() - DATE_MIN).days: "2001",
                        (_dt(2002, 1, 1).date() - DATE_MIN).days: "2002",
                        (_dt(2003, 1, 1).date() - DATE_MIN).days: "2003",
                        (_dt(2004, 1, 1).date() - DATE_MIN).days: "2004",
                        (_dt(2005, 1, 1).date() - DATE_MIN).days: "2005",
                        (_dt(2006, 1, 1).date() - DATE_MIN).days: "2006",
                        (_dt(2007, 1, 1).date() - DATE_MIN).days: "2007",
                        (_dt(2008, 1, 1).date() - DATE_MIN).days: "2008",
                        (_dt(2009, 1, 1).date() - DATE_MIN).days: "2009",
                        (_dt(2010, 1, 1).date() - DATE_MIN).days: "2010",
                        (_dt(2011, 1, 1).date() - DATE_MIN).days: "2011",
                        (_dt(2012, 1, 1).date() - DATE_MIN).days: "2012",
                        (_dt(2013, 1, 1).date() - DATE_MIN).days: "2013",
                        (_dt(2014, 1, 1).date() - DATE_MIN).days: "2014",
                        (_dt(2015, 1, 1).date() - DATE_MIN).days: "2015",
                        (_dt(2016, 1, 1).date() - DATE_MIN).days: "2016",
                        (_dt(2017, 1, 1).date() - DATE_MIN).days: "2017",
                        (_dt(2018, 1, 1).date() - DATE_MIN).days: "2018",
                        (_dt(2019, 1, 1).date() - DATE_MIN).days: "2019",
                        (_dt(2020, 1, 1).date() - DATE_MIN).days: "2020",
                        (_dt(2021, 1, 1).date() - DATE_MIN).days: "2021",
                        (_dt(2022, 1, 1).date() - DATE_MIN).days: "2022",
                        (_dt(2023, 1, 1).date() - DATE_MIN).days: "2023",
                        (_dt(2024, 1, 1).date() - DATE_MIN).days: "2024",
                        (_dt(2025, 1, 1).date() - DATE_MIN).days: "2025",
                        (_dt(2026, 1, 1).date() - DATE_MIN).days: "2026",
                        (_dt(2027, 1, 1).date() - DATE_MIN).days: "2027"
                    },
                    updatemode="mouseup",
                    tooltip={"placement": "bottom"},
                    className="date-slider"
                ),
                html.Div(id="start-date-readout", style={"marginBottom": "0.5rem", "color": CYBERPUNK_COLORS['neon_green']}),
            ], style={"marginBottom": "20px"}),
            
            # Time controls
            html.Div([
                html.Label("WINDOW START TIME (UTC)", 
                          style={
                              "color": CYBERPUNK_COLORS['neon_yellow'],
                              "fontSize": "14px",
                              "fontWeight": "bold",
                              "fontFamily": "'Courier New', monospace",
                              "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_yellow']}",
                              "marginBottom": "10px"
                          }),
                html.Div([
                    dcc.Input(
                        id="time-input",
                        type="text",
                        value=DEFAULT_TIME.strftime("%H:%M"),
                        placeholder="HH:MM or H",
                        style={
                            "marginRight": "10px", 
                            "width": "100px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_yellow']}",
                            "borderRadius": "5px",
                            "fontFamily": "'Courier New', monospace"
                        },
                        debounce=True
                    ),
                    html.Span(" OR USE SLIDER BELOW", 
                             style={
                                 "fontSize": "12px", 
                                 "color": CYBERPUNK_COLORS['text_secondary'],
                                 "fontFamily": "'Courier New', monospace"
                             }),
                    html.Div("FORMAT: HH:MM, HH:MM:SS, OR JUST H (E.G., 14:30, 14:30:45, 14)", 
                            style={
                                "fontSize": "11px", 
                                "color": CYBERPUNK_COLORS['text_secondary'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace"
                            }),
                    html.Div("VALIDATION: Press Enter or click outside to apply", 
                            style={
                                "fontSize": "10px", 
                                "color": CYBERPUNK_COLORS['neon_yellow'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace",
                                "fontStyle": "italic"
                            })
                ], style={"marginBottom": "10px"}),
                
                dcc.Slider(
                    id="start-time", 
                    min=0, 
                    max=86399,  # 23:59:59 in seconds
                    step=60,  # 1 minute steps
                    value=DEFAULT_TIME.hour * 3600 + DEFAULT_TIME.minute * 60 + DEFAULT_TIME.second,
                    marks={
                        0: "00:00",
                        1800: "'",
                        3600: "01:00",
                        5400: "'",
                        7200: "02:00",
                        9000: "'",
                        10800: "03:00",
                        12600: "'",
                        14400: "04:00",
                        16200: "'",
                        18000: "05:00",
                        19800: "'",
                        21600: "06:00",
                        23400: "'",
                        25200: "07:00",
                        27000: "'",
                        28800: "08:00",
                        30600: "'",
                        32400: "09:00",
                        34200: "'",
                        36000: "10:00",
                        37800: "'",
                        39600: "11:00",
                        41400: "'",
                        43200: "12:00",
                        45000: "'",
                        46800: "13:00",
                        48600: "'",
                        50400: "14:00",
                        52200: "'",
                        54000: "15:00",
                        55800: "'",
                        57600: "16:00",
                        59400: "'",
                        61200: "17:00",
                        63000: "'",
                        64800: "18:00",
                        66600: "'",
                        68400: "19:00",
                        70200: "'",
                        72000: "20:00",
                        73800: "'",
                        75600: "21:00",
                        77400: "'",
                        79200: "22:00",
                        81000: "'",
                        82800: "23:00",
                        84600: "'",
                        86399: "23:59"
                    },
                    updatemode="mouseup",
                    tooltip={"placement": "bottom"},
                    className="time-slider"
                ),
                html.Div(id="start-time-readout", style={"marginBottom": "1rem", "color": CYBERPUNK_COLORS['neon_yellow']}),
            ], style={"marginBottom": "20px"}),

            # Window length controls
            html.Div([
                html.Label("WINDOW LENGTH (SECONDS)", 
                          style={
                              "color": CYBERPUNK_COLORS['neon_purple'],
                              "fontSize": "14px",
                              "fontWeight": "bold",
                              "fontFamily": "'Courier New', monospace",
                              "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_purple']}",
                              "marginBottom": "10px"
                          }),
                html.Div([
                    dcc.Input(
                        id="window-length-input",
                        type="text",
                        value=DEFAULT_BINS,
                        placeholder="seconds (60-7776000)",
                        style={
                            "marginRight": "10px", 
                            "width": "120px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_purple']}",
                            "borderRadius": "5px",
                            "fontFamily": "'Courier New', monospace"
                        },
                        debounce=True
                    ),
                    html.Span(" SECONDS OR USE SLIDER BELOW", 
                             style={
                                 "fontSize": "12px", 
                                 "color": CYBERPUNK_COLORS['text_secondary'],
                                 "fontFamily": "'Courier New', monospace"
                             }),
                    html.Div("ENTER SECONDS (60 TO 7,776,000)", 
                            style={
                                "fontSize": "11px", 
                                "color": CYBERPUNK_COLORS['text_secondary'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace"
                            }),
                    html.Div("VALIDATION: Press Enter or click outside to apply", 
                            style={
                                "fontSize": "10px", 
                                "color": CYBERPUNK_COLORS['neon_purple'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace",
                                "fontStyle": "italic"
                            })
                ], style={"marginBottom": "10px"}),
                
                dcc.Slider(
                    id="len", 
                    min=LEN_MIN_S, 
                    max=LEN_MAX_S, 
                    step=60, 
                    value=DEFAULT_WINDOW_LEN,
                    marks={
                        60: "1m",
                        43200: "12h",
                        86400: "1d",
                        172800: "2d",
                        259200: "3d",
                        604800: "1w",
                        1209600: "2w",
                        1814400: "3w",
                        2592000: "30d",
                        7776000: "90d",
                        15552000: "180d",
                        25920000: "300d",
                        31536000: "1y",
                        63072000: "2y",
                        94608000: "3y",
                        157680000: "5y",
                        315360000: "10y",
                        473040000: "15y",
                        630720000: "20y",
                        788400000: "25y",
                        864000000: "27y"
                    },
                    updatemode="mouseup", 
                    tooltip={"placement": "bottom"},
                    className="length-slider"
                ),
                html.Div(id="len-readout", style={"marginBottom": "1rem", "color": CYBERPUNK_COLORS['neon_purple']}),
            ], style={"marginBottom": "20px"}),

            # Bin count controls
            html.Div([
                html.Label("BIN COUNT", 
                          style={
                              "color": CYBERPUNK_COLORS['neon_pink'],
                              "fontSize": "14px",
                              "fontWeight": "bold",
                              "fontFamily": "'Courier New', monospace",
                              "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_pink']}",
                              "marginBottom": "10px"
                          }),
                html.Div([
                    dcc.Input(
                        id="bin-count-input",
                        type="text",
                        value=DEFAULT_BINS,
                        placeholder="bins (1-30000)",
                        style={
                            "marginRight": "10px", 
                            "width": "80px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_pink']}",
                            "borderRadius": "5px",
                            "fontFamily": "'Courier New', monospace"
                        },
                        debounce=True
                    ),
                    html.Span(" BINS OR USE SLIDER BELOW", 
                             style={
                                 "fontSize": "12px", 
                                 "color": CYBERPUNK_COLORS['text_secondary'],
                                 "fontFamily": "'Courier New', monospace"
                             }),
                    html.Div("ENTER NUMBER OF BINS (1 TO 30,000)", 
                            style={
                                "fontSize": "11px", 
                                "color": CYBERPUNK_COLORS['text_secondary'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace"
                            }),
                    html.Div("VALIDATION: Press Enter or click outside to apply", 
                            style={
                                "fontSize": "10px", 
                                "color": CYBERPUNK_COLORS['neon_pink'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace",
                                "fontStyle": "italic"
                            })
                ], style={"marginBottom": "10px"}),
                
                dcc.Slider(
                    id="bins", 
                    min=BINS_MIN, 
                    max=BINS_MAX, 
                    step=1, 
                    value=DEFAULT_BINS,
                    marks={
                        1: "1",
                        100: "100",
                        500: "500",
                        1000: "1K",
                        5000: "5K",
                        10000: "10K",
                        15000: "15K",
                        20000: "20K",
                        25000: "25K",
                        30000: "30K"
                    },
                    updatemode="mouseup", 
                    tooltip={"placement": "bottom"},
                    className="bins-slider"
                ),
                html.Div(id="bins-readout", style={"color": CYBERPUNK_COLORS['neon_pink']}),
            ], style={"marginBottom": "20px"}),
            
            # Status indicator
            html.Div(id="status-indicator", 
                    style={
                        "marginTop": "10px", 
                        "fontSize": "12px", 
                        "color": CYBERPUNK_COLORS['neon_cyan'],
                        "fontFamily": "'Courier New', monospace",
                        "textShadow": f"0 0 5px {CYBERPUNK_COLORS['neon_cyan']}"
                    })
    ], style={
        "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
        "padding": "20px",
        "borderRadius": "15px",
        "border": f"2px solid {CYBERPUNK_COLORS['neon_cyan']}",
        "boxShadow": f"0 0 30px {CYBERPUNK_COLORS['neon_cyan']}40"
    })
], style={
    "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
    "minHeight": "100vh",
    "padding": "20px",
    "fontFamily": "'Courier New', monospace"
})
], style={
    "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
    "minHeight": "100vh"
})

# ───────────────────────────── callbacks ────────────────────────────────────

def create_url_callback(app_instance):
    """Callback to handle URL parameters and set advanced mode."""
    
    @app_instance.callback(
        Output("len", "max"),
        Output("len", "marks"),
        Input("url", "search")
    )
    def update_slider_limits(search):
        """Update slider limits based on URL parameters."""
        import urllib.parse
        
        # Parse URL search parameters
        if search:
            params = urllib.parse.parse_qs(search.lstrip('?'))
            advanced_mode = params.get('adv', [''])[0].lower() == 'true'
        else:
            advanced_mode = False
        
        if advanced_mode:
            # Advanced mode: full 27-year range
            max_seconds = 10000 * 24 * 3600  # ~27 years
            marks = {
                60: "1m",
                43200: "12h",
                86400: "1d",
                172800: "2d",
                259200: "3d",
                604800: "1w",
                1209600: "2w",
                1814400: "3w",
                2592000: "30d",
                7776000: "90d",
                15552000: "180d",
                25920000: "300d",
                31536000: "1y",
                63072000: "2y",
                94608000: "3y",
                157680000: "5y",
                315360000: "10y",
                473040000: "15y",
                630720000: "20y",
                788400000: "25y",
                864000000: "27y"
            }
        else:
            # Standard mode: 3-month range
            max_seconds = 90 * 24 * 3600  # 3 months
            marks = {
                60: "1m",
                43200: "12h",
                86400: "1d",
                172800: "2d",
                259200: "3d",
                604800: "1w",
                1209600: "2w",
                1814400: "3w",
                2592000: "30d",
                7776000: "90d"
            }
        
        return max_seconds, marks

def create_egg_callback(app_instance):
    import time
    from dash import ctx
    
    @app_instance.callback(
        Output("chi2-graph", "figure"),
        Output("start-date-readout", "children"),
        Output("start-time-readout", "children"),
        Output("len-readout", "children"),
        Output("bins-readout", "children"),
        Output("status-indicator", "children"),
        Output("date-input", "value"),
        Output("time-input", "value"),
        Output("window-length-input", "value"),
        Output("bin-count-input", "value"),
        Output("start-date", "value"),
        Output("start-time", "value"),
        Output("len", "value"),
        Output("bins", "value"),
        Output("gcp2-options-container", "style"),
        Output("gcp2-date-warning", "children"),
        Input("start-date", "value"), Input("start-time", "value"), Input("len", "value"), Input("bins", "value"),
        Input("date-input", "value"), Input("time-input", "value"), Input("window-length-input", "value"), Input("bin-count-input", "value"),
        Input("filter-broken-eggs", "value"),
        Input("pseudo-entropy", "value"),
        Input("show-parabolic-curve", "value"),
        Input("clear-cache-btn", "n_clicks"),
        Input("data-source-select", "value"),
        Input("gcp2-network-select", "value"),
        Input("gcp2-display-mode", "value")
    )
    def update_graph(start_date_days, start_time_seconds, window_len, bins,
                    date_input, time_input, window_length_input, bin_count_input, filter_broken_eggs, pseudo_entropy, show_parabolic_curve, clear_cache_clicks,
                    data_sources, gcp2_network, gcp2_display_mode):

        start_time = time.time()
        
        # Determine which input triggered the callback and use that value
        triggered_id = ctx.triggered_id if ctx.triggered else None
        
        # Handle cache clearing
        if triggered_id == "clear-cache-btn" and clear_cache_clicks and clear_cache_clicks > 0:
            # Clear the BigQuery cache
            CACHE.clear()
            print(f"BigQuery cache cleared at {_dt.now()}")
        
        # Initialize values with defaults
        start_date_days = int(start_date_days or 0)
        start_time_seconds = int(start_time_seconds or 0)
        window_len = max(int(window_len or DEFAULT_WINDOW_LEN), LEN_MIN_S)  # 1 hour default
        bins = max(int(bins or DEFAULT_BINS), BINS_MIN)
        
        # Handle date synchronization
        if triggered_id == "date-input" and date_input:
            # Date input was changed - try to parse and validate
            try:
                date_input = str(date_input).strip()
                if date_input:
                    # Try to parse as YYYY-MM-DD format
                    selected_date = _dt.strptime(date_input, "%Y-%m-%d").date()
                    # Clamp to valid range
                    if selected_date < DATE_MIN:
                        selected_date = DATE_MIN
                    elif selected_date > DATE_MAX:
                        selected_date = DATE_MAX
                    start_date_days = (selected_date - DATE_MIN).days
                else:
                    # Empty input - use current slider value
                    selected_date = DATE_MIN + _td(days=start_date_days)
            except (ValueError, TypeError) as e:
                # Invalid input - use current slider value
                selected_date = DATE_MIN + _td(days=start_date_days)
        elif triggered_id in ["start-time", "time-input", "len", "bins", "window-length-input", "bin-count-input", "filter-broken-eggs", "clear-cache-btn", "data-source-select", "gcp2-network-select", "gcp2-display-mode"]:
            # Use slider value when other inputs are triggered (but not when start-date slider is triggered)
            selected_date = DATE_MIN + _td(days=start_date_days)
        elif triggered_id == "start-date":
            # Date slider was changed - update date input
            selected_date = DATE_MIN + _td(days=start_date_days)
        else:
            # Use slider value or default
            selected_date = DATE_MIN + _td(days=start_date_days)
        
        # Handle time synchronization
        if triggered_id == "time-input" and time_input:
            # Time input was changed - try to parse and validate
            time_input = time_input.strip()
            try:
                if ":" in time_input:
                    parts = time_input.split(":")
                    if len(parts) == 2:
                        hours, minutes = map(int, parts)
                        # Allow reasonable time ranges, clamp to valid values
                        hours = max(0, min(23, hours))
                        minutes = max(0, min(59, minutes))
                        selected_time = _dt.strptime(f"{hours:02d}:{minutes:02d}", "%H:%M").time()
                        start_time_seconds = hours * 3600 + minutes * 60
                    elif len(parts) == 3:
                        hours, minutes, seconds = map(int, parts)
                        hours = max(0, min(23, hours))
                        minutes = max(0, min(59, minutes))
                        seconds = max(0, min(59, seconds))
                        selected_time = _dt.strptime(f"{hours:02d}:{minutes:02d}:{seconds:02d}", "%H:%M:%S").time()
                        start_time_seconds = hours * 3600 + minutes * 60 + seconds
                    else:
                        # Invalid format - use current slider value
                        selected_time = _dt.strptime(f"{start_time_seconds//3600:02d}:{(start_time_seconds%3600)//60:02d}", "%H:%M").time()
                else:
                    # Try to parse as just hours
                    hours = int(time_input)
                    hours = max(0, min(23, hours))
                    selected_time = _dt.strptime(f"{hours:02d}:00", "%H:%M").time()
                    start_time_seconds = hours * 3600
            except (ValueError, TypeError) as e:
                # Invalid input - use current slider value
                selected_time = _dt.strptime(f"{start_time_seconds//3600:02d}:{(start_time_seconds%3600)//60:02d}", "%H:%M").time()
        elif triggered_id in ["start-time", "date-input", "start-date", "len", "bins", "window-length-input", "bin-count-input", "filter-broken-eggs", "clear-cache-btn", "data-source-select", "gcp2-network-select", "gcp2-display-mode"]:
            # Use slider value when other inputs are triggered
            selected_time = _dt.strptime(f"{start_time_seconds//3600:02d}:{(start_time_seconds%3600)//60:02d}", "%H:%M").time()
        else:
            # Use slider value
            selected_time = _dt.strptime(f"{start_time_seconds//3600:02d}:{(start_time_seconds%3600)//60:02d}", "%H:%M").time()
        
        # Handle window length synchronization
        if triggered_id == "window-length-input" and window_length_input is not None:
            # Text input was changed - try to parse and validate
            try:
                window_length_input = str(window_length_input).strip()
                if window_length_input:
                    # Try to parse as integer
                    parsed_len = int(window_length_input)
                    # Clamp to valid range
                    window_len = max(LEN_MIN_S, min(LEN_MAX_S, parsed_len))
                else:
                    # Empty input - use current slider value
                    window_len = max(int(window_len), LEN_MIN_S)
            except (ValueError, TypeError) as e:
                # Invalid input - use current slider value
                window_len = max(int(window_len), LEN_MIN_S)
        elif triggered_id in ["start-time", "date-input", "start-date", "time-input", "bins", "bin-count-input", "filter-broken-eggs", "clear-cache-btn", "data-source-select", "gcp2-network-select", "gcp2-display-mode"]:
            # Use slider value when other inputs are triggered
            window_len = max(int(window_len), LEN_MIN_S)
        else:
            # Use slider value
            window_len = max(int(window_len), LEN_MIN_S)
        
        # Handle bin count synchronization
        if triggered_id == "bin-count-input" and bin_count_input is not None:
            # Text input was changed - try to parse and validate
            try:
                bin_count_input = str(bin_count_input).strip()
                if bin_count_input:
                    # Try to parse as integer
                    parsed_bins = int(bin_count_input)
                    # Clamp to valid range
                    bins = max(BINS_MIN, min(BINS_MAX, parsed_bins))
                else:
                    # Empty input - use current slider value
                    bins = max(int(bins), BINS_MIN)
            except (ValueError, TypeError) as e:
                # Invalid input - use current slider value
                bins = max(int(bins), BINS_MIN)
        elif triggered_id in ["start-time", "date-input", "start-date", "time-input", "len", "window-length-input", "filter-broken-eggs", "clear-cache-btn", "data-source-select", "gcp2-network-select", "gcp2-display-mode"]:
            # Use slider value when other inputs are triggered
            bins = max(int(bins), BINS_MIN)
        else:
            # Use slider value
            bins = max(int(bins), BINS_MIN)
        
        # Combine date and time into datetime
        start_ts = _dt.combine(selected_date, selected_time, tzinfo=_tz.utc).timestamp()

        # Handle broken egg filter
        filter_broken_eggs_enabled = filter_broken_eggs and "enabled" in filter_broken_eggs

        # Handle pseudo entropy option
        use_pseudo_entropy = pseudo_entropy and "enabled" in pseudo_entropy

        # Handle parabolic curve option
        show_parabolic_curve_enabled = show_parabolic_curve and "enabled" in show_parabolic_curve

        # Handle data source selection
        data_sources = data_sources or ["gcp1"]
        show_gcp1 = "gcp1" in data_sources
        show_gcp2 = "gcp2" in data_sources

        # GCP2 options visibility
        gcp2_options_style = {"display": "flex", "flexWrap": "wrap", "gap": "10px"} if show_gcp2 else {"display": "none"}

        # Validate GCP2 date range
        gcp2_has_data, gcp2_warning = validate_gcp2_date_range(start_ts, window_len) if show_gcp2 else (False, "")

        # Fetch GCP1 data if enabled
        df = pd.DataFrame()
        is_cached = False
        if show_gcp1:
            real_cache_key = (start_ts, window_len, bins, filter_broken_eggs_enabled, False)
            is_cached = CACHE.get(real_cache_key) is not None
            df = query_bq(start_ts, window_len, bins, filter_broken_eggs_enabled, False)

        # Fetch GCP2 data if enabled
        df_gcp2 = pd.DataFrame()
        if show_gcp2 and gcp2_has_data:
            df_gcp2 = query_gcp2(start_ts, window_len, bins, gcp2_network or "global_network")
        
        # Calculate timing (always do this)
        elapsed_time = time.time() - start_time

        # Check if we have any data from either source
        has_gcp1_data = show_gcp1 and not df.empty
        has_gcp2_data = show_gcp2 and not df_gcp2.empty

        if not has_gcp1_data and not has_gcp2_data:
            fig = go.Figure()
            
            # Format window length in human readable units for empty case
            if window_len < 60:
                window_len_str = f"{window_len}s"
            elif window_len < 3600:
                window_len_str = f"{window_len//60}m {window_len%60}s"
            elif window_len < 86400:
                hours = window_len // 3600
                minutes = (window_len % 3600) // 60
                window_len_str = f"{hours}h {minutes}m"
            elif window_len < 604800:
                days = window_len // 86400
                hours = (window_len % 86400) // 3600
                window_len_str = f"{days}d {hours}h"
            else:
                days = window_len // 86400
                window_len_str = f"{days}d"
            
            # Create comprehensive title for empty case
            pseudo_status = " | PSEUDO ENTROPY: ON" if use_pseudo_entropy else ""
            parabolic_status = " | Parabolic curve: ON" if show_parabolic_curve_enabled else " | Parabolic curve: OFF"
            title_text = f"""
            <b>GCP RNG Statistical Analysis</b><br>
            <span style='font-size: 14px; color: {CYBERPUNK_COLORS['neon_pink']};'>
            Date: {selected_date.strftime('%Y-%m-%d')} | Time: {selected_time.strftime('%H:%M:%S')} UTC<br>
            Window: {window_len_str} ({window_len:,}s) | Bins: {bins}<br>
            0-Filter: {'ON' if filter_broken_eggs_enabled else 'OFF'}{pseudo_status}{parabolic_status} | No data found • {elapsed_time:.2f}s
            </span>
            """
            
            fig.update_layout(
                title=dict(
                    text=title_text,
                    x=0.5,
                    xanchor="center",
                    y=0.98,
                    yanchor="top",
                    font=dict(
                        color=CYBERPUNK_COLORS['text_primary'],
                        family="'Courier New', monospace",
                        size=16
                    )
                ),
                xaxis_title="Time from window start",
                yaxis_title="Cumulative χ²",
                annotations=[dict(
                    text="No data in selected window", 
                    showarrow=False,
                    font=dict(color=CYBERPUNK_COLORS['neon_cyan'])
                )],
                margin=dict(l=40, r=40, t=120, b=40),
                plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                font=dict(
                    color=CYBERPUNK_COLORS['text_primary'],
                    family="'Courier New', monospace"
                )
            )
            
            # Get data range info for status string
            first_date_str, last_date_str, total_count = get_data_range_info()
            sources_str = f"GCP1: {'ON' if show_gcp1 else 'OFF'} | GCP2: {'ON' if show_gcp2 else 'OFF'}"
            status_str = f"⚠ No data found in {elapsed_time:.2f}s | {sources_str} | Total egg basket data from {first_date_str} to {last_date_str}, total rows {total_count:,}"

            return (fig, "No data", "", "", "", status_str,
                    selected_date.strftime("%Y-%m-%d"), selected_time.strftime("%H:%M"), window_len, bins,
                    start_date_days, start_time_seconds, window_len, bins,
                    gcp2_options_style, gcp2_warning)

        # Calculate x-axis values and determine appropriate units
        seconds_per_bin = window_len / bins

        # Choose appropriate time unit based on total window length for better readability
        if window_len < 3600:  # Less than 1 hour
            time_unit = "minutes"
            conversion_factor = 60
        elif window_len < 86400:  # Less than 1 day
            time_unit = "hours"
            conversion_factor = 3600
        elif window_len < 604800:  # Less than 1 week
            time_unit = "days"
            conversion_factor = 86400
        else:  # 1 week or more
            time_unit = "days"
            conversion_factor = 86400

        # Calculate x-axis for GCP1 data
        x = df["bin_idx"] * seconds_per_bin / conversion_factor if has_gcp1_data else pd.Series([], dtype=float)

        # Calculate x-axis for GCP2 data
        x_gcp2 = df_gcp2["bin_idx"] * seconds_per_bin / conversion_factor if has_gcp2_data else pd.Series([], dtype=float)
        
        # Create single-axis plot for cumulative deviation of χ²
        fig = go.Figure()

        # Calculate pointwise 95% (p=0.05) envelope under null using cumulative seconds
        # Under the null: Var(χ²(1) − 1) = 2 ⇒ Var[Σ X_t over T seconds] = 2T
        # Envelope: ± z_{0.975} √(2T), evaluated at cumulative seconds T_i
        # Use GCP1 data for envelope if available, otherwise GCP2
        if has_gcp1_data:
            cumulative_seconds = df["seconds_in_bin"].cumsum()
        elif has_gcp2_data:
            cumulative_seconds = df_gcp2["seconds_in_bin"].cumsum()
        else:
            cumulative_seconds = pd.Series([], dtype=float)

        envelope = compute_pointwise_p05_envelope(cumulative_seconds)
        # Plot x for envelope in the same display units as data
        x_curve = cumulative_seconds / conversion_factor if len(cumulative_seconds) else cumulative_seconds
        upper_curve = envelope
        lower_curve = -envelope

        # Add parabolic probability curves only if checkbox is enabled
        if show_parabolic_curve_enabled and len(x_curve) > 0:
            # Add upper significance curve (p=0.05)
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=upper_curve,
                mode="lines",
                name="p=0.05 significance (upper)",
                line=dict(
                    color="red",
                    width=2
                ),
                showlegend=True
            ))

            # Add lower significance curve (p=0.05)
            fig.add_trace(go.Scatter(
                x=x_curve,
                y=lower_curve,
                mode="lines",
                name="p=0.05 significance (lower)",
                line=dict(
                    color="red",
                    width=2
                ),
                showlegend=True
            ))

        # Add GCP1 data trace if enabled and has data
        if has_gcp1_data:
            fig.add_trace(go.Scatter(
                x=x,
                y=df["cum_stouffer_z"],
                mode="lines",
                name="GCP1 χ² (Stouffer Z)",
                line=dict(
                    color=CYBERPUNK_COLORS['neon_purple'],
                    width=3,
                    shape='spline'
                )
            ))

        # Add GCP2 data trace if enabled and has data
        if has_gcp2_data:
            gcp2_mode = gcp2_display_mode or "rolling_z"
            if gcp2_mode == "rolling_z":
                y_gcp2 = df_gcp2["cum_rolling_z"]
                gcp2_trace_name = f"GCP2 χ² Rolling-Z ({gcp2_network})"
            else:
                y_gcp2 = df_gcp2["cum_nc"]
                gcp2_trace_name = f"GCP2 Raw Cumsum ({gcp2_network})"

            fig.add_trace(go.Scatter(
                x=x_gcp2,
                y=y_gcp2,
                mode="lines",
                name=gcp2_trace_name,
                line=dict(
                    color=CYBERPUNK_COLORS['neon_yellow'],
                    width=2,
                    shape='spline'
                )
            ))
        
        # If pseudo entropy is enabled (only applies to GCP1), add a trace with random data
        if use_pseudo_entropy and show_gcp1:
            # Query for pseudo entropy data (same parameters but with random values)
            # Use a different cache key to avoid conflicts with real data
            pseudo_cache_key = (start_ts, window_len, bins, filter_broken_eggs_enabled, True)
            df_pseudo = CACHE.get(pseudo_cache_key)
            if df_pseudo is None:
                df_pseudo = query_bq(start_ts, window_len, bins, filter_broken_eggs_enabled, True)
                CACHE.set(pseudo_cache_key, df_pseudo, expire=3600)

            if not df_pseudo.empty:
                # Calculate x-axis values for pseudo data (same as real data)
                x_pseudo = df_pseudo["bin_idx"] * seconds_per_bin / conversion_factor

                fig.add_trace(go.Scatter(
                    x=x_pseudo,
                    y=df_pseudo["cum_stouffer_z"],
                    mode="lines",
                    name="GCP1 Pseudo Entropy (random)",
                    line=dict(
                        color=CYBERPUNK_COLORS['neon_green'],  # Green to distinguish from GCP2 yellow
                        width=2,
                        shape='spline',
                        dash='dash'
                    )
                ))
        
        # Format window length in human readable units
        if window_len < 60:
            window_len_str = f"{window_len}s"
        elif window_len < 3600:
            window_len_str = f"{window_len//60}m {window_len%60}s"
        elif window_len < 86400:
            hours = window_len // 3600
            minutes = (window_len % 3600) // 60
            window_len_str = f"{hours}h {minutes}m"
        elif window_len < 604800:
            days = window_len // 86400
            hours = (window_len % 86400) // 3600
            window_len_str = f"{days}d {hours}h"
        else:
            days = window_len // 86400
            window_len_str = f"{days}d"
        
        # Format bin duration in human readable units
        bin_duration = window_len / bins
        if bin_duration < 60:
            bin_duration_str = f"{bin_duration:.1f}s"
        elif bin_duration < 3600:
            bin_duration_str = f"{bin_duration/60:.1f}m"
        elif bin_duration < 86400:
            bin_duration_str = f"{bin_duration/3600:.1f}h"
        else:
            bin_duration_str = f"{bin_duration/86400:.1f}d"
        
        # Get active eggs/devices count
        active_eggs = df['avg_active_eggs'].mean() if has_gcp1_data else 0
        active_devices = df_gcp2['avg_active_devices'].mean() if has_gcp2_data else 0

        # Build data source info for title
        sources_info = []
        if has_gcp1_data:
            sources_info.append(f"GCP1: {active_eggs:.0f} eggs")
        if has_gcp2_data:
            sources_info.append(f"GCP2 ({gcp2_network}): {active_devices:.0f} devices")
        sources_str = " | ".join(sources_info) if sources_info else "No data"

        # Create comprehensive title
        pseudo_status = " | PSEUDO ENTROPY: ON" if use_pseudo_entropy else ""
        parabolic_status = " | Parabolic: ON" if show_parabolic_curve_enabled else ""
        gcp2_mode_str = f" | Mode: {gcp2_display_mode}" if has_gcp2_data else ""
        title_text = f"""
        <b>GCP RNG Statistical Analysis</b><br>
        <span style='font-size: 14px; color: {CYBERPUNK_COLORS['neon_cyan']};'>
        Date: {selected_date.strftime('%Y-%m-%d')} | Time: {selected_time.strftime('%H:%M:%S')} UTC<br>
        Window: {window_len_str} ({window_len:,}s) | Bins: {bins} (≈{bin_duration_str}/bin)<br>
        {sources_str}{gcp2_mode_str}{pseudo_status}{parabolic_status}
        </span>
        """
        
        # Cyberpunk-styled layout
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor="center",
                y=0.98,
                yanchor="top",
                font=dict(
                    color=CYBERPUNK_COLORS['text_primary'],
                    family="'Courier New', monospace",
                    size=16
                )
            ),
            xaxis_title=f"Time from window start ({time_unit})",
            yaxis_title="Cumulative deviation of χ² based on Stouffer Z",
            margin=dict(l=40, r=40, t=120, b=40),
            legend=dict(
                x=0.02, 
                y=0.98,
                bgcolor=CYBERPUNK_COLORS['bg_dark'],
                bordercolor=CYBERPUNK_COLORS['neon_purple'],
                borderwidth=1,
                font=dict(color=CYBERPUNK_COLORS['text_primary']),
                yanchor="top"
            ),
            plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
            paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
            font=dict(
                color=CYBERPUNK_COLORS['text_primary'],
                family="'Courier New', monospace"
            ),
            xaxis=dict(
                gridcolor=CYBERPUNK_COLORS['bg_medium'],
                zerolinecolor=CYBERPUNK_COLORS['neon_cyan'],
                title_font=dict(color=CYBERPUNK_COLORS['neon_cyan']),
                tickfont=dict(color=CYBERPUNK_COLORS['text_secondary'])
            ),
            yaxis=dict(
                gridcolor=CYBERPUNK_COLORS['bg_medium'],
                zerolinecolor=CYBERPUNK_COLORS['neon_cyan'],
                title_font=dict(color=CYBERPUNK_COLORS['neon_cyan']),
                tickfont=dict(color=CYBERPUNK_COLORS['text_secondary'])
            )
        )
        
        start_date_str = f"Date: {selected_date.strftime('%Y-%m-%d')}"
        start_time_str = f"Time: {selected_time.strftime('%H:%M:%S')}"
        len_str   = f"Length: {_td(seconds=window_len)} ({window_len:,} s)"
        # Format bin duration for display
        bin_duration = window_len / bins
        if bin_duration < 60:
            bin_duration_str = f"{bin_duration:.1f}s"
        elif bin_duration < 3600:
            bin_duration_str = f"{bin_duration/60:.1f}m"
        elif bin_duration < 86400:
            bin_duration_str = f"{bin_duration/3600:.1f}h"
        elif bin_duration < 604800:
            bin_duration_str = f"{bin_duration/86400:.1f}d"
        else:
            bin_duration_str = f"{bin_duration/86400:.1f}d"
        
        # Add filter status to bins string
        filter_status = " | 0-filter: ON" if filter_broken_eggs_enabled else " | 0-filter: OFF"
        active_info = f"Eggs: {active_eggs:.0f}" if has_gcp1_data else ""
        if has_gcp2_data:
            active_info += f" | Devices: {active_devices:.0f}" if active_info else f"Devices: {active_devices:.0f}"
        bins_str = f"Bins: {bins} (≈ {bin_duration_str}/bin) | {active_info}{filter_status}"

        # Status indicator with more details
        # Get data range info for status string
        first_date_str, last_date_str, total_count = get_data_range_info()

        sources_status = f"GCP1: {'ON' if show_gcp1 else 'OFF'} | GCP2: {'ON' if show_gcp2 else 'OFF'}"
        pseudo_info = " | PSEUDO: ON" if use_pseudo_entropy else ""
        parabolic_info = " | Parabolic: ON" if show_parabolic_curve_enabled else ""
        if triggered_id == "clear-cache-btn" and clear_cache_clicks and clear_cache_clicks > 0:
            status_str = f"Cache cleared! Data fetched in {elapsed_time:.2f}s | {sources_status}{pseudo_info}{parabolic_info}"
        elif is_cached:
            status_str = f"Cached data loaded in {elapsed_time:.2f}s | {sources_status}{pseudo_info}{parabolic_info}"
        else:
            status_str = f"Data fetched in {elapsed_time:.2f}s | {sources_status}{pseudo_info}{parabolic_info}"

        # Return all outputs including the input components for synchronization
        return (fig, start_date_str, start_time_str, len_str, bins_str, status_str,
                selected_date.strftime("%Y-%m-%d"), selected_time.strftime("%H:%M"), window_len, bins,
                start_date_days, start_time_seconds, window_len, bins,
                gcp2_options_style, gcp2_warning)

if __name__ == "__main__":
    # Register callbacks for standalone app
    create_url_callback(app)
    create_egg_callback(app)
    app.run(debug=True, host="0.0.0.0", port=8051)
