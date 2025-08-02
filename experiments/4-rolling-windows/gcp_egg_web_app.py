"""
Dash web application to explore Global Consciousness Project (GCP) EGG data
stored in BigQuery and reproduce Nelson-style statistical analysis.

* Full egg list ({len(EGG_COLS)} columns)
* Implements correct GCP methodology:
  1. Stouffer Z across eggs: Z_t(s) = ΣZ_i/√N (dynamic N based on active eggs)
  2. χ² based on Stouffer Z: (Z_t(s))² (distributed as χ²(1) under null hypothesis)
  3. Cumulative deviation: Σ((Z_t(s))² - 1) to detect departure from randomness
* Uses published expected values: μ=100, σ=7.0712
* Handles missing egg data by dynamically adjusting N in Stouffer Z calculation
* Guards against division-by-zero and empty windows
* Live slider read-outs
* Sliders fire the callback only on mouse-up
"""

import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime as _dt, timezone as _tz, timedelta as _td

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import diskcache as dc
from google.cloud import bigquery
from google.cloud.bigquery_storage_v1 import BigQueryReadClient

# ───────────────────────────── configuration ──────────────────────────────
GCP_PROJECT  = os.getenv("GCP_PROJECT", "gcpingcp")
GCP_DATASET  = os.getenv("GCP_DATASET", "eggs_us")
GCP_TABLE    = os.getenv("GCP_TABLE", "basket")          # raw second-level table
BASELINE_TBL = os.getenv("BASELINE_TABLE", "baseline_individual")

# Date range for sliders (3rd Aug 1998 to 25 Aug 2023)
DATE_MIN = _dt(1998, 8, 3, tzinfo=_tz.utc).date()
DATE_MAX = _dt(2023, 8, 25, tzinfo=_tz.utc).date()
# Default to start of 911 Nelson experiment (first plane hit WTC at 8:46 AM EDT = 12:46 PM UTC)
DEFAULT_DATE = _dt(2001, 9, 11, tzinfo=_tz.utc).date()
DEFAULT_TIME = _dt(2001, 9, 11, 12, 35, 0, tzinfo=_tz.utc).time()  # 8:35 AM EDT = 12:35 PM UTC
LEN_MIN_S, LEN_MAX_S = 60, 230 * 24 * 3600                # 1 min – 230 days
BINS_MIN, BINS_MAX   = 1, 30000

CACHE = dc.Cache("./bq_cache", size_limit=2 * 1024**3)
# Track which parameter combinations have already been printed this runtime
_PRINTED_KEYS = set()

bq_client  = bigquery.Client(project=GCP_PROJECT)
bqs_client = BigQueryReadClient()

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

# ───────────────────────────── SQL builder ────────────────────────────────
def build_sql() -> str:
    ascii_block = ",\n".join(f"    ASCII({c}) AS {c}" for c in EGG_COLS)  # still used in raw CTE, keeps query readable
    
    # Use published expected values: μ=100, σ=7.0712
    # Keep ASCII conversion as requested
    # Only calculate Z-scores for eggs with non-NULL data
    z_block_terms = []
    for c in EGG_COLS:
        z_block_terms.append(f"    IF({c} IS NOT NULL, SAFE_DIVIDE(({c} - 100), 7.0712), NULL) AS z_{c}")
    z_block = ",\n".join(z_block_terms)
    
    # Count non-null eggs for dynamic N calculation
    null_count_block = " + ".join(f"IF(z_{c} IS NULL, 0, 1)" for c in EGG_COLS)
    
    # Calculate Stouffer Z with dynamic N (only include non-null eggs)
    stouffer_z_terms = []
    for c in EGG_COLS:
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

def render_sql(start_ts: float, window_s: int, bins: int) -> str:
    """Return a BigQuery query with ALL parameters inlined as literals.

    The DECLARE block is removed and any references to its variables are
    substituted so the output can be copy-pasted directly into the BigQuery
    console with no additional parameter binding.  (Requested by user.)

    Scott Wilber emphasises that explicit literals avoid confusion when
    verifying results by hand.
    """
    import re

    sql = build_sql()

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

def query_bq(start_ts: float, window_s: int, bins: int) -> pd.DataFrame:
    key = (start_ts, window_s, bins)
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
        sql = build_sql()
        df = (
            bq_client.query(sql, job_config=cfg)
            .to_dataframe(bqstorage_client=bqs_client, create_bqstorage_client=True)
        )
        df["cum_stouffer_z"] = df["chi2_stouffer_sum"].cumsum()
        CACHE.set(key, df, expire=3600)

    # Always print SQL and result for debugging and verification
    print("\n===== BigQuery SQL =====\n" + render_sql(start_ts, window_s, bins) + "\n========================")
    print(df.head(10).to_string(index=False))
    print("========================\n")

    return df

# ───────────────────────────── Dash layout ─────────────────────────────────
app = dash.Dash(__name__)
app.title = "GCP EGG Statistical Analysis Explorer"

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

app.layout = html.Div([
    # Main container with cyberpunk styling
    html.Div([
        # Header with glowing effect
        html.Div([
            html.H1("GCP EGG STATISTICAL ANALYSIS EXPLORER", 
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
                    dcc.DatePickerSingle(
                        id="date-picker",
                        date=DEFAULT_DATE,
                        min_date_allowed=DATE_MIN,
                        max_date_allowed=DATE_MAX,
                        display_format="YYYY-MM-DD",
                        style={
                            "marginRight": "10px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_green']}",
                            "borderRadius": "5px"
                        }
                    ),
                    html.Span(" OR USE SLIDER BELOW", 
                             style={
                                 "fontSize": "12px", 
                                 "color": CYBERPUNK_COLORS['text_secondary'],
                                 "fontFamily": "'Courier New', monospace"
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
                        (_dt(2023, 1, 1).date() - DATE_MIN).days: "2023"
                    },
                    updatemode="mouseup",
                    tooltip={"placement": "bottom"}
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
                        }
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
                    tooltip={"placement": "bottom"}
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
                        value=15000,  # 4 hours 10 minutes
                        placeholder="seconds (60-2592000)",
                        style={
                            "marginRight": "10px", 
                            "width": "120px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_purple']}",
                            "borderRadius": "5px",
                            "fontFamily": "'Courier New', monospace"
                        }
                    ),
                    html.Span(" SECONDS OR USE SLIDER BELOW", 
                             style={
                                 "fontSize": "12px", 
                                 "color": CYBERPUNK_COLORS['text_secondary'],
                                 "fontFamily": "'Courier New', monospace"
                             }),
                    html.Div("ENTER SECONDS (60 TO 19,872,000)", 
                            style={
                                "fontSize": "11px", 
                                "color": CYBERPUNK_COLORS['text_secondary'],
                                "marginTop": "2px",
                                "fontFamily": "'Courier New', monospace"
                            })
                ], style={"marginBottom": "10px"}),
                
                dcc.Slider(
                    id="len", 
                    min=LEN_MIN_S, 
                    max=LEN_MAX_S, 
                    step=60, 
                    value=15000,  # 4 hours 10 minutes
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
                        19872000: "230d"
                    },
                    updatemode="mouseup", 
                    tooltip={"placement": "bottom"}
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
                        value=15000,
                        placeholder="bins (1-30000)",
                        style={
                            "marginRight": "10px", 
                            "width": "80px",
                            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                            "color": CYBERPUNK_COLORS['text_primary'],
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_pink']}",
                            "borderRadius": "5px",
                            "fontFamily": "'Courier New', monospace"
                        }
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
                            })
                ], style={"marginBottom": "10px"}),
                
                dcc.Slider(
                    id="bins", 
                    min=BINS_MIN, 
                    max=BINS_MAX, 
                    step=1, 
                    value=15000,
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
                    tooltip={"placement": "bottom"}
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

# ───────────────────────────── callback ────────────────────────────────────
@app.callback(
    Output("chi2-graph", "figure"),
    Output("start-date-readout", "children"),
    Output("start-time-readout", "children"),
    Output("len-readout", "children"),
    Output("bins-readout", "children"),
    Output("status-indicator", "children"),
    Output("date-picker", "date"),
    Output("time-input", "value"),
    Output("window-length-input", "value"),
    Output("bin-count-input", "value"),
    Output("start-date", "value"),
    Output("start-time", "value"),
    Output("len", "value"),
    Output("bins", "value"),
    Input("start-date", "value"), Input("start-time", "value"), Input("len", "value"), Input("bins", "value"),
    Input("date-picker", "date"), Input("time-input", "value"), Input("window-length-input", "value"), Input("bin-count-input", "value")
)
def update_graph(start_date_days, start_time_seconds, window_len, bins, 
                date_picker, time_input, window_length_input, bin_count_input):
    import time
    from dash import ctx
    
    start_time = time.time()
    
    # Determine which input triggered the callback and use that value
    triggered_id = ctx.triggered_id if ctx.triggered else None
    
    # Initialize values with defaults
    start_date_days = int(start_date_days or 0)
    start_time_seconds = int(start_time_seconds or 0)
    window_len = max(int(window_len or 15000), LEN_MIN_S)  # 4 hours 10 minutes
    bins = max(int(bins or 15000), BINS_MIN)
    
    # Handle date synchronization
    if triggered_id == "date-picker" and date_picker:
        # Date picker was changed - update slider
        selected_date = _dt.strptime(date_picker, "%Y-%m-%d").date()
        start_date_days = (selected_date - DATE_MIN).days
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
    else:
        # Use slider value
        bins = max(int(bins), BINS_MIN)
    
    # Combine date and time into datetime
    start_ts = _dt.combine(selected_date, selected_time, tzinfo=_tz.utc).timestamp()

    # Check if data is cached or needs to be fetched
    cache_key = (start_ts, window_len, bins)
    is_cached = CACHE.get(cache_key) is not None
    
    df = query_bq(start_ts, window_len, bins)
    
    # Calculate timing (always do this)
    elapsed_time = time.time() - start_time
    
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(
                text=f"GCP EGG Statistical Analysis<br><sub>No data found • {elapsed_time:.2f}s</sub>",
                x=0.5,
                xanchor="center",
                font=dict(color=CYBERPUNK_COLORS['neon_pink'])
            ),
            xaxis_title="Minutes from window start",
            yaxis_title="Cumulative χ²",
            annotations=[dict(
                text="No data in selected window", 
                showarrow=False,
                font=dict(color=CYBERPUNK_COLORS['neon_cyan'])
            )],
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
            paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
            font=dict(
                color=CYBERPUNK_COLORS['text_primary'],
                family="'Courier New', monospace"
            )
        )
        
        # Get data range info for status string
        first_date_str, last_date_str, total_count = get_data_range_info()
        status_str = f"⚠ No data found in {elapsed_time:.2f}s | Total egg basket data from {first_date_str} to {last_date_str}, total rows {total_count:,}"
        
        return (fig, "No data", "", "", "", status_str,
                selected_date, selected_time.strftime("%H:%M"), window_len, bins,
                start_date_days, start_time_seconds, window_len, bins)



    # Calculate x-axis values and determine appropriate units
    seconds_per_bin = window_len / bins
    
    # Choose appropriate time unit based on total window length for better readability
    if window_len < 3600:  # Less than 1 hour
        time_unit = "minutes"
        conversion_factor = 60
        x = df["bin_idx"] * seconds_per_bin / 60
    elif window_len < 86400:  # Less than 1 day
        time_unit = "hours"
        conversion_factor = 3600
        x = df["bin_idx"] * seconds_per_bin / 3600
    elif window_len < 604800:  # Less than 1 week
        time_unit = "days"
        conversion_factor = 86400
        x = df["bin_idx"] * seconds_per_bin / 86400
    else:  # 1 week or more
        time_unit = "days"
        conversion_factor = 86400
        x = df["bin_idx"] * seconds_per_bin / 86400
    

    
    # Create single-axis plot for cumulative deviation of χ² based on Stouffer Z
    fig = go.Figure()
    
    # Add cumulative deviation trace with cyberpunk styling
    fig.add_trace(go.Scatter(
        x=x, 
        y=df["cum_stouffer_z"], 
        mode="lines",
        name="Cumulative deviation of χ² based on Stouffer Z",
        line=dict(
            color=CYBERPUNK_COLORS['neon_purple'],
            width=3,
            shape='spline'
        )
    ))
    
    # Cyberpunk-styled layout
    fig.update_layout(
        xaxis_title=f"Time from window start ({time_unit})",
        yaxis_title="Cumulative deviation of χ² based on Stouffer Z",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            x=0.02, 
            y=0.98,
            bgcolor=CYBERPUNK_COLORS['bg_dark'],
            bordercolor=CYBERPUNK_COLORS['neon_purple'],
            borderwidth=1,
            font=dict(color=CYBERPUNK_COLORS['text_primary'])
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
    
    bins_str  = f"Bins: {bins} (≈ {bin_duration_str}/bin) | Active Eggs: {df['avg_active_eggs'].mean():.1f}/{len(EGG_COLS)} | Method: (Stouffer Z)² - 1"
    
    # Status indicator with more details
    # Get data range info for status string
    first_date_str, last_date_str, total_count = get_data_range_info()
    
    if is_cached:
        status_str = f"✓ Cached data loaded in {elapsed_time:.2f}s | Total egg basket data from {first_date_str} to {last_date_str}, total rows {total_count:,}"
    else:
        status_str = f"🔄 BigQuery data fetched in {elapsed_time:.2f}s | Window: {window_len:,}s, Bins: {bins} | Total egg basket data from {first_date_str} to {last_date_str}, total rows {total_count:,}"
    
    # Return all outputs including the input components for synchronization
    return (fig, start_date_str, start_time_str, len_str, bins_str, status_str,
            selected_date, selected_time.strftime("%H:%M"), window_len, bins,
            start_date_days, start_time_seconds, window_len, bins)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
