"""
Dash web application to explore Global Consciousness Project (GCP) EGG data
stored in BigQuery and reproduce Nelson-style statistical analysis.

* Full egg list ({len(EGG_COLS)} columns)
* Implements both published GCP methods:
  - Stouffer Z (mean shift detection): Z_s = ΣZ_i/√N (dynamic N based on active eggs)
  - Chi-square (variance analysis): χ² = ΣZ_i² (excludes NULL eggs)
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
DEFAULT_TIME = _dt(2001, 9, 11, 12, 46, 0, tzinfo=_tz.utc).time()
LEN_MIN_S, LEN_MAX_S = 60, 30 * 24 * 3600                # 1 min – 30 days
BINS_MIN, BINS_MAX   = 1, 2000

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
    stouffer_z = " + ".join(stouffer_z_terms)
    stouffer_z = f"SAFE_DIVIDE({stouffer_z}, SQRT({null_count_block})) AS stouffer_z"
    
    # Calculate sum of squared Z-scores for variance analysis (only include non-null eggs)
    chi2_terms = []
    for c in EGG_COLS:
        chi2_terms.append(f"IF(z_{c} IS NOT NULL, POW(z_{c},2), 0)")
    chi2_sum = " + ".join(chi2_terms)

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
         {chi2_sum} AS chi2_sec,
         {stouffer_z},
         {null_count_block} AS active_eggs
  FROM z
),
bins AS (
  SELECT
    CAST(FLOOR(TIMESTAMP_DIFF(recorded_at, start_ts, SECOND)/sec_per_bin) AS INT64) AS bin_idx,
    SUM(chi2_sec) AS chi2_bin,
    SUM(stouffer_z) AS stouffer_z_sum,
    COUNT(*) AS seconds_in_bin,
    AVG(active_eggs) AS avg_active_eggs
  FROM sec
  GROUP BY bin_idx
)
SELECT bin_idx, chi2_bin, stouffer_z_sum, seconds_in_bin, avg_active_eggs
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
        df["cum_chi2"] = df["chi2_bin"].cumsum()
        df["cum_stouffer_z"] = df["stouffer_z_sum"].cumsum()
        CACHE.set(key, df, expire=3600)

    # Always print SQL and result for debugging and verification
    print("\n===== BigQuery SQL =====\n" + render_sql(start_ts, window_s, bins) + "\n========================")
    print(df.head(10).to_string(index=False))
    print("========================\n")

    return df

# ───────────────────────────── Dash layout ─────────────────────────────────
app = dash.Dash(__name__)
app.title = "GCP EGG Statistical Analysis Explorer"

app.layout = html.Div([
    html.H3("GCP EGG Statistical Analysis Explorer"),
    html.P([
        "Red line: Cumulative χ² (variance analysis) | ",
        "Blue line: Cumulative Stouffer Z (mean shift analysis)"
    ], style={"fontSize": "14px", "color": "gray", "marginBottom": "10px"}),
    
    # Loading wrapper around the graph
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            dcc.Graph(id="chi2-graph", style={"height": "70vh"})
        ]
    ),

    html.Label("Window start date (UTC)"),
    html.Div([
        dcc.DatePickerSingle(
            id="date-picker",
            date=DEFAULT_DATE,
            display_format="YYYY-MM-DD",
            style={"marginRight": "10px"}
        ),
        html.Span(" or use slider below", style={"fontSize": "12px", "color": "gray"})
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
    html.Div(id="start-date-readout", style={"marginBottom": "0.5rem"}),
    
    html.Label("Window start time (UTC)"),
    html.Div([
        dcc.Input(
            id="time-input",
            type="text",
            value=DEFAULT_TIME.strftime("%H:%M"),
            placeholder="HH:MM",
            style={"marginRight": "10px", "width": "100px"}
        ),
        html.Span(" or use slider below", style={"fontSize": "12px", "color": "gray"})
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
    html.Div(id="start-time-readout", style={"marginBottom": "1rem"}),

    html.Label("Window length (s)"),
    html.Div([
        dcc.Input(
            id="window-length-input",
            type="number",
            value=6*3600,
            min=LEN_MIN_S,
            max=LEN_MAX_S,
            step=60,
            style={"marginRight": "10px", "width": "120px"}
        ),
        html.Span(" seconds or use slider below", style={"fontSize": "12px", "color": "gray"})
    ], style={"marginBottom": "10px"}),
    dcc.Slider(
        id="len", min=LEN_MIN_S, max=LEN_MAX_S, step=60, value=6*3600,
        marks={
            60: "1m",
            43200: "12h",
            86400: "1d",
            172800: "2d",
            259200: "3d",
            604800: "1w",
            1209600: "2w",
            1814400: "3w",
            2592000: "30d"
        },
        updatemode="mouseup", tooltip={"placement": "bottom"}
    ),
    html.Div(id="len-readout", style={"marginBottom": "1rem"}),

    html.Label("Bin count"),
    html.Div([
        dcc.Input(
            id="bin-count-input",
            type="number",
            value=72,
            min=BINS_MIN,
            max=BINS_MAX,
            step=1,
            style={"marginRight": "10px", "width": "80px"}
        ),
        html.Span(" bins or use slider below", style={"fontSize": "12px", "color": "gray"})
    ], style={"marginBottom": "10px"}),
    dcc.Slider(
        id="bins", min=BINS_MIN, max=BINS_MAX, step=1, value=72,
        marks={
            1: "1",
            20: "20",
            75: "75",
            100: "100",
            200: "200",
            300: "300",
            400: "400",
            500: "500",
            600: "600",
            700: "700",
            800: "800",
            900: "900",
            1000: "1K",
            1100: "1.1K",
            1200: "1.2K",
            1300: "1.3K",
            1400: "1.4K",
            1500: "1.5K",
            1600: "1.6K",
            1700: "1.7K",
            1800: "1.8K",
            1900: "1.9K",
            2000: "2K"
        },
        updatemode="mouseup", tooltip={"placement": "bottom"}
    ),
    html.Div(id="bins-readout"),
    
    # Status indicator
    html.Div(id="status-indicator", style={"marginTop": "10px", "fontSize": "12px", "color": "gray"})
])

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
    
    # Handle date input
    if triggered_id == "date-picker" and date_picker:
        selected_date = _dt.strptime(date_picker, "%Y-%m-%d").date()
        start_date_days = (selected_date - DATE_MIN).days
    else:
        selected_date = DATE_MIN + _td(days=int(start_date_days or 0))
    
    # Handle time input
    if triggered_id == "time-input" and time_input:
        try:
            # Validate time format (HH:MM)
            if ":" in time_input and len(time_input.split(":")) == 2:
                hours, minutes = map(int, time_input.split(":"))
                if 0 <= hours <= 23 and 0 <= minutes <= 59:
                    selected_time = _dt.strptime(f"{hours:02d}:{minutes:02d}", "%H:%M").time()
                    start_time_seconds = hours * 3600 + minutes * 60
                else:
                    # Invalid time, use existing value
                    hours = int(start_time_seconds or 0) // 3600
                    minutes = (int(start_time_seconds or 0) % 3600) // 60
                    selected_time = _dt.strptime(f"{hours:02d}:{minutes:02d}", "%H:%M").time()
            else:
                # Invalid format, use existing value
                hours = int(start_time_seconds or 0) // 3600
                minutes = (int(start_time_seconds or 0) % 3600) // 60
                selected_time = _dt.strptime(f"{hours:02d}:{minutes:02d}", "%H:%M").time()
        except (ValueError, TypeError):
            # Invalid input, use existing value
            hours = int(start_time_seconds or 0) // 3600
            minutes = (int(start_time_seconds or 0) % 3600) // 60
            selected_time = _dt.strptime(f"{hours:02d}:{minutes:02d}", "%H:%M").time()
    else:
        hours = int(start_time_seconds or 0) // 3600
        minutes = (int(start_time_seconds or 0) % 3600) // 60
        seconds = int(start_time_seconds or 0) % 60
        selected_time = _dt.strptime(f"{hours:02d}:{minutes:02d}:{seconds:02d}", "%H:%M:%S").time()
    
    # Handle window length input
    if triggered_id == "window-length-input" and window_length_input is not None:
        window_len = max(int(window_length_input), LEN_MIN_S)
    else:
        window_len = max(int(window_len or 1), 1)
    
    # Handle bin count input
    if triggered_id == "bin-count-input" and bin_count_input is not None:
        bins = max(int(bin_count_input), BINS_MIN)
    else:
        bins = max(int(bins or 1), 1)
    
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
                xanchor="center"
            ),
            xaxis_title="Minutes from window start",
            yaxis_title="Cumulative χ²",
            annotations=[dict(text="No data in selected window", showarrow=False)],
            margin=dict(l=40, r=40, t=60, b=40)
        )
        status_str = f"⚠ No data found in {elapsed_time:.2f}s"
        return (fig, "No data", "", "", "", status_str,
                selected_date, selected_time.strftime("%H:%M"), window_len, bins)

    # Debug: Print DataFrame contents
    print("DEBUG: DataFrame contents:")
    print(df.to_string(index=False))
    print(f"DEBUG: DataFrame shape: {df.shape}")
    print(f"DEBUG: cum_chi2 values: {df['cum_chi2'].tolist()}")

    x = df["bin_idx"] * (window_len / bins) / 60  # minutes
    print(f"DEBUG: x values: {x.tolist()}")
    print(f"DEBUG: y values: {df['cum_chi2'].tolist()}")
    print(f"DEBUG: Stouffer Z values: {df['cum_stouffer_z'].tolist()}")
    print(f"DEBUG: Average active eggs: {df['avg_active_eggs'].tolist()}")
    
    # Create dual-axis plot with both chi-square and Stouffer Z
    fig = go.Figure()
    
    # Add chi-square trace (left y-axis)
    fig.add_trace(go.Scatter(
        x=x, 
        y=df["cum_chi2"], 
        mode="lines",
        name="Cumulative χ²",
        line=dict(color="red"),
        yaxis="y"
    ))
    
    # Add Stouffer Z trace (right y-axis)
    fig.add_trace(go.Scatter(
        x=x, 
        y=df["cum_stouffer_z"], 
        mode="lines",
        name="Cumulative Stouffer Z",
        line=dict(color="blue"),
        yaxis="y2"
    ))
    
    
    fig.update_layout(
        xaxis_title="Minutes from window start",
        yaxis=dict(title="Cumulative χ²", side="left", color="red"),
        yaxis2=dict(title="Cumulative Stouffer Z", side="right", color="blue", overlaying="y"),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(x=0.02, y=0.98)
    )
    
    start_date_str = f"Date: {selected_date.strftime('%Y-%m-%d')}"
    start_time_str = f"Time: {selected_time.strftime('%H:%M:%S')}"
    len_str   = f"Length: {_td(seconds=window_len)} ({window_len:,} s)"
    bins_str  = f"Bins: {bins} (≈ {window_len/bins:.1f} s/bin) | Active Eggs: {df['avg_active_eggs'].mean():.1f}/{len(EGG_COLS)}"
    
    # Status indicator with more details
    if is_cached:
        status_str = f"✓ Cached data loaded in {elapsed_time:.2f}s"
    else:
        status_str = f"🔄 BigQuery data fetched in {elapsed_time:.2f}s | Window: {window_len:,}s, Bins: {bins}"
    
    # Return all outputs including the input components for synchronization
    return (fig, start_date_str, start_time_str, len_str, bins_str, status_str,
            selected_date, selected_time.strftime("%H:%M"), window_len, bins)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
