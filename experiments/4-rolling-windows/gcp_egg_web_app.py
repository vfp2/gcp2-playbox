"""
Dash web application to explore Global Consciousness Project (GCP) EGG data
stored in BigQuery and reproduce Nelson-style cumulative χ² plots.

* Full egg list (128 columns)
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
BASELINE_TBL = os.getenv("BASELINE_TABLE", "baseline_day")

START_MIN_TS = _dt(2001, 3, 3, tzinfo=_tz.utc).timestamp()
START_MAX_TS = _dt(2001, 5, 3, 23, 59, 59, tzinfo=_tz.utc).timestamp()
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
    # Use SAFE_DIVIDE to avoid division-by-zero; COALESCE in chi² sum to treat NULL as 0
    z_block     = ",\n".join(
        f"    SAFE_DIVIDE(({c} - COALESCE(b.mu_{c.split('_')[1]}, 100)), COALESCE(b.sigma_{c.split('_')[1]}, 1)) AS z_{c}"
        for c in EGG_COLS
    )
    chi2_sum = " + ".join(f"COALESCE(POW(z_{c},2),0)" for c in EGG_COLS)

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
  FROM raw r
  JOIN `{GCP_PROJECT}.{GCP_DATASET}.{BASELINE_TBL}` b
    ON b.day = DATE(r.recorded_at)
),
sec AS (
  SELECT recorded_at, {chi2_sum} AS chi2_sec FROM z
),
bins AS (
  SELECT
    CAST(FLOOR(TIMESTAMP_DIFF(recorded_at, start_ts, SECOND)/sec_per_bin) AS INT64) AS bin_idx,
    SUM(chi2_sec) AS chi2_bin
  FROM sec
  GROUP BY bin_idx
)
SELECT bin_idx, chi2_bin
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
        CACHE.set(key, df, expire=3600)

    # Always print SQL and result for debugging and verification
    print("\n===== BigQuery SQL =====\n" + render_sql(start_ts, window_s, bins) + "\n========================")
    print(df.head(10).to_string(index=False))
    print("========================\n")

    return df

# ───────────────────────────── Dash layout ─────────────────────────────────
app = dash.Dash(__name__)
app.title = "GCP EGG Cumulative χ² Explorer"

app.layout = html.Div([
    html.H3("GCP EGG Cumulative χ² Explorer"),
    dcc.Graph(id="chi2-graph", style={"height": "70vh"}),

    html.Label("Window start (UTC)"),
    dcc.Slider(
        id="start", min=START_MIN_TS, max=START_MAX_TS, step=3600,
        value=START_MIN_TS, updatemode="mouseup",
        tooltip={"placement": "bottom"}
    ),
    html.Div(id="start-readout", style={"marginBottom": "1rem"}),

    html.Label("Window length (s)"),
    dcc.Slider(
        id="len", min=LEN_MIN_S, max=LEN_MAX_S, step=60, value=6*3600,
        marks={60:"1m",3600:"1h",86400:"1d",2592000:"30d"},
        updatemode="mouseup", tooltip={"placement": "bottom"}
    ),
    html.Div(id="len-readout", style={"marginBottom": "1rem"}),

    html.Label("Bin count"),
    dcc.Slider(
        id="bins", min=BINS_MIN, max=BINS_MAX, step=10, value=72,
        updatemode="mouseup", tooltip={"placement": "bottom"}
    ),
    html.Div(id="bins-readout"),
])

# ───────────────────────────── callback ────────────────────────────────────
@app.callback(
    Output("chi2-graph", "figure"),
    Output("start-readout", "children"),
    Output("len-readout", "children"),
    Output("bins-readout", "children"),
    Input("start", "value"), Input("len", "value"), Input("bins", "value")
)
def update_graph(start_ts, window_len, bins):
    window_len = max(int(window_len or 1), 1)
    bins       = max(int(bins or 1), 1)

    df = query_bq(start_ts, window_len, bins)
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="Minutes from window start",
            yaxis_title="Cumulative χ²",
            annotations=[dict(text="No data in selected window", showarrow=False)]
        )
        return fig, "No data", "", ""

    x = df["bin_idx"] * (window_len / bins) / 60  # minutes
    fig = go.Figure(go.Scatter(x=x, y=df["cum_chi2"], mode="lines"))
    fig.update_layout(
        xaxis_title="Minutes from window start",
        yaxis_title="Cumulative χ²",
        margin=dict(l=40, r=20, t=40, b=40)
    )

    start_str = f"Start: {_dt.fromtimestamp(start_ts, _tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    len_str   = f"Length: {_td(seconds=window_len)} ({window_len:,} s)"
    bins_str  = f"Bins: {bins} (≈ {window_len/bins:.1f} s/bin)"
    return fig, start_str, len_str, bins_str

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
