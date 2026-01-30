#!/usr/bin/env python3
"""
GCP Holmberg Analysis Dashboard

Web-based visualization of Ulf Holmberg's Max[Z]-market correlation methodology
applied to both GCP1 and GCP2 data sources.

Implements:
- Daily Max[Z] calculation from GCP data
- Correlation with VIX and SPY returns
- Lag analysis and threshold conditioning
- Out-of-sample backtesting simulation
- Interactive visualizations

Run: python holmberg_dashboard.py
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ───────────────────────────── Configuration ──────────────────────────────
GCP2_DATA_DIR = Path(__file__).parent.parent.parent / "gcp2.net-rng-data-downloaded"
GCP2_NETWORK_DIR = GCP2_DATA_DIR / "network"

ROLLING_WINDOW = 3600  # 1-hour rolling window for Z-score computation
MIN_ROLLING_PERIODS = 360  # minimum 6 minutes of data

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
}

# ───────────────────────────── GCP2 Data Loading ──────────────────────────
@lru_cache(maxsize=4)
def load_gcp2_network_data(months: int = 6) -> pd.DataFrame:
    """Load GCP2 Global Network data for the specified number of recent months."""
    network_dir = GCP2_NETWORK_DIR / "global_network"
    if not network_dir.exists():
        return pd.DataFrame()

    # Find all CSV files (already extracted)
    csv_files = []
    for year_dir in sorted(network_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for csv_file in sorted(year_dir.glob("*.csv")):
            if ".csv.zip" in csv_file.name:
                continue
            parts = csv_file.stem.split("_")
            try:
                year = int(parts[-2])
                month = int(parts[-1])
                csv_files.append((year, month, csv_file))
            except (ValueError, IndexError):
                continue

    csv_files.sort(key=lambda x: (x[0], x[1]))
    selected = csv_files[-months:] if months > 0 else csv_files

    if not selected:
        return pd.DataFrame()

    frames = []
    for year, month, csv_path in selected:
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
    combined.reset_index(drop=True, inplace=True)

    combined["datetime_utc"] = pd.to_datetime(combined["epoch_time_utc"], unit="s", utc=True)
    combined["date"] = combined["datetime_utc"].dt.date
    combined["hour_utc"] = combined["datetime_utc"].dt.hour

    return combined


def compute_rolling_z(nc_series: pd.Series) -> pd.Series:
    """Compute rolling Z-score of network_coherence over 1-hour window."""
    roll = nc_series.rolling(ROLLING_WINDOW, min_periods=MIN_ROLLING_PERIODS)
    return (nc_series - roll.mean()) / (roll.std(ddof=0) + 1e-9)


def compute_daily_gcp_metrics(gcp_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily metrics from per-second GCP2 data."""
    if gcp_df.empty:
        return pd.DataFrame()

    gcp_df = gcp_df.copy()
    gcp_df["rolling_z"] = compute_rolling_z(gcp_df["network_coherence"])

    daily = gcp_df.groupby("date").agg(
        max_rolling_z=("rolling_z", lambda x: x.dropna().abs().max() if x.notna().any() else np.nan),
        peak_nc=("network_coherence", "max"),
        mean_nc=("network_coherence", "mean"),
        netvar=("network_coherence", lambda x: (x ** 2).mean()),
        median_devices=("active_devices", "median"),
        n_seconds=("network_coherence", "count"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ───────────────────────────── Market Data ────────────────────────────────
@lru_cache(maxsize=16)
def fetch_market_data(start_date: str, end_date: str, custom_ticker: str = "") -> pd.DataFrame:
    """Fetch SPY, VIX, and optionally a custom ticker via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Run: pip install yfinance")
        return pd.DataFrame()

    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)

    if spy.empty:
        return pd.DataFrame()

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
    else:
        market["vix_close"] = np.nan
        market["vix_change"] = np.nan
        market["vix_pct_change"] = np.nan

    # Fetch custom ticker if provided
    if custom_ticker and custom_ticker.strip():
        ticker = custom_ticker.strip().upper()
        try:
            custom_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not custom_data.empty:
                if isinstance(custom_data.columns, pd.MultiIndex):
                    custom_data.columns = custom_data.columns.get_level_values(0)
                custom_aligned = custom_data["Close"].reindex(spy.index)
                market["custom_close"] = custom_aligned.values
                market["custom_return"] = custom_aligned.pct_change().values
                market["custom_ticker"] = ticker
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    market["date"] = pd.to_datetime(market["date"]).dt.tz_localize(None)
    return market


# ───────────────────────────── Correlation Analysis ───────────────────────
def pearson_with_pvalue(x, y):
    """Pearson correlation with p-value, handling NaN."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 5:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return r, p


def lag_analysis(gcp_series, market_series, max_lag=5):
    """Test correlations at different lags."""
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


# ───────────────────────────── Backtesting Simulation ─────────────────────
def run_backtest(merged_df: pd.DataFrame, signal_col: str = "max_rolling_z",
                 threshold_pct: float = 75, hold_days: int = 1) -> pd.DataFrame:
    """
    Run a simple backtesting simulation based on Max[Z] signals.

    Strategy: Go long SPY when Max[Z] exceeds the Nth percentile threshold.
    This replicates Holmberg's out-of-sample trading tests.
    """
    if merged_df.empty or signal_col not in merged_df.columns:
        return pd.DataFrame()

    df = merged_df.copy().sort_values("date").reset_index(drop=True)
    threshold = df[signal_col].quantile(threshold_pct / 100)

    # Initialize portfolio tracking
    portfolio_value = 100.0
    buy_hold_value = 100.0
    positions = []
    holding = False
    hold_until = None

    for i, row in df.iterrows():
        date = row["date"]
        spy_return = row["spy_return"] if not np.isnan(row["spy_return"]) else 0
        signal = row[signal_col] if not np.isnan(row[signal_col]) else 0

        # Buy & hold benchmark
        buy_hold_value *= (1 + spy_return)

        # Strategy: Enter when signal exceeds threshold
        if signal >= threshold and not holding:
            holding = True
            hold_until = date + timedelta(days=hold_days)

        # Apply returns if in position
        if holding:
            portfolio_value *= (1 + spy_return)
            if date >= hold_until:
                holding = False

        positions.append({
            "date": date,
            "signal": signal,
            "spy_return": spy_return * 100,
            "in_position": holding,
            "portfolio_value": portfolio_value,
            "buy_hold_value": buy_hold_value,
            "excess_return": (portfolio_value - buy_hold_value) / buy_hold_value * 100,
        })

    return pd.DataFrame(positions)


# ───────────────────────────── Dashboard App ──────────────────────────────
def create_layout():
    """Create the dashboard layout."""
    return html.Div([
    # Header
    html.Div([
        html.H1("GCP HOLMBERG ANALYSIS DASHBOARD",
                style={
                    "textAlign": "center",
                    "color": CYBERPUNK_COLORS['text_primary'],
                    "fontSize": "2.5rem",
                    "fontWeight": "900",
                    "textShadow": f"0 0 20px {CYBERPUNK_COLORS['neon_purple']}",
                    "fontFamily": "'Orbitron', monospace",
                }),
        html.P([
            "Replicating Ulf Holmberg's Max[Z] vs VIX/SPY correlation methodology • ",
            "GCP2 Network Coherence Analysis • Out-of-Sample Backtesting"
        ], style={
            "fontSize": "14px",
            "color": CYBERPUNK_COLORS['neon_cyan'],
            "textAlign": "center",
            "fontFamily": "'Courier New', monospace",
        }),
    ], style={
        "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
        "padding": "20px",
        "borderRadius": "15px",
        "border": f"2px solid {CYBERPUNK_COLORS['neon_purple']}",
        "marginBottom": "20px"
    }),

    # Controls Row 1
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Data Months:", style={"color": CYBERPUNK_COLORS['neon_green']}),
                dcc.Dropdown(
                    id="months-select",
                    options=[
                        {"label": "3 months", "value": 3},
                        {"label": "6 months", "value": 6},
                        {"label": "12 months", "value": 12},
                        {"label": "All available", "value": 0},
                    ],
                    value=6,
                    clearable=False,
                    style={"backgroundColor": CYBERPUNK_COLORS['bg_dark']}
                ),
            ], width=2),
            dbc.Col([
                html.Label("Backtest Start Date:", style={"color": CYBERPUNK_COLORS['neon_green']}),
                dcc.DatePickerSingle(
                    id="backtest-start-date",
                    date=None,  # None means use all available data
                    placeholder="Use all data",
                    display_format="YYYY-MM-DD",
                    style={"backgroundColor": CYBERPUNK_COLORS['bg_dark']}
                ),
            ], width=2),
            dbc.Col([
                html.Label("Custom Ticker:", style={"color": CYBERPUNK_COLORS['neon_green']}),
                dcc.Input(
                    id="custom-ticker-input",
                    type="text",
                    placeholder="e.g., QQQ, BTC-USD",
                    debounce=True,
                    style={
                        "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                        "color": CYBERPUNK_COLORS['text_primary'],
                        "border": f"1px solid {CYBERPUNK_COLORS['neon_green']}",
                        "borderRadius": "5px",
                        "padding": "8px",
                        "width": "100%"
                    }
                ),
            ], width=2),
            dbc.Col([
                html.Label("Backtest Threshold:", style={"color": CYBERPUNK_COLORS['neon_green']}),
                dcc.Slider(
                    id="threshold-slider",
                    min=50, max=95, step=5, value=75,
                    marks={i: f"P{i}" for i in range(50, 100, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], width=3),
            dbc.Col([
                html.Label("Hold Days:", style={"color": CYBERPUNK_COLORS['neon_green']}),
                dcc.Dropdown(
                    id="hold-days-select",
                    options=[{"label": f"{d} day{'s' if d > 1 else ''}", "value": d} for d in [1, 2, 3, 5, 10]],
                    value=1,
                    clearable=False,
                    style={"backgroundColor": CYBERPUNK_COLORS['bg_dark']}
                ),
            ], width=1),
            dbc.Col([
                html.Br(),
                dbc.Button("RUN ANALYSIS", id="run-btn", color="success", size="lg",
                           style={"backgroundColor": CYBERPUNK_COLORS['neon_green'],
                                  "color": CYBERPUNK_COLORS['bg_dark'],
                                  "fontWeight": "bold"}),
            ], width=2),
        ]),
    ], style={
        "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
        "padding": "20px",
        "borderRadius": "15px",
        "border": f"2px solid {CYBERPUNK_COLORS['neon_green']}",
        "marginBottom": "20px"
    }),

    # Loading indicator
    dcc.Loading(
        id="loading",
        type="circle",
        color=CYBERPUNK_COLORS['neon_cyan'],
        children=[
            # Summary Stats
            html.Div(id="summary-stats", style={"marginBottom": "20px"}),

            # Correlation Table
            html.Div(id="correlation-table", style={"marginBottom": "20px"}),

            # Charts
            dcc.Graph(id="time-series-chart", style={"marginBottom": "20px"}),
            dcc.Graph(id="correlation-chart", style={"marginBottom": "20px"}),
            dcc.Graph(id="backtest-chart", style={"marginBottom": "20px"}),
            dcc.Graph(id="lag-chart", style={"marginBottom": "20px"}),
        ]
    ),

    # Footer
    html.Div([
        dcc.Link("← Back to Portal", href="/",
                 style={"color": CYBERPUNK_COLORS['neon_cyan'], "fontSize": "16px"}),
    ], style={"textAlign": "center", "marginTop": "30px"})

], style={
    "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
    "minHeight": "100vh",
    "padding": "20px"
})


def register_callbacks(app):
    """Register all callbacks for the dashboard."""
    @app.callback(
        [Output("summary-stats", "children"),
         Output("correlation-table", "children"),
         Output("time-series-chart", "figure"),
         Output("correlation-chart", "figure"),
         Output("backtest-chart", "figure"),
         Output("lag-chart", "figure")],
        [Input("run-btn", "n_clicks")],
        [State("months-select", "value"),
         State("threshold-slider", "value"),
         State("hold-days-select", "value"),
         State("backtest-start-date", "date"),
         State("custom-ticker-input", "value")],
        prevent_initial_call=False
    )
    def update_analysis(n_clicks, months, threshold_pct, hold_days, backtest_start_date, custom_ticker):
            """Run the full Holmberg analysis and update all visualizations."""

            # Load GCP2 data
            gcp_df = load_gcp2_network_data(months)
            if gcp_df.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="No GCP2 data available",
                    paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                    plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                    font=dict(color=CYBERPUNK_COLORS['text_primary'])
                )
                return (
                    html.P("No GCP2 data found. Ensure CSV files are in the network directory.",
                           style={"color": CYBERPUNK_COLORS['neon_pink']}),
                    html.Div(),
                    empty_fig, empty_fig, empty_fig, empty_fig
                )

            # Compute daily metrics
            gcp_daily = compute_daily_gcp_metrics(gcp_df)

            # Fetch market data (including custom ticker if provided)
            start_date = gcp_daily["date"].min().strftime("%Y-%m-%d")
            end_date = (gcp_daily["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
            custom_ticker = custom_ticker or ""
            market = fetch_market_data(start_date, end_date, custom_ticker)

            if market.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="No market data available",
                    paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                    plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                    font=dict(color=CYBERPUNK_COLORS['text_primary'])
                )
                return (
                    html.P("Failed to fetch market data from yfinance.",
                           style={"color": CYBERPUNK_COLORS['neon_pink']}),
                    html.Div(),
                    empty_fig, empty_fig, empty_fig, empty_fig
                )

            # Merge datasets
            merged = gcp_daily.merge(market, on="date", how="inner")

            # Check for custom ticker in data
            has_custom_ticker = "custom_close" in merged.columns and merged["custom_close"].notna().any()
            custom_ticker_name = merged["custom_ticker"].iloc[0] if has_custom_ticker and "custom_ticker" in merged.columns else ""

            if len(merged) < 10:
                empty_fig = go.Figure()
                return (
                    html.P(f"Insufficient data: only {len(merged)} trading days",
                           style={"color": CYBERPUNK_COLORS['neon_pink']}),
                    html.Div(),
                    empty_fig, empty_fig, empty_fig, empty_fig
                )

            # Filter by backtest start date if provided
            backtest_merged = merged.copy()
            if backtest_start_date:
                backtest_start = pd.to_datetime(backtest_start_date)
                backtest_merged = merged[merged["date"] >= backtest_start].copy()
                if len(backtest_merged) < 5:
                    backtest_merged = merged.copy()  # Fall back to all data if too few points

            # Calculate correlations (on full merged data)
            rz_vix_r, rz_vix_p = pearson_with_pvalue(merged["max_rolling_z"].values, merged["vix_close"].values)
            rz_dvix_r, rz_dvix_p = pearson_with_pvalue(merged["max_rolling_z"].values, merged["vix_change"].values)
            rz_spy_r, rz_spy_p = pearson_with_pvalue(merged["max_rolling_z"].values, merged["spy_return"].values)
            pnc_vix_r, pnc_vix_p = pearson_with_pvalue(merged["peak_nc"].values, merged["vix_close"].values)
            pnc_spy_r, pnc_spy_p = pearson_with_pvalue(merged["peak_nc"].values, merged["spy_return"].values)

            # Calculate custom ticker correlations if available
            rz_custom_r, rz_custom_p = (np.nan, np.nan)
            if has_custom_ticker:
                rz_custom_r, rz_custom_p = pearson_with_pvalue(merged["max_rolling_z"].values, merged["custom_return"].values)

            # Run backtest (uses filtered data if start date provided)
            backtest = run_backtest(backtest_merged, "max_rolling_z", threshold_pct, hold_days)

            # Run lag analysis
            lags = lag_analysis(merged["max_rolling_z"], merged["vix_change"])

            # ── Summary Stats ──
            final_excess = backtest["excess_return"].iloc[-1] if not backtest.empty else 0
            backtest_start_str = backtest_merged["date"].min().strftime('%Y-%m-%d') if not backtest_merged.empty else "N/A"

            # Build stats columns
            stats_cols = [
                dbc.Col([
                    html.Div([
                        html.H4(f"{len(merged)}", style={"color": CYBERPUNK_COLORS['neon_cyan'], "fontSize": "2rem", "marginBottom": "0"}),
                        html.P("Trading Days", style={"color": CYBERPUNK_COLORS['text_secondary']})
                    ], style={"textAlign": "center"})
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H4(f"{merged['median_devices'].median():.0f}", style={"color": CYBERPUNK_COLORS['neon_green'], "fontSize": "2rem", "marginBottom": "0"}),
                        html.P("Avg Devices", style={"color": CYBERPUNK_COLORS['text_secondary']})
                    ], style={"textAlign": "center"})
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H4(f"{rz_vix_r:.3f}", style={"color": CYBERPUNK_COLORS['neon_purple'], "fontSize": "2rem", "marginBottom": "0"}),
                        html.P(f"Z vs VIX (p={rz_vix_p:.3f})", style={"color": CYBERPUNK_COLORS['text_secondary']})
                    ], style={"textAlign": "center"})
                ], width=2),
                dbc.Col([
                    html.Div([
                        html.H4(f"{rz_spy_r:.3f}", style={"color": CYBERPUNK_COLORS['neon_yellow'], "fontSize": "2rem", "marginBottom": "0"}),
                        html.P(f"Z vs SPY (p={rz_spy_p:.3f})", style={"color": CYBERPUNK_COLORS['text_secondary']})
                    ], style={"textAlign": "center"})
                ], width=2),
            ]

            # Add custom ticker correlation if available
            if has_custom_ticker and not np.isnan(rz_custom_r):
                stats_cols.append(dbc.Col([
                    html.Div([
                        html.H4(f"{rz_custom_r:.3f}", style={"color": CYBERPUNK_COLORS['neon_pink'], "fontSize": "2rem", "marginBottom": "0"}),
                        html.P(f"Z vs {custom_ticker_name} (p={rz_custom_p:.3f})", style={"color": CYBERPUNK_COLORS['text_secondary']})
                    ], style={"textAlign": "center"})
                ], width=2))
            else:
                excess_color = CYBERPUNK_COLORS['neon_green'] if final_excess > 0 else CYBERPUNK_COLORS['neon_pink']
                stats_cols.append(dbc.Col([
                    html.Div([
                        html.H4(f"{final_excess:+.1f}%", style={"color": excess_color, "fontSize": "2rem", "marginBottom": "0"}),
                        html.P(f"Backtest (from {backtest_start_str})", style={"color": CYBERPUNK_COLORS['text_secondary']})
                    ], style={"textAlign": "center"})
                ], width=2))

            stats_cols.append(dbc.Col([
                html.Div([
                    html.H4(f"{merged['date'].min().strftime('%Y-%m-%d')}", style={"color": CYBERPUNK_COLORS['text_primary'], "fontSize": "1rem", "marginBottom": "0"}),
                    html.P(f"to {merged['date'].max().strftime('%Y-%m-%d')}", style={"color": CYBERPUNK_COLORS['text_secondary']})
                ], style={"textAlign": "center"})
            ], width=2))

            summary_stats = html.Div([
                dbc.Row(stats_cols)
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "15px",
                "borderRadius": "10px",
                "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}",
            })

            # ── Correlation Table ──
            def sig_star(p):
                if np.isnan(p): return ""
                if p < 0.001: return "***"
                if p < 0.01: return "**"
                if p < 0.05: return "*"
                return ""

            # Build table headers
            table_headers = [
                html.Th("Metric", style={"color": CYBERPUNK_COLORS['neon_cyan']}),
                html.Th("VIX Level", style={"color": CYBERPUNK_COLORS['text_primary']}),
                html.Th("VIX Change", style={"color": CYBERPUNK_COLORS['text_primary']}),
                html.Th("SPY Return", style={"color": CYBERPUNK_COLORS['text_primary']}),
            ]
            if has_custom_ticker:
                table_headers.append(html.Th(f"{custom_ticker_name} Return", style={"color": CYBERPUNK_COLORS['neon_pink']}))

            # Build table rows
            row1_cells = [
                html.Td("Max Rolling-Z", style={"color": CYBERPUNK_COLORS['neon_yellow']}),
                html.Td(f"{rz_vix_r:.4f} {sig_star(rz_vix_p)}"),
                html.Td(f"{rz_dvix_r:.4f} {sig_star(rz_dvix_p)}"),
                html.Td(f"{rz_spy_r:.4f} {sig_star(rz_spy_p)}"),
            ]
            if has_custom_ticker:
                row1_cells.append(html.Td(f"{rz_custom_r:.4f} {sig_star(rz_custom_p)}"))

            pnc_custom_r, pnc_custom_p = (np.nan, np.nan)
            if has_custom_ticker:
                pnc_custom_r, pnc_custom_p = pearson_with_pvalue(merged['peak_nc'].values, merged['custom_return'].values)

            row2_cells = [
                html.Td("Peak NC", style={"color": CYBERPUNK_COLORS['neon_green']}),
                html.Td(f"{pnc_vix_r:.4f} {sig_star(pnc_vix_p)}"),
                html.Td(f"{pearson_with_pvalue(merged['peak_nc'].values, merged['vix_change'].values)[0]:.4f}"),
                html.Td(f"{pnc_spy_r:.4f} {sig_star(pnc_spy_p)}"),
            ]
            if has_custom_ticker:
                row2_cells.append(html.Td(f"{pnc_custom_r:.4f} {sig_star(pnc_custom_p)}"))

            correlation_table = html.Div([
                html.H4("Correlation Analysis", style={"color": CYBERPUNK_COLORS['neon_purple'], "marginBottom": "10px"}),
                dbc.Table([
                    html.Thead(html.Tr(table_headers)),
                    html.Tbody([
                        html.Tr(row1_cells),
                        html.Tr(row2_cells),
                    ])
                ], bordered=True, dark=True, hover=True, size="sm")
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "15px",
                "borderRadius": "10px",
                "border": f"1px solid {CYBERPUNK_COLORS['neon_purple']}",
            })

            # ── Time Series Chart ──
            # Compute rolling correlations (30-day window)
            rolling_window = 30
            merged_sorted = merged.sort_values("date").copy()
            merged_sorted["rolling_corr_vix"] = merged_sorted["max_rolling_z"].rolling(rolling_window).corr(merged_sorted["vix_change"])
            merged_sorted["rolling_corr_spy"] = merged_sorted["max_rolling_z"].rolling(rolling_window).corr(merged_sorted["spy_return"])
            if has_custom_ticker:
                merged_sorted["rolling_corr_custom"] = merged_sorted["max_rolling_z"].rolling(rolling_window).corr(merged_sorted["custom_return"])

            n_rows = 5 if has_custom_ticker else 4
            subplot_titles = ["Max Rolling-Z (GCP2)", "VIX Level", "SPY Price", "Rolling Correlation (30-day)"]
            if has_custom_ticker:
                subplot_titles.insert(3, f"{custom_ticker_name} Price")

            ts_fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                                   subplot_titles=subplot_titles,
                                   vertical_spacing=0.05)

            ts_fig.add_trace(go.Scatter(x=merged_sorted["date"], y=merged_sorted["max_rolling_z"],
                                        mode="lines", name="Max Rolling-Z",
                                        line=dict(color=CYBERPUNK_COLORS['neon_purple'])), row=1, col=1)
            ts_fig.add_trace(go.Scatter(x=merged_sorted["date"], y=merged_sorted["vix_close"],
                                        mode="lines", name="VIX",
                                        line=dict(color=CYBERPUNK_COLORS['neon_pink'])), row=2, col=1)
            ts_fig.add_trace(go.Scatter(x=merged_sorted["date"], y=merged_sorted["spy_close"],
                                        mode="lines", name="SPY",
                                        line=dict(color=CYBERPUNK_COLORS['neon_green'])), row=3, col=1)

            if has_custom_ticker:
                ts_fig.add_trace(go.Scatter(x=merged_sorted["date"], y=merged_sorted["custom_close"],
                                            mode="lines", name=custom_ticker_name,
                                            line=dict(color=CYBERPUNK_COLORS['neon_yellow'])), row=4, col=1)
                corr_row = 5
            else:
                corr_row = 4

            # Add rolling correlation traces
            ts_fig.add_trace(go.Scatter(x=merged_sorted["date"], y=merged_sorted["rolling_corr_vix"],
                                        mode="lines", name="r(Z, VIX Δ)",
                                        line=dict(color=CYBERPUNK_COLORS['neon_pink'], width=1.5)), row=corr_row, col=1)
            ts_fig.add_trace(go.Scatter(x=merged_sorted["date"], y=merged_sorted["rolling_corr_spy"],
                                        mode="lines", name="r(Z, SPY %)",
                                        line=dict(color=CYBERPUNK_COLORS['neon_green'], width=1.5)), row=corr_row, col=1)
            if has_custom_ticker:
                ts_fig.add_trace(go.Scatter(x=merged_sorted["date"], y=merged_sorted["rolling_corr_custom"],
                                            mode="lines", name=f"r(Z, {custom_ticker_name} %)",
                                            line=dict(color=CYBERPUNK_COLORS['neon_yellow'], width=1.5)), row=corr_row, col=1)
            # Add zero line for correlation reference
            ts_fig.add_hline(y=0, line_dash="dash", line_color=CYBERPUNK_COLORS['text_secondary'],
                             line_width=1, row=corr_row, col=1)

            chart_height = 900 if has_custom_ticker else 750
            ts_fig.update_layout(
                title="Time Series: GCP2 Max[Z] vs Market Indicators",
                height=chart_height,
                paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                font=dict(color=CYBERPUNK_COLORS['text_primary']),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            ts_fig.update_xaxes(gridcolor=CYBERPUNK_COLORS['bg_light'])
            ts_fig.update_yaxes(gridcolor=CYBERPUNK_COLORS['bg_light'])
            # Set y-axis range for correlation panel
            ts_fig.update_yaxes(range=[-1, 1], row=corr_row, col=1)

            # ── Correlation Scatter Chart ──
            n_scatter_cols = 3 if has_custom_ticker else 2
            scatter_titles = ["Max[Z] vs VIX Change", "Max[Z] vs SPY Return"]
            if has_custom_ticker:
                scatter_titles.append(f"Max[Z] vs {custom_ticker_name} Return")

            corr_fig = make_subplots(rows=1, cols=n_scatter_cols, subplot_titles=scatter_titles)

            corr_fig.add_trace(go.Scatter(
                x=merged["max_rolling_z"], y=merged["vix_change"],
                mode="markers", name="vs VIX",
                marker=dict(color=CYBERPUNK_COLORS['neon_pink'], size=5, opacity=0.6)
            ), row=1, col=1)

            corr_fig.add_trace(go.Scatter(
                x=merged["max_rolling_z"], y=merged["spy_return"] * 100,
                mode="markers", name="vs SPY%",
                marker=dict(color=CYBERPUNK_COLORS['neon_green'], size=5, opacity=0.6)
            ), row=1, col=2)

            if has_custom_ticker:
                corr_fig.add_trace(go.Scatter(
                    x=merged["max_rolling_z"], y=merged["custom_return"] * 100,
                    mode="markers", name=f"vs {custom_ticker_name}%",
                    marker=dict(color=CYBERPUNK_COLORS['neon_yellow'], size=5, opacity=0.6)
                ), row=1, col=3)

            title_str = f"Scatter: Max[Z] Correlations (r_VIX={rz_dvix_r:.3f}, r_SPY={rz_spy_r:.3f}"
            if has_custom_ticker and not np.isnan(rz_custom_r):
                title_str += f", r_{custom_ticker_name}={rz_custom_r:.3f}"
            title_str += ")"

            corr_fig.update_layout(
                title=title_str,
                height=400,
                paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                font=dict(color=CYBERPUNK_COLORS['text_primary']),
            )
            corr_fig.update_xaxes(title_text="Max Rolling-Z", gridcolor=CYBERPUNK_COLORS['bg_light'])
            corr_fig.update_yaxes(gridcolor=CYBERPUNK_COLORS['bg_light'])

            # ── Backtest Chart ──
            bt_fig = go.Figure()
            if not backtest.empty:
                bt_fig.add_trace(go.Scatter(
                    x=backtest["date"], y=backtest["portfolio_value"],
                    mode="lines", name=f"Max[Z] Strategy (P{threshold_pct})",
                    line=dict(color=CYBERPUNK_COLORS['neon_cyan'], width=2)
                ))
                bt_fig.add_trace(go.Scatter(
                    x=backtest["date"], y=backtest["buy_hold_value"],
                    mode="lines", name="Buy & Hold SPY",
                    line=dict(color=CYBERPUNK_COLORS['text_secondary'], width=2, dash="dash")
                ))

            backtest_title = f"Backtest: Max[Z] > P{threshold_pct} Strategy vs Buy & Hold"
            if backtest_start_date:
                backtest_title += f" (from {backtest_start_str})"
            backtest_title += f" | Excess: {final_excess:+.1f}%"

            bt_fig.update_layout(
                title=backtest_title,
                height=400,
                paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                font=dict(color=CYBERPUNK_COLORS['text_primary']),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis_title="Portfolio Value ($100 start)"
            )
            bt_fig.update_xaxes(gridcolor=CYBERPUNK_COLORS['bg_light'])
            bt_fig.update_yaxes(gridcolor=CYBERPUNK_COLORS['bg_light'])

            # ── Lag Analysis Chart ──
            lag_fig = go.Figure()
            lag_fig.add_trace(go.Bar(
                x=lags["lag"], y=lags["r"],
                marker_color=[CYBERPUNK_COLORS['neon_green'] if p < 0.05 else CYBERPUNK_COLORS['text_secondary']
                              for p in lags["p_value"]],
                name="Correlation"
            ))
            lag_fig.add_hline(y=0, line_dash="dash", line_color=CYBERPUNK_COLORS['neon_pink'])

            lag_fig.update_layout(
                title="Lag Analysis: Max[Z] → VIX Change (positive lag = GCP leads market)",
                height=350,
                paper_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                plot_bgcolor=CYBERPUNK_COLORS['bg_dark'],
                font=dict(color=CYBERPUNK_COLORS['text_primary']),
                xaxis_title="Lag (days)",
                yaxis_title="Correlation (r)"
            )
            lag_fig.update_xaxes(gridcolor=CYBERPUNK_COLORS['bg_light'])
            lag_fig.update_yaxes(gridcolor=CYBERPUNK_COLORS['bg_light'])

            return summary_stats, correlation_table, ts_fig, corr_fig, bt_fig, lag_fig


def mount_holmberg_dashboard(server, base_path="/experiment-6/"):
    """Mount the Holmberg dashboard on an existing Flask server.

    Args:
        server: Flask server instance
        base_path: URL path prefix for the dashboard

    Returns:
        The mounted Dash app
    """
    app = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_path,
        external_stylesheets=[dbc.themes.CYBORG]
    )
    app.title = "GCP Holmberg Analysis"
    app.layout = create_layout()
    register_callbacks(app)
    return app


# Standalone app for direct execution
app = None

if __name__ == "__main__":
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.title = "GCP Holmberg Analysis"
    app.layout = create_layout()
    register_callbacks(app)

    print("GCP Holmberg Analysis Dashboard")
    print("=" * 50)
    print("Starting server at http://localhost:8052")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=8052)
