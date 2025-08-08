from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html
from dash.dependencies import ALL, State

from ..config import AppConfig, load_config
from ..core.buffers import CircularBuffer
from ..core.predict import MarketTick, Predictor, SensorSample
from ..core.tracker import PredictionTracker
from ..core.maxz import max_abs_z_over_samples
from ..data.gcp_collector import GcpCollector
from ..data.market_collector import MarketCollector

# ───────────────────────────── cyberpunk styling ──────────────────────────────
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


class DashboardApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.sensor_buffer: CircularBuffer[SensorSample] = CircularBuffer(
            capacity=config.runtime.sensor_buffer_size
        )
        self.market_buffer: CircularBuffer[MarketTick] = CircularBuffer(
            capacity=config.runtime.market_buffer_size
        )
        self.tracker = PredictionTracker()
        self.predictor = Predictor(
            config=config,
            sensor_buffer=self.sensor_buffer,
            market_buffer=self.market_buffer,
            on_prediction=self.tracker.record,
        )
        self.gcp = GcpCollector(config=config, buffer=self.sensor_buffer)
        self.market = MarketCollector(config=config, buffer=self.market_buffer)
        self.running = False

        self.app: Dash = dash.Dash(
            __name__, external_stylesheets=[dbc.themes.COSMO]
        )
        self._layout()
        self._callbacks()

    def _layout(self) -> None:
        self.app.layout = html.Div([
            # Header Section
            html.Div([
                html.H1("GCP REAL-TIME MARKET PREDICTOR", 
                       style={
                           "textAlign": "center",
                           "color": CYBERPUNK_COLORS['text_primary'],
                           "fontSize": "2.5rem",
                           "fontWeight": "900",
                           "textShadow": f"0 0 20px {CYBERPUNK_COLORS['neon_purple']}",
                           "marginBottom": "10px",
                           "fontFamily": "'Orbitron', 'Courier New', monospace",
                           "letterSpacing": "3px"
                       }),
                html.P([
                    "NEURAL INTERFACE: Real-time GCP egg data analysis • ",
                    "Max[Z] calculations • Market direction prediction • Performance tracking"
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
                "border": f"2px solid {CYBERPUNK_COLORS['neon_purple']}",
                "boxShadow": f"0 0 30px {CYBERPUNK_COLORS['neon_purple']}40",
                "marginBottom": "30px"
            }),
            
            # Control Panel
            html.Div([
                html.H3("SYSTEM CONTROLS", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_green'],
                           "fontSize": "1.2rem",
                           "fontWeight": "bold",
                           "marginBottom": "20px",
                           "fontFamily": "'Orbitron', monospace"
                       }),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "START SYSTEM",
                            id="btn-start",
                            color="success",
                            size="lg",
                            style={
                                "backgroundColor": CYBERPUNK_COLORS['neon_green'],
                                "borderColor": CYBERPUNK_COLORS['neon_green'],
                                "color": CYBERPUNK_COLORS['bg_dark'],
                                "fontFamily": "'Orbitron', monospace",
                                "fontWeight": "bold"
                            }
                        )
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Button(
                            "STOP SYSTEM",
                            id="btn-stop",
                            color="danger",
                            size="lg",
                            style={
                                "backgroundColor": CYBERPUNK_COLORS['neon_pink'],
                                "borderColor": CYBERPUNK_COLORS['neon_pink'],
                                "color": CYBERPUNK_COLORS['bg_dark'],
                                "fontFamily": "'Orbitron', monospace",
                                "fontWeight": "bold"
                            }
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.Div(id="status-text", children="SYSTEM OFFLINE", 
                                style={
                                    "color": CYBERPUNK_COLORS['neon_yellow'],
                                    "fontFamily": "'Orbitron', monospace",
                                    "fontSize": "1.1rem",
                                    "textAlign": "center",
                                    "padding": "10px"
                                })
                    ], width=6)
                ], className="mb-3")
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "20px",
                "borderRadius": "15px",
                "border": f"2px solid {CYBERPUNK_COLORS['neon_green']}",
                "marginBottom": "30px"
            }),
            
            # Real-time Data Display
            html.Div([
                html.H3("REAL-TIME DATA", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_cyan'],
                           "fontSize": "1.2rem",
                           "fontWeight": "bold",
                           "marginBottom": "20px",
                           "fontFamily": "'Orbitron', monospace"
                       }),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Sensor Buffer", 
                                   style={
                                       "color": CYBERPUNK_COLORS['neon_cyan'],
                                       "fontSize": "1rem",
                                       "fontFamily": "'Orbitron', monospace",
                                       "marginBottom": "5px"
                                   }),
                            html.Div(id="sensor-size", children="0", 
                                    style={
                                        "color": CYBERPUNK_COLORS['text_primary'],
                                        "fontFamily": "'Courier New', monospace",
                                        "fontSize": "1.5rem",
                                        "fontWeight": "bold"
                                    })
                        ], style={
                            "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                            "padding": "15px",
                            "borderRadius": "10px",
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}",
                            "textAlign": "center"
                        })
                    ], width=4),
                    
                    dbc.Col([
                        html.Div([
                            html.H4("Market Buffer", 
                                   style={
                                       "color": CYBERPUNK_COLORS['neon_cyan'],
                                       "fontSize": "1rem",
                                       "fontFamily": "'Orbitron', monospace",
                                       "marginBottom": "5px"
                                   }),
                            html.Div(id="market-size", children="0", 
                                    style={
                                        "color": CYBERPUNK_COLORS['text_primary'],
                                        "fontFamily": "'Courier New', monospace",
                                        "fontSize": "1.5rem",
                                        "fontWeight": "bold"
                                    })
                        ], style={
                            "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                            "padding": "15px",
                            "borderRadius": "10px",
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}",
                            "textAlign": "center"
                        })
                    ], width=4),
                    
                    dbc.Col([
                        html.Div([
                            html.H4("Current Score", 
                                   style={
                                       "color": CYBERPUNK_COLORS['neon_cyan'],
                                       "fontSize": "1rem",
                                       "fontFamily": "'Orbitron', monospace",
                                       "marginBottom": "5px"
                                   }),
                            html.Div(id="current-score", children="0.00", 
                                    style={
                                        "color": CYBERPUNK_COLORS['text_primary'],
                                        "fontFamily": "'Courier New', monospace",
                                        "fontSize": "1.5rem",
                                        "fontWeight": "bold"
                                    })
                        ], style={
                            "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                            "padding": "15px",
                            "borderRadius": "10px",
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}",
                            "textAlign": "center"
                        })
                    ], width=4)
                ])
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "20px",
                "borderRadius": "15px",
                "border": f"2px solid {CYBERPUNK_COLORS['neon_cyan']}",
                "marginBottom": "30px"
            }),
            
            # Configuration Controls
            html.Div([
                html.H3("CONFIGURATION", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_yellow'],
                           "fontSize": "1.2rem",
                           "fontWeight": "bold",
                           "marginBottom": "20px",
                           "fontFamily": "'Orbitron', monospace"
                       }),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Label("Cadence (s)", 
                                     style={
                                         "color": CYBERPUNK_COLORS['text_primary'],
                                         "fontFamily": "'Orbitron', monospace",
                                         "fontSize": "0.9rem",
                                         "marginBottom": "5px"
                                     }),
                            html.Span("Run frequency.", 
                                    style={
                                        "color": CYBERPUNK_COLORS['text_secondary'],
                                        "fontSize": "0.8rem",
                                        "fontFamily": "'Courier New', monospace"
                                    }),
                            dcc.Slider(
                                id="cadence",
                                min=2,
                                max=120,
                                step=1,
                                value=self.config.runtime.bin_duration_sec,
                                tooltip={"placement": "bottom"},
                                marks={2: "2", 10: "10", 30: "30", 60: "60", 120: "120"},
                            )
                        ])
                    ], width=6),
                    
                    dbc.Col([
                        html.Div([
                            html.Label("Horizon (s)", 
                                     style={
                                         "color": CYBERPUNK_COLORS['text_primary'],
                                         "fontFamily": "'Orbitron', monospace",
                                         "fontSize": "0.9rem",
                                         "marginBottom": "5px"
                                     }),
                            html.Span("Time to grade predictions.", 
                                    style={
                                        "color": CYBERPUNK_COLORS['text_secondary'],
                                        "fontSize": "0.8rem",
                                        "fontFamily": "'Courier New', monospace"
                                    }),
                            dcc.Slider(
                                id="horizon",
                                min=10,
                                max=1800,
                                step=10,
                                value=self.config.runtime.horizon_sec,
                                tooltip={"placement": "bottom"},
                                marks={60: "60", 300: "300", 600: "600", 1200: "1200", 1800: "1800"},
                            )
                        ])
                    ], width=6)
                ])
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "20px",
                "borderRadius": "15px",
                "border": f"2px solid {CYBERPUNK_COLORS['neon_yellow']}",
                "marginBottom": "30px"
            }),
            
            # Prediction Performance
            html.Div([
                html.H3("PREDICTION PERFORMANCE", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_purple'],
                           "fontSize": "1.2rem",
                           "fontWeight": "bold",
                           "marginBottom": "20px",
                           "fontFamily": "'Orbitron', monospace"
                       }),
                
                html.Div(id="stats", children=[
                    html.P("No predictions yet", 
                          style={
                              "color": CYBERPUNK_COLORS['text_secondary'],
                              "fontFamily": "'Courier New', monospace",
                              "textAlign": "center"
                          })
                ])
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "20px",
                "borderRadius": "15px",
                "border": f"2px solid {CYBERPUNK_COLORS['neon_purple']}",
                "marginBottom": "30px"
            }),
            
            # Charts Section
            html.Div([
                html.H3("DATA VISUALIZATION", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_green'],
                           "fontSize": "1.2rem",
                           "fontWeight": "bold",
                           "marginBottom": "20px",
                           "fontFamily": "'Orbitron', monospace"
                       }),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Max[Z] Time Series", 
                                   style={
                                       "color": CYBERPUNK_COLORS['neon_green'],
                                       "fontSize": "1rem",
                                       "fontFamily": "'Orbitron', monospace",
                                       "marginBottom": "10px"
                                   }),
                            dcc.Graph(id="score-series", 
                                     style={
                                         "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                                         "borderRadius": "10px",
                                         "height": "400px"
                                     },
                                     config={'displayModeBar': False})
                        ], style={"height": "100%"})
                    ], width=6),
                    
                    dbc.Col([
                        html.Div([
                            html.H4("Price Series", 
                                   style={
                                       "color": CYBERPUNK_COLORS['neon_green'],
                                       "fontSize": "1rem",
                                       "fontFamily": "'Orbitron', monospace",
                                       "marginBottom": "10px"
                                   }),
                            dcc.Graph(id="price-series", 
                                     style={
                                         "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                                         "borderRadius": "10px",
                                         "height": "400px"
                                     },
                                     config={'displayModeBar': False}),
                            html.Div([
                                html.Label("Symbols:", 
                                         style={
                                             "color": CYBERPUNK_COLORS['text_primary'],
                                             "fontFamily": "'Orbitron', monospace",
                                             "fontSize": "0.9rem",
                                             "marginBottom": "5px",
                                             "display": "block"
                                         }),
                                html.Div(id="symbol-info", children=[
                                    # This will be populated by callback with formatted checklist
                                ], style={
                                    "marginBottom": "10px"
                                }),
                                # Add-symbol controls
                                html.Div([
                                    dcc.Input(
                                        id="new-symbol-input",
                                        type="text",
                                        placeholder="Add ticker (e.g., TSLA)",
                                        debounce=True,
                                        style={
                                            "backgroundColor": CYBERPUNK_COLORS['bg_light'],
                                            "color": CYBERPUNK_COLORS['text_primary'],
                                            "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}",
                                            "borderRadius": "6px",
                                            "padding": "6px 10px",
                                            "marginRight": "8px",
                                            "width": "220px"
                                        }
                                    ),
                                    dbc.Button(
                                        "Add",
                                        id="btn-add-symbol",
                                        color="primary",
                                        size="sm",
                                        style={
                                            "backgroundColor": CYBERPUNK_COLORS['neon_cyan'],
                                            "borderColor": CYBERPUNK_COLORS['neon_cyan'],
                                            "color": CYBERPUNK_COLORS['bg_dark'],
                                            "fontFamily": "'Orbitron', monospace",
                                            "fontWeight": "bold"
                                        }
                                    )
                                ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginTop": "6px"}),
                                # Hidden symbol select for compatibility
                                dcc.Checklist(
                                    id="symbol-select",
                                    options=[{"label": symbol, "value": symbol} for symbol in self.config.runtime.symbols],
                                    value=self.config.runtime.symbols,
                                    style={"display": "none"}
                                )
                            ], style={"marginTop": "10px"})
                        ], style={"height": "100%"})
                    ], width=6)
                ], style={"alignItems": "stretch"})
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "20px",
                "borderRadius": "15px",
                "border": f"2px solid {CYBERPUNK_COLORS['neon_green']}",
                "marginBottom": "30px"
            }),
            
            # Hidden elements
            dcc.Interval(id="tick", interval=2000, n_intervals=0),
            html.Div(id="config-ack", style={"display": "none"}),
            # Toast notification
            dbc.Toast(
                id="symbol-toast",
                header="",
                is_open=False,
                dismissable=True,
                duration=4000,
                icon="danger",
                style={
                    "position": "fixed",
                    "top": 20,
                    "right": 20,
                    "minWidth": "280px",
                    "zIndex": 1060
                }
            ),
            
        ], style={
            "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
            "minHeight": "100vh",
            "padding": "20px",
            "fontFamily": "'Courier New', monospace"
        })

    def _callbacks(self) -> None:
        @self.app.callback(Output("status-text", "children"), Input("tick", "n_intervals"))
        def status(_: int) -> str:
            return "SYSTEM ONLINE" if self.running else "SYSTEM OFFLINE"

        @self.app.callback(Output("sensor-size", "children"), Input("tick", "n_intervals"))
        def sensor_size(_: int) -> str:
            return str(self.sensor_buffer.size())

        @self.app.callback(Output("market-size", "children"), Input("tick", "n_intervals"))
        def market_size(_: int) -> str:
            return str(self.market_buffer.size())

        @self.app.callback(Output("current-score", "children"), Input("tick", "n_intervals"))
        def current_score(_: int) -> str:
            # compute quick score for display
            try:
                window_start = datetime.now(timezone.utc).timestamp() - self.config.runtime.method.window_size
                window_end = datetime.now(timezone.utc).timestamp()
                samples = [
                    s.value.values for s in self.sensor_buffer.get_window(window_start, window_end)
                ]
                if samples:
                    score = max_abs_z_over_samples(
                        samples, 
                        self.config.runtime.method.expected_mean,
                        self.config.runtime.method.expected_std
                    )
                    return f"{score:.3f}"
                return "0.000"
            except Exception:
                return "0.000"

        @self.app.callback(
            Output("score-series", "figure"), Input("tick", "n_intervals"),
        )
        def score_series(_: int):
            snaps = self.sensor_buffer.snapshot()[-100:]
            if not snaps:
                fig = go.Figure()
                fig.add_annotation(
                    text="No data available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(color=CYBERPUNK_COLORS['text_secondary'])
                )
                fig.update_layout(
                    plot_bgcolor=CYBERPUNK_COLORS['bg_medium'],
                    paper_bgcolor=CYBERPUNK_COLORS['bg_medium'],
                    font=dict(color=CYBERPUNK_COLORS['text_primary'])
                )
                return fig
            
            times = []
            scores = []
            for i in range(0, len(snaps), max(1, len(snaps) // 20)):
                if i < len(snaps):
                    # Calculate Max[Z] for this sample
                    sample_values = snaps[i].value.values
                    if sample_values:
                        score = max_abs_z_over_samples(
                            [sample_values], 
                            self.config.runtime.method.expected_mean,
                            self.config.runtime.method.expected_std
                        )
                        times.append(snaps[i].timestamp)
                        scores.append(score)
            
            fig = go.Figure()
            if times and scores:
                fig.add_trace(go.Scatter(
                    x=times, 
                    y=scores, 
                    mode="lines", 
                    name="Max[Z]",
                    line=dict(color=CYBERPUNK_COLORS['neon_cyan'], width=2)
                ))
            fig.update_layout(
                title=dict(
                    text="Rolling Max[Z]",
                    font=dict(color=CYBERPUNK_COLORS['text_primary'])
                ),
                plot_bgcolor=CYBERPUNK_COLORS['bg_medium'],
                paper_bgcolor=CYBERPUNK_COLORS['bg_medium'],
                font=dict(color=CYBERPUNK_COLORS['text_primary']),
                xaxis=dict(
                    gridcolor=CYBERPUNK_COLORS['bg_light'],
                    title=dict(text="Time", font=dict(color=CYBERPUNK_COLORS['text_primary'])),
                    tickfont=dict(color=CYBERPUNK_COLORS['text_primary'])
                ),
                yaxis=dict(
                    gridcolor=CYBERPUNK_COLORS['bg_light'],
                    title=dict(text="Max[Z]", font=dict(color=CYBERPUNK_COLORS['text_primary'])),
                    tickfont=dict(color=CYBERPUNK_COLORS['text_primary'])
                ),
                legend=dict(font=dict(color=CYBERPUNK_COLORS['text_primary'])),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return fig

        @self.app.callback(Output("price-series", "figure"),
                           [Input("tick", "n_intervals"), Input("symbol-select", "value")])
        def price_series(_: int, selected_symbols: list[str]):
            snaps = self.market_buffer.snapshot()[-600:]
            fig = go.Figure()
            symbols = selected_symbols or self.config.runtime.symbols
            
            colors = [CYBERPUNK_COLORS['neon_pink'], CYBERPUNK_COLORS['neon_cyan'], 
                     CYBERPUNK_COLORS['neon_green'], CYBERPUNK_COLORS['neon_yellow'], 
                     CYBERPUNK_COLORS['neon_purple']]
            
            for i, symbol in enumerate(symbols):
                xs = [s.timestamp for s in snaps if s.value.symbol == symbol]
                ys = [s.value.price for s in snaps if s.value.symbol == symbol]
                if xs and ys:
                    fig.add_trace(go.Scatter(
                        x=xs, 
                        y=ys, 
                        mode="lines", 
                        name=symbol,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            
            fig.update_layout(
                title=dict(
                    text="Recent Prices",
                    font=dict(color=CYBERPUNK_COLORS['text_primary'])
                ),
                plot_bgcolor=CYBERPUNK_COLORS['bg_medium'],
                paper_bgcolor=CYBERPUNK_COLORS['bg_medium'],
                font=dict(color=CYBERPUNK_COLORS['text_primary']),
                xaxis=dict(
                    gridcolor=CYBERPUNK_COLORS['bg_light'],
                    title=dict(text="Time", font=dict(color=CYBERPUNK_COLORS['text_primary'])),
                    tickfont=dict(color=CYBERPUNK_COLORS['text_primary'])
                ),
                yaxis=dict(
                    gridcolor=CYBERPUNK_COLORS['bg_light'],
                    title=dict(text="Price", font=dict(color=CYBERPUNK_COLORS['text_primary'])),
                    tickfont=dict(color=CYBERPUNK_COLORS['text_primary'])
                ),
                legend=dict(font=dict(color=CYBERPUNK_COLORS['text_primary'])),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            return fig

        @self.app.callback(Output("stats", "children"), Input("tick", "n_intervals"))
        def stats(_: int):
            # Try resolve any matured predictions using market prices
            snaps = self.market_buffer.snapshot()
            latest_price: Dict[str, float] = {}
            for t in snaps:
                latest_price[t.value.symbol] = t.value.price
            now_ts = datetime.now(timezone.utc).timestamp()
            # Resolve by finding first price at/after prediction timestamp
            for p in self.tracker.recent(500):
                if p.resolved:
                    continue
                if now_ts - p.timestamp < p.horizon_sec:
                    continue
                # find earliest tick >= p.timestamp for symbol; if missing, use nearest prior
                price_then = None
                for t in snaps:
                    if t.value.symbol == p.symbol and t.timestamp >= p.timestamp:
                        price_then = t.value.price
                        break
                if price_then is None:
                    for t in reversed(snaps):
                        if t.value.symbol == p.symbol and t.timestamp <= p.timestamp:
                            price_then = t.value.price
                            break
                price_now = latest_price.get(p.symbol)
                if price_then is not None and price_now is not None:
                    self.tracker.try_resolve(p.symbol, price_then, price_now, now_ts)

            s = self.tracker.stats()
            if not s:
                return html.P("No predictions yet", 
                            style={
                                "color": CYBERPUNK_COLORS['text_secondary'],
                                "fontFamily": "'Courier New', monospace",
                                "textAlign": "center"
                            })
            
            children = []
            for symbol, st in s.items():
                accuracy_color = CYBERPUNK_COLORS['neon_green'] if st.accuracy > 0.6 else CYBERPUNK_COLORS['neon_pink']
                children.append(
                    html.Div([
                        html.Strong(f"{symbol}: ", style={"color": CYBERPUNK_COLORS['neon_cyan']}),
                        f"total {st.total}, correct {st.correct}, accuracy {st.accuracy:.2%}, up {st.up}, down {st.down}"
                    ], style={
                        "color": accuracy_color,
                        "fontFamily": "'Courier New', monospace",
                        "marginBottom": "5px",
                        "padding": "5px",
                        "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                        "borderRadius": "5px",
                        "border": f"1px solid {accuracy_color}"
                    })
                )
            return children

        @self.app.callback(Output("btn-start", "n_clicks"), Input("btn-start", "n_clicks"))
        def on_start(n: int | None) -> int | None:
            if not n:
                return n
            if not self.running:
                self.gcp.start()
                self.market.start()
                self.predictor.start()
                self.running = True
            return n

        @self.app.callback(Output("btn-stop", "n_clicks"), Input("btn-stop", "n_clicks"))
        def on_stop(n: int | None) -> int | None:
            if not n:
                return n
            if self.running:
                self.predictor.stop()
                self.gcp.stop()
                self.market.stop()
                self.running = False
            return n

        @self.app.callback(Output("config-ack", "children"),
                           [Input("cadence", "value"), Input("horizon", "value")])
        def update_config(cadence: int, horizon: int):
            # Apply updated cadence and horizon to runtime config; predictor reads these live
            try:
                self.config.runtime.bin_duration_sec = int(cadence)
                self.config.runtime.horizon_sec = int(horizon)
            except Exception:
                pass
            return "ok"

        @self.app.callback(Output("symbol-info", "children"), Input("tick", "n_intervals"))
        def update_symbol_info(_: int):
            """Update symbol information display."""
            try:
                # Simple approach - just use the symbols from config
                symbols = self.config.runtime.symbols
                
                symbol_divs = []
                for symbol in symbols:
                    # Create checkbox for this symbol
                    checkbox = dcc.Checklist(
                        id={"type": "symbol-checkbox", "symbol": symbol},
                        options=[{"label": "", "value": symbol}],
                        value=[symbol],
                        style={"marginRight": "10px"}
                    )
                    
                    # Dynamic exchange info
                    info = self.market.get_exchange_info(symbol)
                    exchange = info.exchange if info else "NYSE"
                    is_open = info.is_open if info is not None else False
                    status_text = "OPEN" if is_open else "CLOSED"
                    status_color = CYBERPUNK_COLORS["neon_green"] if is_open else CYBERPUNK_COLORS["neon_pink"]
                    trading_hours = info.trading_hours if info else "9:30 AM - 4:00 PM ET"
                    countdown = self._get_countdown_text(info) if info else ""

                    symbol_div = html.Div([
                        checkbox,
                        html.Span(f"{symbol} ", style={"color": CYBERPUNK_COLORS["neon_cyan"], "fontWeight": "bold"}),
                        html.Span(f"({exchange}) ", style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(status_text, style={"color": status_color, "fontWeight": "bold"}),
                        html.Span(" • ", style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(trading_hours, style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(" • ", style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(countdown, style={"color": CYBERPUNK_COLORS["neon_yellow"], "fontWeight": "bold"}),
                    ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"})
                    
                    symbol_divs.append(symbol_div)
                
                return symbol_divs
                
            except Exception as e:
                return []

        @self.app.callback(
            Output("symbol-select", "options"),
            Output("symbol-select", "value"),
            Input({"type": "symbol-checkbox", "symbol": ALL}, "value"),
            Input("tick", "n_intervals"),
        )
        def sync_symbol_select(values_lists: list[list[str]] | None, _: int):
            # Flatten selected values from all per-symbol checklists
            try:
                symbols_all = list(self.config.runtime.symbols)
                options = [{"label": s, "value": s} for s in symbols_all]
                selected: list[str] = []
                if values_lists:
                    for v in values_lists:
                        if v:
                            selected.extend(v)
                # Default to all if nothing explicitly selected
                if not selected:
                    selected = symbols_all
                # Deduplicate while preserving order
                seen = set()
                selected_unique = [s for s in selected if not (s in seen or seen.add(s))]
                return options, selected_unique
            except Exception:
                syms = list(self.config.runtime.symbols)
                return ([{"label": s, "value": s} for s in syms], syms)

        @self.app.callback(
            Output("new-symbol-input", "value"),
            Output("symbol-toast", "is_open"),
            Output("symbol-toast", "header"),
            Output("symbol-toast", "children"),
            Output("symbol-toast", "icon"),
            Input("btn-add-symbol", "n_clicks"),
            Input("new-symbol-input", "n_submit"),
            State("new-symbol-input", "value"),
            prevent_initial_call=True,
        )
        def on_add_symbol(n_clicks: int | None, n_submit: int | None, value: str | None):
            try:
                sym_raw = (value or "").strip()
                if not sym_raw:
                    return "", True, "Invalid ticker", "Please enter a ticker symbol.", "danger"
                symbol = sym_raw.upper()
                if symbol in self.config.runtime.symbols:
                    # Nothing to do; silently accept
                    return "", False, "", "", "danger"
                ok = False
                try:
                    ok = self.market.add_symbol(symbol)
                except Exception:
                    ok = False
                if ok:
                    # Added; clear input, no toast
                    return "", False, "", "", "success"
                # Invalid symbol
                return symbol, True, "Ticker not found", f"'{symbol}' is not a valid ticker.", "danger"
            except Exception:
                return value or "", True, "Error", "Unexpected error adding ticker.", "danger"

    def _get_countdown_text(self, info) -> str:
        """Get countdown text for next open/close."""
        try:
            now = datetime.now(timezone.utc)
            
            # Convert to ET (Eastern Time)
            # ET is UTC-5 (EST) or UTC-4 (EDT) - using EDT for simplicity
            # In production, this should use pytz for proper timezone handling
            et_offset = -4  # EDT offset (UTC-4)
            et_hour = now.hour + et_offset
            if et_hour < 0:
                et_hour += 24
            elif et_hour >= 24:
                et_hour -= 24
            et_minute = now.minute
            current_time = et_hour * 60 + et_minute
            
            # Market hours in minutes
            market_start = 9 * 60 + 30  # 9:30 AM
            market_end = 16 * 60  # 4:00 PM
            
            if info.is_open:
                # Market is open - countdown to close
                if current_time < market_end:
                    minutes_to_close = market_end - current_time
                    hours = minutes_to_close // 60
                    minutes = minutes_to_close % 60
                    if hours > 0:
                        return f"Closes in {hours}h {minutes}m"
                    else:
                        return f"Closes in {minutes}m"
                else:
                    return "Closes soon"
            else:
                # Market is closed - countdown to next open
                weekday = now.weekday()  # Monday = 0, Sunday = 6
                
                # Calculate days until next market day
                if weekday < 5:  # Weekday
                    if current_time < market_start:
                        # Today, before market opens
                        minutes_to_open = market_start - current_time
                        hours = minutes_to_open // 60
                        minutes = minutes_to_open % 60
                        if hours > 0:
                            return f"Opens in {hours}h {minutes}m"
                        else:
                            return f"Opens in {minutes}m"
                    else:
                        # Today, after market closes - next business day
                        days_to_next = 1
                        while (weekday + days_to_next) % 7 >= 5:  # Skip weekends
                            days_to_next += 1
                        next_weekday = (weekday + days_to_next) % 7
                        if days_to_next == 1:
                            return "Opens tomorrow"
                        else:
                            # Calculate exact time until next open
                            target_date = now.replace(hour=9, minute=30, second=0, microsecond=0)
                            target_date = target_date.replace(day=target_date.day + days_to_next)
                            # Adjust for weekends
                            while target_date.weekday() >= 5:
                                target_date = target_date.replace(day=target_date.day + 1)
                            
                            time_diff = target_date - now
                            total_seconds = int(time_diff.total_seconds())
                            days = total_seconds // 86400
                            hours = (total_seconds % 86400) // 3600
                            minutes = (total_seconds % 3600) // 60
                            
                            if days > 0:
                                return f"Opens in {days}d {hours}h {minutes}m"
                            elif hours > 0:
                                return f"Opens in {hours}h {minutes}m"
                            else:
                                return f"Opens in {minutes}m"
                else:
                    # Weekend
                    if weekday == 5:  # Saturday
                        # Calculate time until Monday 9:30 AM
                        target_date = now.replace(hour=9, minute=30, second=0, microsecond=0)
                        target_date = target_date.replace(day=target_date.day + 2)  # Monday
                        
                        time_diff = target_date - now
                        total_seconds = int(time_diff.total_seconds())
                        days = total_seconds // 86400
                        hours = (total_seconds % 86400) // 3600
                        minutes = (total_seconds % 3600) // 60
                        
                        if days > 0:
                            return f"Opens Monday in {days}d {hours}h {minutes}m"
                        elif hours > 0:
                            return f"Opens Monday in {hours}h {minutes}m"
                        else:
                            return f"Opens Monday in {minutes}m"
                    else:  # Sunday
                        # Calculate time until Monday 9:30 AM
                        target_date = now.replace(hour=9, minute=30, second=0, microsecond=0)
                        target_date = target_date.replace(day=target_date.day + 1)  # Monday
                        
                        time_diff = target_date - now
                        total_seconds = int(time_diff.total_seconds())
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        
                        if hours > 0:
                            return f"Opens tomorrow in {hours}h {minutes}m"
                        else:
                            return f"Opens tomorrow in {minutes}m"
        except Exception:
            return ""


def build_dash_app(config: AppConfig | None = None) -> Dash:
    cfg = config or load_config()
    d = DashboardApp(cfg)
    return d.app


