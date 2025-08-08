from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html

from ..config import AppConfig, load_config
from ..core.buffers import CircularBuffer
from ..core.predict import MarketTick, Predictor, SensorSample
from ..core.tracker import PredictionTracker
from ..data.gcp_collector import GcpCollector
from ..data.market_collector import MarketCollector


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
        self.app.layout = dbc.Container(
            [
                html.H2("GCP Real-Time Market Predictor"),
                dbc.Row(
                    [
                        dbc.Col(dbc.Button("Start", id="btn-start", color="success"), width="auto"),
                        dbc.Col(dbc.Button("Stop", id="btn-stop", color="danger"), width="auto"),
                        dbc.Col(html.Div(id="status-text"), width="auto"),
                    ],
                    className="my-2",
                ),
                dbc.Row(
                    [
                        dbc.Col(dbc.Card(
                            [dbc.CardHeader("Sensor Buffer"), dbc.CardBody(html.H4(id="sensor-size"))]
                        )),
                        dbc.Col(dbc.Card(
                            [dbc.CardHeader("Market Buffer"), dbc.CardBody(html.H4(id="market-size"))]
                        )),
                        dbc.Col(dbc.Card(
                            [dbc.CardHeader("Current Score"), dbc.CardBody(html.H4(id="current-score"))]
                        )),
                    ]
                ),
                dcc.Interval(id="tick", interval=2000, n_intervals=0),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        html.Label("Cadence (s)", style={"marginRight": "8px"}),
                                        html.Span(
                                            "Run frequency.",
                                            className="text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "8px",
                                        "flexWrap": "wrap",
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Slider(
                                    id="cadence",
                                    min=2,
                                    max=120,
                                    step=1,
                                    value=self.config.runtime.bin_duration_sec,
                                    tooltip={"placement": "bottom"},
                                    marks={2: "2", 10: "10", 30: "30", 60: "60", 120: "120"},
                                ),
                                html.Div(style={"height": "12px"}),
                                html.Div(
                                    [
                                        html.Label("Horizon (s)", style={"marginRight": "8px"}),
                                        html.Span(
                                            "Time to grade predictions.",
                                            className="text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "alignItems": "center",
                                        "gap": "8px",
                                        "flexWrap": "wrap",
                                        "marginBottom": "6px",
                                    },
                                ),
                                dcc.Slider(
                                    id="horizon",
                                    min=10,
                                    max=1800,
                                    step=10,
                                    value=self.config.runtime.horizon_sec,
                                    tooltip={"placement": "bottom"},
                                    marks={60: "60", 300: "300", 600: "600", 1200: "1200", 1800: "1800"},
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Symbols"),
                                dcc.Checklist(
                                    id="symbol-select",
                                    options=[{"label": s, "value": s} for s in self.config.runtime.symbols],
                                    value=list(self.config.runtime.symbols),
                                    inline=True,
                                ),
                            ], width=6,
                        ),
                    ], className="my-2"
                ),
                html.Div(id="config-ack", style={"display": "none"}),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="score-series"), width=6),
                        dbc.Col(dcc.Graph(id="price-series"), width=6),
                    ]
                ),
                html.Hr(),
                html.Div(id="stats"),
            ],
            fluid=True,
        )

    def _callbacks(self) -> None:
        @self.app.callback(Output("status-text", "children"), Input("tick", "n_intervals"))
        def status(_: int) -> str:
            return "Running" if self.running else "Stopped"

        @self.app.callback(Output("sensor-size", "children"), Input("tick", "n_intervals"))
        def sensor_size(_: int) -> str:
            return str(self.sensor_buffer.size())

        @self.app.callback(Output("market-size", "children"), Input("tick", "n_intervals"))
        def market_size(_: int) -> str:
            return str(self.market_buffer.size())

        @self.app.callback(Output("current-score", "children"), Input("tick", "n_intervals"))
        def current_score(_: int) -> str:
            # compute quick score for display
            from ..core.methods import build_registry

            samples = [s.value.values for s in self.sensor_buffer.snapshot()[-100:]]
            reg = build_registry(
                expected_mean=self.config.runtime.method.expected_mean,
                expected_std=self.config.runtime.method.expected_std,
            )
            score = reg[self.config.runtime.method.method].compute(samples)
            return f"{score:.3f}"

        @self.app.callback(
            Output("score-series", "figure"), Input("tick", "n_intervals"),
        )
        def score_series(_: int):
            times: List[float] = []
            scores: List[float] = []
            from ..core.methods import build_registry

            reg = build_registry(
                expected_mean=self.config.runtime.method.expected_mean,
                expected_std=self.config.runtime.method.expected_std,
            )
            method = reg[self.config.runtime.method.method].compute
            snaps = self.sensor_buffer.snapshot()
            for i in range(0, len(snaps), max(1, len(snaps) // 100)):
                window = [s.value.values for s in snaps[max(0, i - 50) : i + 1]]
                score = method(window)
                times.append(snaps[i].timestamp)
                scores.append(score)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=times, y=scores, mode="lines", name="Max[Z]"))
            fig.update_layout(title="Rolling Max[Z]")
            return fig

        @self.app.callback(Output("price-series", "figure"),
                           [Input("tick", "n_intervals"), Input("symbol-select", "value")])
        def price_series(_: int, selected_symbols: list[str]):
            snaps = self.market_buffer.snapshot()[-600:]
            fig = go.Figure()
            symbols = selected_symbols or self.config.runtime.symbols
            for symbol in symbols:
                xs = [s.timestamp for s in snaps if s.value.symbol == symbol]
                ys = [s.value.price for s in snaps if s.value.symbol == symbol]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=symbol))
            fig.update_layout(title="Recent Prices")
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
            children = []
            for symbol, st in s.items():
                children.append(
                    html.Div(
                        f"{symbol}: total {st.total}, correct {st.correct}, accuracy {st.accuracy:.2%}, up {st.up}, down {st.down}"
                    )
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


def build_dash_app(config: AppConfig | None = None) -> Dash:
    cfg = config or load_config()
    d = DashboardApp(cfg)
    return d.app


