from __future__ import annotations

import threading
from datetime import datetime, timezone
import math
from typing import Dict, List
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Dash, Input, Output, dcc, html
from dash.dependencies import ALL, State

from ..config import AppConfig, load_config
from ..core.buffers import CircularBuffer
from ..core.predict import MarketTick, Predictor, SensorSample
from ..core.tracker import PredictionTracker
from ..core.maxz import max_abs_z_over_samples, max_abs_z
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
    def __init__(self, config: AppConfig, app: Dash | None = None) -> None:
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
        self.start_time = None  # type: datetime | None
        self.last_run_elapsed_seconds = 0  # seconds since last start; freezes on stop

        assets_dir = str((Path(__file__).parent / "assets").resolve())
        if app is None:
            self.app: Dash = dash.Dash(
                __name__, external_stylesheets=[dbc.themes.COSMO], assets_folder=assets_dir
            )
        else:
            self.app = app
        self._layout()
        self._callbacks()

    def _layout(self) -> None:
        self.app.layout = html.Div([
            # Inline CSS via dcc (html.Style is not available in dash.html)
            dcc.Markdown(
                children="""
<style>
/* Darker disabled state */
#btn-start:disabled, #btn-stop:disabled {
  background-color: #3a3a3f !important;
  border-color: #3a3a3f !important;
  color: #888888 !important;
  opacity: 1 !important;
  box-shadow: none !important;
  transform: none !important;
}

/* Smooth transitions */
#btn-start, #btn-stop {
  transition: box-shadow 120ms ease, transform 120ms ease, filter 120ms ease;
}

/* Hover effects (enabled only) */
#btn-start:not(:disabled):hover {
  box-shadow: 0 0 10px rgba(0, 255, 136, 0.7), 0 0 18px rgba(0, 255, 136, 0.35) !important; /* #00ff88 */
  transform: translateY(-1px);
}
#btn-stop:not(:disabled):hover {
  box-shadow: 0 0 10px rgba(255, 0, 110, 0.7), 0 0 18px rgba(255, 0, 110, 0.35) !important; /* #ff006e */
  transform: translateY(-1px);
}

/* Pressed (active) effects (enabled only) */
#btn-start:not(:disabled):active,
#btn-start:not(:disabled).active,
#btn-start:not(:disabled):focus:active {
  transform: translateY(2px) !important;
  box-shadow: inset 0 3px 8px rgba(0,0,0,0.55) !important;
  filter: brightness(0.92) !important;
}
#btn-stop:not(:disabled):active,
#btn-stop:not(:disabled).active,
#btn-stop:not(:disabled):focus:active {
  transform: translateY(2px) !important;
  box-shadow: inset 0 3px 8px rgba(0,0,0,0.55) !important;
  filter: brightness(0.92) !important;
}
/* Pointer cursor */
#btn-start:not(:disabled), #btn-stop:not(:disabled) {
  cursor: pointer;
}
</style>
""",
                dangerously_allow_html=True,
            ),
            dcc.Markdown(
                children="""
<script>
(function(){
  function attachButtonEffects(){
    var start = document.getElementById('btn-start');
    var stop = document.getElementById('btn-stop');
    if(!start || !stop) return false;
    function addEffects(el, glowColor){
      if(el._effectsBound) return; // avoid duplicate bindings
      el.addEventListener('mouseenter', function(){ if(!el.disabled){ el.style.boxShadow = '0 0 10px '+glowColor+', 0 0 18px '+glowColor.replace('0.7','0.35'); el.style.transform = 'translateY(-1px)'; }});
      el.addEventListener('mouseleave', function(){ el.style.boxShadow=''; el.style.transform=''; });
      el.addEventListener('mousedown', function(){ if(!el.disabled){ el.style.transform = 'translateY(2px)'; el.style.boxShadow = 'inset 0 3px 8px rgba(0,0,0,0.55)'; el.style.filter = 'brightness(0.92)'; }});
      el.addEventListener('mouseup', function(){ if(!el.disabled){ el.style.transform = 'translateY(-1px)'; el.style.boxShadow = ''; el.style.filter = ''; }});
      el._effectsBound = true;
    }
    addEffects(start, 'rgba(0, 255, 136, 0.7)');
    addEffects(stop, 'rgba(255, 0, 110, 0.7)');
    return true;
  }
  var tries = 0; var timer = setInterval(function(){
    tries += 1;
    if(attachButtonEffects() || tries > 50){ clearInterval(timer); }
  }, 100);
})();
</script>
""",
                dangerously_allow_html=True,
            ),
            # Header Section
            html.Div([
                html.H1("GCP MARKET PREDICTOR - UNDER DEVELOPMENT", 
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
                    "NEURAL INTERFACE: GCP egg data analysis • ",
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
                                "fontWeight": "bold",
                                "outline": "none",
                                "boxShadow": "0 0 0 rgba(0,0,0,0)",
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
                                "fontWeight": "bold",
                                "outline": "none",
                                "boxShadow": "0 0 0 rgba(0,0,0,0)",
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
                        ,
                        html.Div(id="uptime", children="UPTIME: 00:00:00",
                                 style={
                                     "color": CYBERPUNK_COLORS['neon_cyan'],
                                     "fontFamily": "'Courier New', monospace",
                                     "fontSize": "0.95rem",
                                     "textAlign": "center",
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

            # Backtest / Live Mode Controls
            html.Div([
                html.H3("MODE & BACKTEST", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_yellow'],
                           "fontSize": "1.2rem",
                           "fontWeight": "bold",
                           "marginBottom": "20px",
                           "fontFamily": "'Orbitron', monospace"
                       }),
                dbc.Row([
                    dbc.Col([
                        html.Label("Mode", style={"color": CYBERPUNK_COLORS['text_primary'], "fontFamily": "'Orbitron', monospace"}),
                        dcc.RadioItems(
                            id="mode-select",
                            options=[{"label": "Live", "value": "live"}, {"label": "Backtest", "value": "backtest"}],
                            value="live",
                            labelStyle={"display": "inline-block", "marginRight": "15px"},
                            inputStyle={"marginRight": "6px"},
                            style={"color": CYBERPUNK_COLORS['text_primary']},
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Start (UTC ISO8601)", style={"color": CYBERPUNK_COLORS['text_primary'], "fontFamily": "'Orbitron', monospace"}),
                        dcc.Input(id="bt-start", type="text", placeholder="2024-06-01T14:30:00Z", debounce=True,
                                  style={"backgroundColor": CYBERPUNK_COLORS['bg_light'], "color": CYBERPUNK_COLORS['text_primary'], "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}", "borderRadius": "6px", "padding": "6px 10px", "width": "100%"}),
                    ], width=3),
                    dbc.Col([
                        html.Label("End (UTC ISO8601)", style={"color": CYBERPUNK_COLORS['text_primary'], "fontFamily": "'Orbitron', monospace"}),
                        dcc.Input(id="bt-end", type="text", placeholder="2024-06-01T16:00:00Z", debounce=True,
                                  style={"backgroundColor": CYBERPUNK_COLORS['bg_light'], "color": CYBERPUNK_COLORS['text_primary'], "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}", "borderRadius": "6px", "padding": "6px 10px", "width": "100%"}),
                    ], width=3),
                    dbc.Col([
                        html.Label("Speed (x)", style={"color": CYBERPUNK_COLORS['text_primary'], "fontFamily": "'Orbitron', monospace"}),
                        dcc.Slider(id="bt-speed", min=10, max=3600, step=10, value=600, marks={10: "10x", 60: "60x", 600: "600x", 1200: "1200x", 3600: "3600x"}),
                    ], width=3),
                ])
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "20px",
                "borderRadius": "15px",
                "border": f"2px solid {CYBERPUNK_COLORS['neon_yellow']}",
                "marginBottom": "30px"
            }),
            
            
            # Data Summary
            html.Div([
                html.H3("STREAM SUMMARY", 
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

            # ENTROPY SOURCE - GCP EGGS (moved to very bottom)
            html.Div([
                html.H3("ENTROPY SOURCE - GCP EGGS", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_yellow'],
                           "fontSize": "1.2rem",
                           "fontWeight": "bold",
                           "marginBottom": "12px",
                           "fontFamily": "'Orbitron', monospace"
                       }),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            "Pause Stream",
                            id="btn-pause-gcp",
                            color="secondary",
                            size="sm",
                            style={
                                "backgroundColor": CYBERPUNK_COLORS['bg_light'],
                                "borderColor": CYBERPUNK_COLORS['neon_purple'],
                                "color": CYBERPUNK_COLORS['text_primary'],
                                "fontFamily": "'Orbitron', monospace",
                                "fontWeight": "bold"
                            }
                        ),
                        dbc.Button(
                            "Resume",
                            id="btn-resume-gcp",
                            color="secondary",
                            size="sm",
                            style={
                                "marginLeft": "8px",
                                "backgroundColor": CYBERPUNK_COLORS['bg_light'],
                                "borderColor": CYBERPUNK_COLORS['neon_purple'],
                                "color": CYBERPUNK_COLORS['text_primary'],
                                "fontFamily": "'Orbitron', monospace",
                                "fontWeight": "bold"
                            }
                        )
                    ], width=12)
                ], className="mb-2"),
                html.Div(id="gcp-log", style={
                    "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                    "border": f"1px solid {CYBERPUNK_COLORS['neon_purple']}",
                    "borderRadius": "8px",
                    "padding": "10px",
                    "fontFamily": "'Courier New', monospace",
                    "color": CYBERPUNK_COLORS['text_primary']
                })
            ], style={
                "background": f"linear-gradient(135deg, {CYBERPUNK_COLORS['bg_medium']} 0%, {CYBERPUNK_COLORS['bg_light']} 100%)",
                "padding": "20px",
                "borderRadius": "15px",
                "border": f"2px solid {CYBERPUNK_COLORS['neon_yellow']}",
                "marginBottom": "30px"
            }),
            
            # Hidden elements
            dcc.Interval(id="tick", interval=1000, n_intervals=0),
            html.Div(id="config-ack", style={"display": "none"}),
            # Browser timezone store (filled by clientside callback)
            dcc.Store(id="client-tz", storage_type="memory"),
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

        @self.app.callback(
            Output("btn-start", "disabled"), Output("btn-stop", "disabled"), Input("tick", "n_intervals")
        )
        def disable_buttons(_: int) -> tuple[bool, bool]:
            # When running, disable start; when stopped, disable stop
            return self.running, (not self.running)

        @self.app.callback(Output("uptime", "children"), Input("tick", "n_intervals"))
        def uptime(_: int) -> str:
            try:
                if self.running and self.start_time is not None:
                    delta = datetime.now(timezone.utc) - self.start_time
                    total = int(delta.total_seconds())
                    self.last_run_elapsed_seconds = total
                total = int(self.last_run_elapsed_seconds)
                h, rem = divmod(total, 3600)
                m, s = divmod(rem, 60)
                return f"UPTIME: {h:02d}:{m:02d}:{s:02d}"
            except Exception:
                return "UPTIME: 00:00:00"

        # Capture browser timezone and local datetime on the client
        self.app.clientside_callback(
            """
            function(n) {
                try {
                    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || 'Local';
                    const d = new Date();
                    // Offset in minutes east of UTC (JS returns minutes behind UTC)
                    const offsetMinutes = -d.getTimezoneOffset();
                    const dateStr = d.toLocaleString([], {year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit'});
                    return { tzName: tz, offsetMinutes: offsetMinutes, localDate: dateStr };
                } catch (e) {
                    return { tzName: 'Local', offsetMinutes: 0, localDate: '' };
                }
            }
            """,
            Output("client-tz", "data"),
            Input("tick", "n_intervals"),
        )

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

        @self.app.callback(Output("btn-start", "n_clicks"),
                           Input("btn-start", "n_clicks"),
                           State("mode-select", "value"),
                           State("bt-start", "value"),
                           State("bt-end", "value"),
                           State("bt-speed", "value"))
        def on_start(n: int | None, mode: str | None, bt_start: str | None, bt_end: str | None, bt_speed: int | None) -> int | None:
            if not n:
                return n
            if not self.running:
                # Parse backtest params
                def parse_iso8601(val: str | None) -> float | None:
                    if not val:
                        return None
                    try:
                        s = val.strip()
                        # Support Z suffix
                        s = s.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(s)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        return dt.timestamp()
                    except Exception:
                        return None

                if (mode or "live") == "backtest":
                    st_ts = parse_iso8601(bt_start)
                    en_ts = parse_iso8601(bt_end)
                    sp = float(bt_speed or 600)
                    # Start collectors in backtest mode
                    self.gcp.start(start_ts=st_ts, end_ts=en_ts, realtime_interval_sec=1)
                    self.market.start(start_ts=st_ts, end_ts=en_ts, speed=sp)
                else:
                    # Live mode
                    self.gcp.start()
                    self.market.start()
                self.predictor.start()
                self.running = True
                self.start_time = datetime.now(timezone.utc)
                # reset timer on start (do not reset on stop)
                self.last_run_elapsed_seconds = 0
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
                # freeze timer: keep last_run_elapsed_seconds; do not reset here
                self.start_time = None
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

        # ───────────────────────────── GCP log rendering ─────────────────────────────
        self.gcp_display_paused: bool = False

        @self.app.callback(
            Output("gcp-log", "children"),
            Input("tick", "n_intervals"),
            Input("btn-pause-gcp", "n_clicks"),
            Input("btn-resume-gcp", "n_clicks"),
        )
        def update_gcp_log(_: int, pause_clicks: int | None, resume_clicks: int | None):
            try:
                # Toggle pause/resume state
                ctx = dash.callback_context
                if ctx.triggered:
                    trig = ctx.triggered[0]['prop_id'].split('.')[0]
                    if trig == 'btn-pause-gcp':
                        self.gcp_display_paused = True
                    elif trig == 'btn-resume-gcp':
                        self.gcp_display_paused = False

                if self.gcp_display_paused:
                    return dash.no_update

                # Get last N samples from sensor buffer
                snaps = self.sensor_buffer.snapshot()[-20:]
                if not snaps:
                    return html.Div("No GCP samples yet")

                # Determine egg headers from collector active set (if available)
                egg_headers = self.gcp.get_active_eggs() if hasattr(self.gcp, 'get_active_eggs') else []

                # Compose table headers: Time | eggs... | Variance | StdDev | Max[Z]
                headers = ["Time"]
                if egg_headers:
                    headers.extend(egg_headers)
                else:
                    # fallback generic names based on first row length
                    if snaps:
                        ncols = len(snaps[-1].value.values)
                        headers.extend([f"egg_{i+1}" for i in range(ncols)])
                headers.extend(["Variance", "StdDev", "Max[Z]"])

                # Build most-recent-first rows
                body_rows = []
                for item in reversed(snaps):
                    ts = item.timestamp
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
                    # Convert per-egg values robustly to floats; non-parsable -> NaN placeholder
                    values: list[float] = []
                    for v in item.value.values:
                        try:
                            values.append(float(v))
                        except Exception:
                            values.append(float("nan"))
                    # Stats across eggs for this row
                    try:
                        finite_vals = [v for v in values if not math.isnan(v)]
                        if finite_vals:
                            mean = sum(finite_vals) / len(finite_vals)
                            var = sum((v - mean) ** 2 for v in finite_vals) / len(finite_vals)
                            std = var ** 0.5
                            maxz = max_abs_z(finite_vals, self.config.runtime.method.expected_mean, self.config.runtime.method.expected_std)
                        else:
                            var = 0.0
                            std = 0.0
                            maxz = 0.0
                    except Exception:
                        var = 0.0
                        std = 0.0
                        maxz = 0.0
                    # Cell styles based on z-score/p-value
                    def hex_to_rgb(h: str) -> tuple[int, int, int]:
                        h = h.lstrip('#')
                        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

                    exp_mean = self.config.runtime.method.expected_mean
                    exp_std = self.config.runtime.method.expected_std or 1.0

                    def style_for_value(val: float) -> dict:
                        if val != val:  # NaN
                            return {"padding": "6px", "textAlign": "right"}
                        z = abs((val - exp_mean) / exp_std)
                        # Two-sided p-value from z
                        try:
                            phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                            p_two = 2.0 * (1.0 - phi)
                        except Exception:
                            p_two = 1.0
                        # Choose base color by significance tier
                        if z >= 3.0:
                            base = CYBERPUNK_COLORS['neon_pink']
                        elif z >= 2.0:
                            base = CYBERPUNK_COLORS['neon_yellow']
                        elif z >= 1.0:
                            base = CYBERPUNK_COLORS['neon_cyan']
                        else:
                            return {"padding": "6px", "textAlign": "right"}
                        r, g, b = hex_to_rgb(base)
                        # Intensity scales with surprise (lower p -> higher alpha)
                        alpha = max(0.15, min(0.85, 0.2 + 0.6 * (1.0 - min(1.0, p_two * 5.0))))
                        return {
                            "padding": "6px",
                            "textAlign": "right",
                            "backgroundColor": f"rgba({r},{g},{b},{alpha})",
                            "border": f"1px solid rgba({r},{g},{b}, {min(0.9, alpha+0.1)})",
                        }

                    # Build TDs
                    row_tds = [html.Td(dt, style={"padding": "6px"})]
                    eggs_count = len(headers) - 1 - 3  # Time + 3 stats
                    for idx in range(eggs_count):
                        cell_val = ""
                        try:
                            cell_val = f"{values[idx]:.0f}"
                        except Exception:
                            cell_val = ""
                        row_tds.append(html.Td(cell_val, style=style_for_value(values[idx] if idx < len(values) else float('nan'))))
                    # Stats columns (right-aligned)
                    row_tds.append(html.Td(f"{var:.2f}", style={"padding": "6px", "textAlign": "right"}))
                    row_tds.append(html.Td(f"{std:.2f}", style={"padding": "6px", "textAlign": "right"}))
                    row_tds.append(html.Td(f"{maxz:.2f}", style={"padding": "6px", "textAlign": "right"}))
                    body_rows.append(row_tds)

                # Render as plain HTML table (fits content, no scroll)
                table = html.Table([
                    html.Thead(html.Tr([html.Th(h, style={"padding": "6px", "textAlign": "left"}) for h in headers])),
                    html.Tbody([html.Tr(r) for r in body_rows]),
                ], style={
                    "width": "100%",
                    "borderCollapse": "collapse",
                    "tableLayout": "auto",
                })
                return table
            except Exception:
                return ""

        @self.app.callback(
            Output("symbol-info", "children"),
            Input("tick", "n_intervals"),
            State("client-tz", "data"),
            State("symbol-select", "value"),
        )
        def update_symbol_info(_: int, client_tz: dict | None, selected_symbols: list[str] | None):
            """Update symbol information display."""
            try:
                # Simple approach - just use the symbols from config
                symbols = self.config.runtime.symbols
                selected_set = set(selected_symbols or symbols)
                
                symbol_divs = []
                for symbol in symbols:
                    # Create checkbox for this symbol
                    checkbox = dcc.Checklist(
                        id={"type": "symbol-checkbox", "symbol": symbol},
                        options=[{"label": "", "value": symbol}],
                        value=[symbol] if symbol in selected_set else [],
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
                    # Local timezone conversion for trading hours using provided client offset
                    local_label = ""
                    try:
                        tz_name = (client_tz or {}).get("tzName") or "Local"
                        offset_min = int((client_tz or {}).get("offsetMinutes") or 0)
                        # Fixed ET hours for now
                        et_start_min = 9 * 60 + 30
                        et_end_min = 16 * 60
                        # Convert ET (UTC-4) to UTC minutes
                        utc_start_min = et_start_min + 4 * 60
                        utc_end_min = et_end_min + 4 * 60
                        def to_local_str(utc_mins: int) -> str:
                            m = (utc_mins + offset_min) % (24 * 60)
                            hh = m // 60
                            mm = m % 60
                            h12 = ((hh + 11) % 12) + 1
                            ampm = "PM" if hh >= 12 else "AM"
                            return f"{h12}:{mm:02d} {ampm}"
                        local_start = to_local_str(utc_start_min)
                        local_end = to_local_str(utc_end_min)
                        local_label = f"({tz_name} {local_start} - {local_end})"
                    except Exception:
                        local_label = ""

                    symbol_div = html.Div([
                        checkbox,
                        html.Span(f"{symbol} ", style={"color": CYBERPUNK_COLORS["neon_cyan"], "fontWeight": "bold"}),
                        html.Span(f"({exchange}) ", style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(status_text, style={"color": status_color, "fontWeight": "bold"}),
                        html.Span(" • ", style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(trading_hours, style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(" ", style={"color": CYBERPUNK_COLORS["text_secondary"]}),
                        html.Span(local_label, style={"color": CYBERPUNK_COLORS["text_secondary"]}),
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
            State("symbol-select", "value"),
        )
        def sync_symbol_select(values_lists: list[list[str]] | None, _: int, prev_selected: list[str] | None):
            # Preserve user selection across ticks; only change on checkbox input.
            try:
                symbols_all = list(self.config.runtime.symbols)
                options = [{"label": s, "value": s} for s in symbols_all]

                ctx = dash.callback_context
                triggered = ctx.triggered[0]["prop_id"].split(".")[0] if ctx and ctx.triggered else ""

                if triggered == "tick" or not values_lists:
                    # Just update options; keep prior selection
                    return options, (prev_selected or symbols_all)

                # Flatten selected values from all per-symbol checklists
                selected: list[str] = []
                for v in values_lists:
                    if v:
                        selected.extend(v)
                # Deduplicate while preserving order
                seen: set[str] = set()
                selected_unique = [s for s in selected if not (s in seen or seen.add(s))]
                return options, selected_unique
            except Exception:
                syms = list(self.config.runtime.symbols)
                return ([{"label": s, "value": s} for s in syms], prev_selected or syms)

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


def mount_dashboard(server, base_path: str = "/experiment-6/", config: AppConfig | None = None) -> Dash:
    """Mount the dashboard onto an existing Flask server under a base path.

    This enables serving the portal and dashboard on the same port without iframes.
    """
    cfg = config or load_config()
    assets_dir = str((Path(__file__).parent / "assets").resolve())
    dash_app: Dash = dash.Dash(
        __name__,
        server=server,
        url_base_pathname=base_path,
        external_stylesheets=[dbc.themes.COSMO],
        assets_folder=assets_dir,
    )
    # Build app logic and callbacks bound to this dash instance
    DashboardApp(cfg, app=dash_app)
    return dash_app


