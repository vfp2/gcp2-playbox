#!/usr/bin/env python3
"""
GCP Experiments Portal - Main Landing Page
Cyberpunk-styled web interface for accessing various Global Consciousness Project experiments.

Experiments Available:
1. EGG Statistical Analysis Explorer (Experiment #4) - Rolling windows analysis
2. Financial Market Correlations (Experiment #6) - Max[Z] trading strategies
3. [Future experiments to be added]

Author: Research Team
Based on Scott Wilber's GCP methodology expertise as single source of truth.
Refer to canon.yaml for experimental protocols.
"""

import os
from datetime import datetime as _dt
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Import experiment modules (conditional imports to avoid BigQuery initialization)
import sys
sys.path.append('../4-rolling-windows')

# ───────────────────────────── global state ────────────────────────────────
# Track which callbacks have been registered to prevent duplicates
registered_callbacks = set()

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

# ───────────────────────────── portal app setup ────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "GCP Experiments Portal"

# Cyberpunk CSS with enhanced matrix-style animations
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Courier+Prime:wght@400;700&display=swap');
            
            @keyframes matrixRain {
                0% { transform: translateY(-100vh); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(100vh); opacity: 0; }
            }
            
            @keyframes glowPulse {
                0%, 100% { text-shadow: 0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor; }
                50% { text-shadow: 0 0 10px currentColor, 0 0 20px currentColor, 0 0 30px currentColor; }
            }
            
            @keyframes neonBorder {
                0%, 100% { border-color: #ff006e; box-shadow: 0 0 10px #ff006e40; }
                25% { border-color: #00d4ff; box-shadow: 0 0 10px #00d4ff40; }
                50% { border-color: #9d4edd; box-shadow: 0 0 10px #9d4edd40; }
                75% { border-color: #00ff88; box-shadow: 0 0 10px #00ff8840; }
            }
            
            .matrix-bg {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: -1;
                overflow: hidden;
            }
            
            .matrix-char {
                position: absolute;
                color: #00ff88;
                font-family: 'Courier Prime', monospace;
                font-size: 18px;
                animation: matrixRain 8s linear infinite;
                opacity: 0.3;
            }
            
            .glow-text {
                animation: glowPulse 3s ease-in-out infinite;
            }
            
            .experiment-card {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 2px solid #ff006e;
                border-radius: 15px;
                padding: 20px;
                margin: 15px 0;
                transition: all 0.3s ease;
                animation: neonBorder 6s ease-in-out infinite;
            }
            
            .experiment-card:hover {
                transform: translateY(-5px) scale(1.02);
                box-shadow: 0 15px 30px rgba(255, 0, 110, 0.3);
                border-color: #00d4ff !important;
            }
            
            .experiment-link {
                color: #00d4ff !important;
                text-decoration: none !important;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            
            .experiment-link:hover {
                color: #ff006e !important;
                text-shadow: 0 0 10px #ff006e;
            }
            
            .status-badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
                margin-left: 10px;
            }
            
            .status-active {
                background: linear-gradient(45deg, #00ff88, #4dffa3);
                color: #0a0a0f;
            }
            
            .status-development {
                background: linear-gradient(45deg, #ffbe0b, #ffd23f);
                color: #0a0a0f;
            }
            
            .status-planned {
                background: linear-gradient(45deg, #9d4edd, #b366e6);
                color: #ffffff;
            }
        </style>
    </head>
    <body>
        <div class="matrix-bg" id="matrix-bg"></div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // Matrix rain effect
            function createMatrix() {
                const chars = '01αβγδεζηθικλμνξοπρστυφχψωΩΨΦΧΥΤΣΡΠΞΝΜΛΚΙΘΗΖΕΔΓΒΑ';
                const container = document.getElementById('matrix-bg');
                
                function addChar() {
                    const char = document.createElement('div');
                    char.className = 'matrix-char';
                    char.textContent = chars[Math.floor(Math.random() * chars.length)];
                    char.style.left = Math.random() * 100 + '%';
                    char.style.animationDelay = Math.random() * 8 + 's';
                    char.style.animationDuration = (Math.random() * 4 + 6) + 's';
                    container.appendChild(char);
                    
                    setTimeout(() => {
                        if (char.parentNode) {
                            char.parentNode.removeChild(char);
                        }
                    }, 12000);
                }
                
                // Add characters periodically
                setInterval(addChar, 300);
                
                // Initial burst
                for (let i = 0; i < 20; i++) {
                    setTimeout(addChar, i * 200);
                }
            }
            
            // Start matrix effect when page loads
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', createMatrix);
            } else {
                createMatrix();
            }
        </script>
    </body>
</html>
'''

# ───────────────────────────── portal layout ────────────────────────────────
def create_portal_layout():
    return html.Div([
        # Header Section
        html.Div([
            html.Div([
                html.H1("GLOBAL CONSCIOUSNESS PROJECT", 
                       className="glow-text",
                       style={
                           "textAlign": "center",
                           "color": CYBERPUNK_COLORS['text_primary'],
                           "fontSize": "3rem",
                           "fontWeight": "900",
                           "fontFamily": "'Orbitron', 'Courier New', monospace",
                           "letterSpacing": "4px",
                           "marginBottom": "10px",
                           "textShadow": f"0 0 20px {CYBERPUNK_COLORS['neon_cyan']}"
                       }),
                html.H2("EXPERIMENTAL RESEARCH PORTAL", 
                       style={
                           "textAlign": "center",
                           "color": CYBERPUNK_COLORS['neon_pink'],
                           "fontSize": "1.8rem",
                           "fontWeight": "700",
                           "fontFamily": "'Orbitron', 'Courier New', monospace",
                           "letterSpacing": "2px",
                           "marginBottom": "20px",
                           "textShadow": f"0 0 15px {CYBERPUNK_COLORS['neon_pink']}"
                       }),
                html.P([
                    "Advanced statistical analysis of Random Number Generator networks detecting ",
                    "departure from randomness during periods of ",
                    html.A("collective human consciousness", 
                           href="https://gcp2.net/",
                           style={"color": CYBERPUNK_COLORS['neon_cyan'], "textDecoration": "underline"}),
                    ". ",
                    html.Br(),
                    html.Strong("by FP2 ", 
                               style={"color": CYBERPUNK_COLORS['neon_yellow']}),
                    html.A("forum", href="https://forum.fp2.dev", 
                           style={"color": CYBERPUNK_COLORS['neon_cyan'], "textDecoration": "underline"}),
                    html.Strong(" members", 
                               style={"color": CYBERPUNK_COLORS['neon_yellow']})
                ], style={
                    "fontSize": "16px", 
                    "color": CYBERPUNK_COLORS['text_secondary'],
                    "marginBottom": "30px",
                    "textAlign": "center",
                    "fontFamily": "'Courier Prime', monospace",
                    "lineHeight": "1.6"
                }),
            ], style={
                "background": CYBERPUNK_COLORS['accent_gradient'],
                "padding": "30px",
                "borderRadius": "20px",
                "border": f"3px solid {CYBERPUNK_COLORS['neon_cyan']}",
                "boxShadow": f"0 0 40px {CYBERPUNK_COLORS['neon_cyan']}30",
                "marginBottom": "40px"
            })
        ]),
        
        # Experiments Section
        html.Div([
            html.H3("ACTIVE EXPERIMENTS", 
                   style={
                       "color": CYBERPUNK_COLORS['neon_green'],
                       "fontSize": "1.5rem",
                       "fontWeight": "bold",
                       "fontFamily": "'Orbitron', monospace",
                       "marginBottom": "30px",
                       "textAlign": "center",
                       "textShadow": f"0 0 10px {CYBERPUNK_COLORS['neon_green']}"
                   }),
            
            # Experiment #4 - EGG Analysis
            html.Div([
                html.Div([
                    html.H4([
                        "EXPERIMENT #4: EGG STATISTICAL ANALYSIS EXPLORER",
                        html.Span("ACTIVE", className="status-badge status-active")
                    ], style={
                        "color": CYBERPUNK_COLORS['neon_cyan'],
                        "fontSize": "1.2rem",
                        "fontWeight": "bold",
                        "marginBottom": "15px",
                        "fontFamily": "'Orbitron', monospace"
                    }),
                    html.P([
                        "Original ",
                        html.A("Roger Nelson statistical analysis methods", 
                               href="https://noosphere.princeton.edu/papers/jseNelson.pdf?utm_source=chatgpt.com",
                               style={"color": CYBERPUNK_COLORS['neon_cyan'], "textDecoration": "underline"}),
                        " of Global Consciousness Project EGG network data. ",
                        "Implements Stouffer Z-score methodology across filtered RNG nodes, calculating ",
                        "cumulative χ² deviations to detect departure from randomness during significant events."
                    ], style={
                        "color": CYBERPUNK_COLORS['text_secondary'],
                        "fontSize": "14px",
                        "marginBottom": "15px",
                        "fontFamily": "'Courier Prime', monospace",
                        "lineHeight": "1.5"
                    }),
                    html.Div([
                        html.Strong("Features: ", style={"color": CYBERPUNK_COLORS['neon_yellow']}),
                        "Interactive time window selection • Stouffer Z across active eggs • ",
                        "Chi-squared statistical analysis • BigQuery data pipeline • ",
                        "Dynamic visualization"
                    ], style={
                        "color": CYBERPUNK_COLORS['text_secondary'],
                        "fontSize": "13px",
                        "marginBottom": "20px",
                        "fontFamily": "'Courier Prime', monospace"
                    }),
                    html.Div([
                        dcc.Link(
                            "→ LAUNCH EGG ANALYSIS INTERFACE",
                            href="/experiment-4",
                            className="experiment-link",
                            style={
                                "fontSize": "16px",
                                "fontFamily": "'Orbitron', monospace",
                                "padding": "10px 20px",
                                "border": f"2px solid {CYBERPUNK_COLORS['neon_cyan']}",
                                "borderRadius": "25px",
                                "display": "inline-block",
                                "textDecoration": "none"
                            }
                        )
                    ])
                ], className="experiment-card")
            ]),
            
            # Experiment #6 - Financial Correlations
            html.Div([
                html.Div([
                    html.H4([
                        "EXPERIMENT #6: FINANCIAL MARKET CORRELATIONS",
                        html.Span("IN-DEVELOPMENT", className="status-badge status-development")
                    ], style={
                        "color": CYBERPUNK_COLORS['neon_purple'],
                        "fontSize": "1.2rem",
                        "fontWeight": "bold",
                        "marginBottom": "15px",
                        "fontFamily": "'Orbitron', monospace"
                    }),
                    html.P([
                        "Advanced analysis correlating Max[Z] anomaly metrics from GCP network with financial market ",
                        "movements. Implements ",
                        html.A("Ulf Holmberg research methodologies", 
                               href="https://ulfholmberg.info/",
                               style={"color": CYBERPUNK_COLORS['neon_cyan'], "textDecoration": "underline"}),
                        " (2020-2022) for predictive modeling, ",
                        "regression analysis, and backtested trading strategy simulation."
                    ], style={
                        "color": CYBERPUNK_COLORS['text_secondary'],
                        "fontSize": "14px",
                        "marginBottom": "15px",
                        "fontFamily": "'Courier Prime', monospace",
                        "lineHeight": "1.5"
                    }),
                    html.Div([
                        html.Strong("Implementation Pipeline: ", style={"color": CYBERPUNK_COLORS['neon_yellow']}),
                        "Max[Z] extraction from GCP logs • Linear & multivariate regression r_(t+1) = α + β×Max[Z]_t • ",
                        "Bootstrap null distributions • Threshold conditioning analysis • ",
                        "Walk-forward backtesting • Trading signal generation • Sharpe ratio optimization"
                    ], style={
                        "color": CYBERPUNK_COLORS['text_secondary'],
                        "fontSize": "13px",
                        "marginBottom": "20px",
                        "fontFamily": "'Courier Prime', monospace"
                    }),
                    html.Div([
                        dcc.Link(
                            "→ LAUNCH FINANCIAL ANALYSIS INTERFACE",
                            href="/experiment-6",
                            className="experiment-link",
                            style={
                                "fontSize": "16px",
                                "fontFamily": "'Orbitron', monospace",
                                "padding": "10px 20px",
                                "border": f"2px solid {CYBERPUNK_COLORS['neon_purple']}",
                                "borderRadius": "25px",
                                "display": "inline-block",
                                "textDecoration": "none"
                            }
                        )
                    ])
                ], className="experiment-card")
            ]),
            
            # Future Experiments Section
            html.Div([
                html.H4("RESEARCH PIPELINE", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_yellow'],
                           "fontSize": "1.1rem",
                           "fontWeight": "bold",
                           "marginTop": "40px",
                           "marginBottom": "20px",
                           "fontFamily": "'Orbitron', monospace",
                           "textAlign": "center"
                       }),
                html.Div([
                    html.Div([
                        "• Hurst Exponent Analysis (Experiment #3)",
                        html.Span("PLANNED", className="status-badge status-planned")
                    ], style={"marginBottom": "10px", "color": CYBERPUNK_COLORS['text_secondary']}),
                    html.Div([
                        "• Network Correlation Mapping",
                        html.Span("PLANNED", className="status-badge status-planned")
                    ], style={"marginBottom": "10px", "color": CYBERPUNK_COLORS['text_secondary']}),
                    html.Div([
                        "• Real-time Event Detection",
                        html.Span("PLANNED", className="status-badge status-planned")
                    ], style={"marginBottom": "10px", "color": CYBERPUNK_COLORS['text_secondary']}),
                ], style={
                    "fontFamily": "'Courier Prime', monospace",
                    "fontSize": "14px",
                    "textAlign": "center"
                })
            ])
        ], style={
            "maxWidth": "1200px",
            "margin": "0 auto",
            "padding": "20px"
        }),
        
        # Footer
        html.Div([
            html.P([
                f"System Online: {_dt.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                html.Br(),
                html.A("Source Code", href="https://github.com/vfp2/gcp2-playbox", 
                       style={
                           "color": CYBERPUNK_COLORS['neon_cyan'],
                           "textDecoration": "underline",
                           "marginLeft": "20px"
                       })
            ], style={
                "textAlign": "center",
                "color": CYBERPUNK_COLORS['text_secondary'],
                "fontSize": "12px",
                "fontFamily": "'Courier Prime', monospace",
                "marginTop": "50px",
                "padding": "20px",
                "borderTop": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}30"
            })
        ])
    ], style={
        "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
        "minHeight": "100vh",
        "padding": "20px",
        "fontFamily": "'Courier Prime', monospace"
    })

# ───────────────────────────── routing logic ────────────────────────────────
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/experiment-4':
        try:
            # Import and serve the EGG analysis app directly from the original file
            sys.path.append('../4-rolling-windows')
            from gcp_egg_web_app import app as egg_app, create_egg_callback
            
            # Register the EGG callback with our main app (only once)
            if 'egg_callback' not in registered_callbacks:
                create_egg_callback(app)
                registered_callbacks.add('egg_callback')
            
            # Get the layout from the EGG app and create a fresh copy
            egg_layout = egg_app.layout
            
            # Create a new layout with return button to avoid duplication
            if isinstance(egg_layout, html.Div):
                # Create a fresh copy of the children
                children = list(egg_layout.children)
                
                # Add return button at the end
                return_button = html.Div([
                    html.A(
                        "← RETURN TO EXPERIMENTS PORTAL",
                        href="/",
                        style={
                            "color": CYBERPUNK_COLORS['neon_cyan'],
                            "fontSize": "14px",
                            "fontFamily": "'Orbitron', monospace",
                            "textDecoration": "none",
                            "padding": "8px 16px",
                            "border": f"1px solid {CYBERPUNK_COLORS['neon_cyan']}",
                            "borderRadius": "20px",
                            "display": "inline-block",
                            "marginTop": "20px"
                        }
                    )
                ], style={"textAlign": "center", "marginTop": "20px"})
                
                # Create a new layout with the return button
                new_layout = html.Div(children + [return_button], style=egg_layout.style)
                return new_layout
            
            return egg_layout
            
        except Exception as e:
            # Show the actual error that occurred
            return html.Div([
                html.H3("Experiment #4 Error", 
                       style={"color": CYBERPUNK_COLORS['neon_pink'], "textAlign": "center"}),
                html.P([
                    f"Error loading EGG analysis: {str(e)}"
                ], style={
                    "color": CYBERPUNK_COLORS['text_secondary'], 
                    "textAlign": "center",
                    "fontFamily": "'Courier New', monospace",
                    "fontSize": "14px",
                    "marginBottom": "20px"
                }),
                html.Div([
                    html.P("Common issues:", style={
                        "color": CYBERPUNK_COLORS['neon_yellow'],
                        "fontWeight": "bold",
                        "marginBottom": "10px"
                    }),
                    html.Ul([
                        html.Li("BigQuery credentials not configured"),
                        html.Li("Missing Google Cloud service account file"),
                        html.Li("Network connectivity issues"),
                        html.Li("Missing dependencies")
                    ], style={
                        "color": CYBERPUNK_COLORS['text_secondary'],
                        "fontFamily": "'Courier New', monospace",
                        "fontSize": "12px",
                        "textAlign": "left",
                        "display": "inline-block"
                    })
                ], style={"textAlign": "center", "marginBottom": "20px"}),
                html.Div([
                    dcc.Link("← Return to Portal", href="/", 
                            style={"color": CYBERPUNK_COLORS['neon_cyan']})
                ], style={"textAlign": "center", "marginTop": "20px"})
            ], style={
                "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
                "minHeight": "100vh",
                "padding": "50px",
                "fontFamily": "'Courier New', monospace"
            })
    elif pathname == '/experiment-6':
        # Import and serve the financial analysis app
        from gcp_finance_analysis import get_finance_layout, register_finance_callbacks
        # Register the financial analysis callbacks (only once)
        if 'finance_callback' not in registered_callbacks:
            register_finance_callbacks(app)
            registered_callbacks.add('finance_callback')
        return get_finance_layout()
    else:
        # Default to portal landing page
        return create_portal_layout()

# ───────────────────────────── main server ────────────────────────────────
if __name__ == "__main__":
    print("GCP Experiments Portal starting...")
    print("Portal: http://localhost:8050")
    print("EGG Analysis: http://localhost:8050/experiment-4")
    print("Financial Analysis: http://localhost:8050/experiment-6")
    print("Neural interface online...")
    
    app.run_server(debug=True, host="0.0.0.0", port=8050)