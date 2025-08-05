#!/usr/bin/env python3
"""
GCP Real-Time Market Prediction System (Experiment #6)
Real-time analysis of Global Consciousness Project egg data to predict
financial market movements using Max[Z] calculations and time-series analysis.

Architecture:
- Real-time GCP data collection via BigQuery
- Market data via Alpaca API
- Time-binned Max[Z] calculations
- Direction prediction (Up/Down)
- Performance tracking and statistics
"""

import os
import time
import threading
import queue
import json
from datetime import datetime, timedelta
from collections import deque, defaultdict
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import requests
import csv
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash.dependencies
import dash_bootstrap_components as dbc

# Load environment variables
load_dotenv()

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

# ───────────────────────────── configuration ─────────────────────────────
class Config:
    """Configuration settings for the prediction system."""
    
    # GCP Settings
    GCP_BUFFER_SIZE = 1000  # Number of GCP readings to keep in memory
    GCP_MIN_VALUE = 0.1     # Minimum GCP value (filter out zeros)
    
    # Time Binning
    BIN_DURATION_SECONDS = 30  # Time window for predictions (30 seconds)
    PREDICTION_HORIZON = 60    # Prediction horizon in seconds
    
    # Market Settings
    MARKET_SYMBOLS = ['SPY', 'IVV', 'VOO', 'VXX', 'UVXY']
    MARKET_UPDATE_INTERVAL = 5  # Market data update interval (seconds)
    
    # Prediction Settings
    MIN_SAMPLES_FOR_PREDICTION = 10  # Minimum GCP samples needed for prediction
    CONFIDENCE_THRESHOLD = 0.6       # Minimum confidence for prediction

# ───────────────────────────── data structures ─────────────────────────────
class DataBuffer:
    """Circular buffer for storing time-series data."""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, data_point):
        """Add a data point to the buffer."""
        with self.lock:
            self.buffer.append(data_point)
    
    def get_recent(self, n_samples=None):
        """Get the most recent n samples."""
        with self.lock:
            if n_samples is None:
                return list(self.buffer)
            return list(self.buffer)[-n_samples:]
    
    def get_time_window(self, seconds):
        """Get data from the last N seconds."""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        with self.lock:
            return [dp for dp in self.buffer if dp['timestamp'] >= cutoff_time]
    
    def size(self):
        """Get current buffer size."""
        with self.lock:
            return len(self.buffer)

class PredictionTracker:
    """Track prediction performance and statistics."""
    
    def __init__(self):
        self.predictions = []
        self.lock = threading.Lock()
    
    def add_prediction(self, symbol, prediction, confidence, actual_direction=None):
        """Add a new prediction."""
        with self.lock:
            pred = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'predicted_direction': prediction,  # 'up' or 'down'
                'confidence': confidence,
                'actual_direction': actual_direction,
                'correct': None
            }
            self.predictions.append(pred)
    
    def update_actual(self, symbol, actual_direction):
        """Update the most recent prediction with actual result."""
        with self.lock:
            for pred in reversed(self.predictions):
                if pred['symbol'] == symbol and pred['actual_direction'] is None:
                    pred['actual_direction'] = actual_direction
                    pred['correct'] = (pred['predicted_direction'] == actual_direction)
                    break
    
    def get_stats(self, symbol=None, hours=24):
        """Get prediction statistics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            recent_preds = [p for p in self.predictions 
                          if p['timestamp'] >= cutoff_time and p['actual_direction'] is not None]
            
            if symbol:
                recent_preds = [p for p in recent_preds if p['symbol'] == symbol]
            
            if not recent_preds:
                return {'total': 0, 'correct': 0, 'accuracy': 0.0}
            
            correct = sum(1 for p in recent_preds if p['correct'])
            total = len(recent_preds)
            
            return {
                'total': total,
                'correct': correct,
                'accuracy': correct / total if total > 0 else 0.0,
                'up_predictions': sum(1 for p in recent_preds if p['predicted_direction'] == 'up'),
                'down_predictions': sum(1 for p in recent_preds if p['predicted_direction'] == 'down')
            }

# ───────────────────────────── data collection ─────────────────────────────
class GCPDataCollector:
    """Collect real-time GCP egg data from global-mind.org."""
    
    def __init__(self, buffer):
        self.buffer = buffer
        self.base_url = "http://global-mind.org"
        self.running = False
        self.thread = None
    
    def start(self):
        """Start collecting GCP data."""
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop collecting GCP data."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collect_loop(self):
        """Main collection loop - fetch real-time data from global-mind.org."""
        while self.running:
            try:
                # Fetch real-time GCP data from global-mind.org
                self._fetch_gcp_data()
                time.sleep(1)  # Collect every second
            except Exception as e:
                print(f"GCP collection error: {e}")
                time.sleep(5)
    
    def _fetch_gcp_data(self):
        """Fetch real-time GCP egg data from global-mind.org using the official API."""
        try:
            # Use the exact API endpoint from the BigQuery inserter
            # http://global-mind.org/cgi-bin/eggdatareq.pl?z=1&year=YYYY&month=MM&day=DD&stime=00:00:00&etime=23:59:59
            
            # Get current UTC date for the API request
            current_date = datetime.utcnow()
            year = current_date.strftime('%Y')
            month = current_date.strftime('%m')
            day = current_date.strftime('%d')
            
            # Construct the API URL exactly as in the BigQuery inserter
            api_url = (f"{self.base_url}/cgi-bin/eggdatareq.pl"
                      f"?z=1"
                      f"&year={year}"
                      f"&month={month}"
                      f"&day={day}"
                      f"&stime=00:00:00"
                      f"&etime=23:59:59")
            
            print(f"Fetching GCP data from: {api_url}")
            
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                # Parse the CSV response using the same logic as the BigQuery inserter
                egg_data = self._parse_gcp_csv_response(response.text)
                if egg_data:
                    for egg in egg_data:
                        if egg['value'] > Config.GCP_MIN_VALUE:
                            self.buffer.add(egg)
                    print(f"Successfully fetched {len(egg_data)} GCP data points")
                    return
                else:
                    print("No valid egg data found in response")
                    raise Exception("No valid egg data in API response")
            else:
                print(f"GCP API returned status code: {response.status_code}")
                raise Exception(f"GCP API error: {response.status_code}")
            
        except Exception as e:
            print(f"Error fetching GCP data: {e}")
            raise e  # Re-raise the exception instead of falling back to simulation
    
    def _parse_gcp_csv_response(self, csv_content):
        """Parse CSV response from GCP API using the same logic as the BigQuery inserter."""
        try:
            # Parse the CSV output exactly as in the BigQuery inserter
            # Reference: http://noosphere.princeton.edu/basket_CSV_v2.html
            
            egg_data = []
            egg_ids = []
            current_timestamp = None
            
            # Parse CSV content
            csv_reader = csv.reader(StringIO(csv_content))
            
            for record in csv_reader:
                if not record:  # Skip empty rows
                    continue
                
                try:
                    field_type = int(record[0])
                    
                    if field_type == 12:  # Field type 12: IDs of EGGs in today's sample set
                        # Extract egg IDs (skip field type, "gmtime" and empty)
                        egg_ids = [f"egg_{egg_id}" for egg_id in record[3:] if egg_id.strip()]
                        print(f"Found {len(egg_ids)} active eggs: {egg_ids[:5]}...")  # Show first 5
                    
                    elif field_type == 13:  # Field type 13: actual sample data
                        if not egg_ids:
                            print("No egg IDs found, skipping sample data")
                            continue
                        
                        # Parse timestamp (Unix timestamp in seconds)
                        unix_timestamp = int(record[1])
                        current_timestamp = datetime.fromtimestamp(unix_timestamp)
                        
                        # Extract sample values (skip field type, unix timestamp, user friendly timestamp)
                        sample_values = record[3:]
                        
                        # Create data points for each egg
                        for i, value_str in enumerate(sample_values):
                            if i < len(egg_ids) and value_str.strip():
                                try:
                                    value = int(value_str)
                                    if value > Config.GCP_MIN_VALUE:  # Filter out low values
                                        egg_data.append({
                                            'timestamp': current_timestamp,
                                            'egg_id': egg_ids[i],
                                            'value': value,
                                            'site_id': egg_ids[i]  # Use egg_id as site_id
                                        })
                                except ValueError:
                                    continue  # Skip invalid values
                
                except (ValueError, IndexError) as e:
                    # Skip malformed records
                    continue
            
            print(f"Parsed {len(egg_data)} valid egg data points from CSV")
            return egg_data
            
        except Exception as e:
            print(f"Error parsing GCP CSV response: {e}")
            return []
    
    # Removed _simulate_gcp_data method - no simulation fallback

class MarketDataCollector:
    """Collect real-time market data via Alpaca API."""
    
    def __init__(self, buffer):
        self.buffer = buffer
        
        # Load credentials from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")
        
        self.api = tradeapi.REST(
            api_key,
            secret_key,
            base_url='https://paper-api.alpaca.markets'
        )
        self.running = False
        self.thread = None
    
    def start(self):
        """Start collecting market data."""
        self.running = True
        self.thread = threading.Thread(target=self._collect_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop collecting market data."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collect_loop(self):
        """Main collection loop."""
        while self.running:
            try:
                for symbol in Config.MARKET_SYMBOLS:
                    trade = self.api.get_latest_trade(symbol)
                    data_point = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'price': trade.price,
                        'volume': trade.size
                    }
                    self.buffer.add(data_point)
                
                time.sleep(Config.MARKET_UPDATE_INTERVAL)
            except Exception as e:
                print(f"Market collection error: {e}")
                time.sleep(Config.MARKET_UPDATE_INTERVAL)

# ───────────────────────────── analysis engine ─────────────────────────────
class MaxZCalculator:
    """Calculate Max[Z] from GCP egg data."""
    
    @staticmethod
    def calculate_max_z(gcp_data):
        """Calculate Max[Z] from a list of GCP readings using GCP methodology."""
        if not gcp_data:
            return 0.0
        
        # Convert to numpy array for calculations
        values = np.array([dp['value'] for dp in gcp_data])
        
        # GCP methodology: Expected values μ=100, σ=7.0712
        # Calculate Z-scores using GCP expected parameters
        expected_mean = 100.0
        expected_std = 7.0712
        
        z_scores = (values - expected_mean) / expected_std
        
        # Return the maximum absolute Z-score (Max[Z])
        return np.max(np.abs(z_scores))

class PredictionEngine:
    """Generate market direction predictions based on GCP data."""
    
    def __init__(self, gcp_buffer, market_buffer, tracker):
        self.gcp_buffer = gcp_buffer
        self.market_buffer = market_buffer
        self.tracker = tracker
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the prediction engine."""
        self.running = True
        self.thread = threading.Thread(target=self._prediction_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the prediction engine."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _prediction_loop(self):
        """Main prediction loop."""
        while self.running:
            try:
                # Get all available GCP data (since GCP data is historical with same timestamps)
                gcp_data = self.gcp_buffer.get_recent()
                
                if len(gcp_data) >= Config.MIN_SAMPLES_FOR_PREDICTION:
                    # Calculate Max[Z]
                    max_z = MaxZCalculator.calculate_max_z(gcp_data)
                    
                    # Generate predictions for each symbol
                    for symbol in Config.MARKET_SYMBOLS:
                        prediction, confidence = self._predict_direction(max_z, symbol)
                        
                        if confidence >= Config.CONFIDENCE_THRESHOLD:
                            self.tracker.add_prediction(symbol, prediction, confidence)
                
                time.sleep(Config.BIN_DURATION_SECONDS)
            except Exception as e:
                print(f"Prediction error: {e}")
                time.sleep(5)
    
    def _predict_direction(self, max_z, symbol):
        """Predict market direction based on Max[Z]."""
        # Simple prediction model based on Max[Z] thresholds
        # This can be enhanced with more sophisticated ML models
        
        if max_z > 2.5:
            # High anomaly - predict significant movement
            if max_z > 3.0:
                return 'up', 0.8  # High confidence up
            else:
                return 'down', 0.7  # Medium confidence down
        elif max_z > 1.5:
            # Medium anomaly - predict moderate movement
            return 'up', 0.6
        else:
            # Low anomaly - predict minimal movement
            return 'down', 0.5

# ───────────────────────────── dashboard ─────────────────────────────
def get_finance_layout():
    """Return the Dash layout for the real-time prediction dashboard."""
    
    return html.Div([
        # Header Section
        html.Div([
            html.H1("GCP REAL-TIME MARKET PREDICTION SYSTEM", 
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
                        id="start-system-btn",
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
                        id="stop-system-btn",
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
                    html.Div(id="system-status", children="SYSTEM OFFLINE", 
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
                    html.Div(id="gcp-stats", children="GCP Data: 0 samples", 
                            style={
                                "color": CYBERPUNK_COLORS['text_primary'],
                                "fontFamily": "'Courier New', monospace",
                                "padding": "10px",
                                "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                                "borderRadius": "5px"
                            })
                ], width=4),
                
                dbc.Col([
                    html.Div(id="market-stats", children="Market Data: 0 samples", 
                            style={
                                "color": CYBERPUNK_COLORS['text_primary'],
                                "fontFamily": "'Courier New', monospace",
                                "padding": "10px",
                                "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                                "borderRadius": "5px"
                            })
                ], width=4),
                
                dbc.Col([
                    html.Div(id="current-maxz", children="Current Max[Z]: 0.00", 
                            style={
                                "color": CYBERPUNK_COLORS['text_primary'],
                                "fontFamily": "'Courier New', monospace",
                                "padding": "10px",
                                "backgroundColor": CYBERPUNK_COLORS['bg_medium'],
                                "borderRadius": "5px"
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
            
            html.Div(id="prediction-stats", children=[
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
        
        # Charts
        html.Div(id="charts-container", children=[
                    html.Div([
                html.H4("DATA VISUALIZATION", 
                               style={
                                   "textAlign": "center",
                                   "color": CYBERPUNK_COLORS['text_secondary'],
                                   "fontSize": "1.5rem",
                                   "fontFamily": "'Orbitron', monospace",
                                   "marginTop": "50px"
                               }),
                html.P("Start the system to see real-time data visualization.",
                              style={
                                  "textAlign": "center",
                                  "color": CYBERPUNK_COLORS['text_secondary'],
                                  "fontFamily": "'Courier New', monospace",
                                  "marginTop": "20px"
                              })
                    ])
        ]),
        
        # Back to Portal Link
        html.Div([
            dcc.Link(
                "← RETURN TO EXPERIMENTS PORTAL",
                href="/",
                style={
                    "color": CYBERPUNK_COLORS['neon_cyan'],
                    "fontSize": "16px",
                    "fontFamily": "'Orbitron', monospace",
                    "textDecoration": "none",
                    "padding": "10px 20px",
                    "border": f"2px solid {CYBERPUNK_COLORS['neon_cyan']}",
                    "borderRadius": "25px",
                    "display": "inline-block"
                }
            )
        ], style={"textAlign": "center", "marginTop": "40px"})
        
    ], style={
        "backgroundColor": CYBERPUNK_COLORS['bg_dark'],
        "minHeight": "100vh",
        "padding": "20px",
        "fontFamily": "'Courier New', monospace"
    })

# ───────────────────────────── global system state ─────────────────────────────
# Initialize global data structures
gcp_buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
market_buffer = DataBuffer(Config.GCP_BUFFER_SIZE)
prediction_tracker = PredictionTracker()

# Initialize collectors and engine
gcp_collector = GCPDataCollector(gcp_buffer)
market_collector = MarketDataCollector(market_buffer)
prediction_engine = PredictionEngine(gcp_buffer, market_buffer, prediction_tracker)

system_running = False

# ───────────────────────────── callbacks ────────────────────────────────
def register_finance_callbacks(app):
    """Register callbacks for the real-time prediction system."""
    
    @app.callback(
        [Output("system-status", "children"),
         Output("start-system-btn", "disabled"),
         Output("stop-system-btn", "disabled")],
        [Input("start-system-btn", "n_clicks"),
         Input("stop-system-btn", "n_clicks")]
    )
    def control_system(start_clicks, stop_clicks):
        global system_running
        
        ctx = dash.callback_context
        if not ctx.triggered:
            return "SYSTEM OFFLINE", False, True
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == "start-system-btn" and not system_running:
            # Start the system
            gcp_collector.start()
            market_collector.start()
            prediction_engine.start()
            system_running = True
            return "SYSTEM ONLINE", True, False
        
        elif button_id == "stop-system-btn" and system_running:
            # Stop the system
            gcp_collector.stop()
            market_collector.stop()
            prediction_engine.stop()
            system_running = False
            return "SYSTEM OFFLINE", False, True
        
        return ("SYSTEM ONLINE" if system_running else "SYSTEM OFFLINE"), system_running, not system_running
    
    @app.callback(
        [Output("gcp-stats", "children"),
         Output("market-stats", "children"),
         Output("current-maxz", "children")],
        [Input("start-system-btn", "n_clicks"),
         Input("stop-system-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def update_data_stats(start_clicks, stop_clicks):
        if not system_running:
            return "GCP Data: 0 samples", "Market Data: 0 samples", "Current Max[Z]: 0.00"
        
        # Get current stats
        gcp_count = gcp_buffer.size()
        market_count = market_buffer.size()
        
        # Calculate current Max[Z]
        recent_gcp = gcp_buffer.get_time_window(Config.BIN_DURATION_SECONDS)
        current_maxz = MaxZCalculator.calculate_max_z(recent_gcp)
        
        return (f"GCP Data: {gcp_count} samples",
                f"Market Data: {market_count} samples",
                f"Current Max[Z]: {current_maxz:.2f}")
    
    @app.callback(
        Output("prediction-stats", "children"),
        [Input("start-system-btn", "n_clicks"),
         Input("stop-system-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def update_prediction_stats(start_clicks, stop_clicks):
        if not system_running:
            return html.P("No predictions yet", 
                         style={
                             "color": CYBERPUNK_COLORS['text_secondary'],
                             "fontFamily": "'Courier New', monospace",
                             "textAlign": "center"
                         })
        
        # Get prediction stats for each symbol
        stats_html = []
        
        for symbol in Config.MARKET_SYMBOLS:
            stats = prediction_tracker.get_stats(symbol, hours=24)
            
            if stats['total'] > 0:
                accuracy_color = CYBERPUNK_COLORS['neon_green'] if stats['accuracy'] > 0.6 else CYBERPUNK_COLORS['neon_pink']
                
                stats_html.append(html.Div([
                    html.Strong(f"{symbol}: ", style={"color": CYBERPUNK_COLORS['neon_cyan']}),
                    f"Accuracy: {stats['accuracy']*100:.1f}% ",
                    f"({stats['correct']}/{stats['total']}) ",
                    f"Up: {stats['up_predictions']} Down: {stats['down_predictions']}"
                ], style={
                    "color": accuracy_color,
                            "fontFamily": "'Courier New', monospace",
                            "marginBottom": "5px"
                }))
        
        if not stats_html:
            return html.P("No predictions yet", 
                         style={
                        "color": CYBERPUNK_COLORS['text_secondary'],
                        "fontFamily": "'Courier New', monospace",
                             "textAlign": "center"
                         })
        
        return stats_html