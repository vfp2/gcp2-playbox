# GCP Real-Time Market Prediction System

A real-time system that analyzes Global Consciousness Project (GCP) egg data to predict financial market movements using Max[Z] calculations and time-series analysis.

## Architecture Overview

The system consists of several key components:

### 1. **Data Collection Layer**
- **GCP Data**: Real-time egg data collection directly from [global-mind.org](http://global-mind.org)
- **Market Data**: Real-time price feeds via Alpaca Trade API
- **Buffer System**: In-memory circular buffers for recent data storage

### 2. **Data Processing Pipeline**
- **Filtering**: Ignores GCP values of 0 (noise filtering)
- **Max[Z] Calculation**: Computes Max[Z] from filtered egg data
- **Time Binning**: Groups data into configurable time windows (30s, 1min, 5min)
- **Feature Engineering**: Creates prediction features from GCP data

### 3. **Prediction Engine**
- **Direction Prediction**: Up/Down movement prediction for each market index
- **Confidence Scoring**: Probability of prediction accuracy
- **Multiple Timeframes**: Predicts different time horizons

### 4. **Performance Tracking**
- **Prediction Logging**: Records all predictions and outcomes
- **Success Metrics**: Hit rate, accuracy, profit/loss simulation
- **Real-time Dashboard**: Live statistics and performance visualization

## Key Features

### Real-Time Data Processing
- **GCP Buffer**: Stores 1000 most recent GCP readings
- **Market Buffer**: Stores recent price data for multiple symbols
- **Thread-Safe**: Concurrent data collection and analysis

### Max[Z] Calculations
- **Real-time Computation**: Calculates Max[Z] from recent GCP data
- **Threshold Filtering**: Filters out low-value readings (configurable)
- **Statistical Analysis**: Z-score based anomaly detection

### Market Prediction
- **Multi-Symbol Support**: SPY, IVV, VOO, VXX, UVXY
- **Direction Prediction**: Up/Down movement forecasts
- **Confidence Thresholds**: Only high-confidence predictions are logged

### Performance Tracking
- **24-Hour Statistics**: Rolling accuracy metrics
- **Per-Symbol Tracking**: Individual performance for each market symbol
- **Real-time Updates**: Live dashboard with current performance

## Configuration

The system is highly configurable through the `Config` class:

```python
class Config:
    # GCP Settings
    GCP_BUFFER_SIZE = 1000          # Number of GCP readings to keep
    GCP_MIN_VALUE = 0.1             # Minimum GCP value (filter out zeros)
    
    # Time Binning
    BIN_DURATION_SECONDS = 30       # Time window for predictions
    PREDICTION_HORIZON = 60         # Prediction horizon in seconds
    
    # Market Settings
    MARKET_SYMBOLS = ['SPY', 'IVV', 'VOO', 'VXX', 'UVXY']
    MARKET_UPDATE_INTERVAL = 5      # Market data update interval
    
    # Prediction Settings
    MIN_SAMPLES_FOR_PREDICTION = 10 # Minimum GCP samples needed
    CONFIDENCE_THRESHOLD = 0.6      # Minimum confidence for prediction
```

## Setup Instructions

### 1. Environment Variables
Create a `.env` file with your API credentials:

```bash
# Alpaca Trade API (required for market data)
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# No additional credentials needed for GCP data
# The system fetches real-time data directly from global-mind.org
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. GCP Data Access
The system fetches real-time data directly from the [Global Consciousness Project](http://global-mind.org) website:

- Real-time egg readings from the global network
- Live Max[Z] calculations from current data
- No simulation fallback - only real data is used

### 4. Test GCP Connection
The system will automatically test connectivity to global-mind.org when started.

## Usage

### Starting the System
1. Navigate to the experiments portal
2. Select "Finance Correlations" experiment
3. Click "START SYSTEM" to begin real-time data collection
4. Monitor the dashboard for live predictions and performance

### Dashboard Features
- **System Status**: Shows if data collection is active
- **Real-Time Data**: Current GCP and market data counts
- **Current Max[Z]**: Live Max[Z] calculation
- **Prediction Performance**: 24-hour accuracy statistics per symbol

### Prediction Model
The current prediction model uses simple threshold-based logic:

```python
if max_z > 2.5:
    if max_z > 3.0:
        return 'up', 0.8      # High confidence up
    else:
        return 'down', 0.7    # Medium confidence down
elif max_z > 1.5:
    return 'up', 0.6          # Medium confidence up
else:
    return 'down', 0.5        # Low confidence down
```

This can be enhanced with more sophisticated machine learning models.

## Data Flow

1. **GCP Data Collection**: Real-time egg readings from [global-mind.org](http://global-mind.org)
2. **Market Data Collection**: Price feeds from Alpaca API
3. **Data Filtering**: Remove zero values and noise
4. **Max[Z] Calculation**: Compute anomaly scores
5. **Prediction Generation**: Generate market direction forecasts
6. **Performance Tracking**: Log predictions and outcomes
7. **Dashboard Updates**: Real-time visualization

## Performance Metrics

The system tracks several key metrics:

- **Accuracy**: Percentage of correct predictions
- **Hit Rate**: Number of successful predictions
- **Up/Down Distribution**: Balance of prediction types
- **Per-Symbol Performance**: Individual symbol accuracy

## Future Enhancements

### Advanced Prediction Models
- Machine learning models (Random Forest, Neural Networks)
- Ensemble methods combining multiple models
- Feature engineering from historical patterns

### Enhanced Data Sources
- Additional market indicators (VIX, volatility)
- Sentiment analysis integration
- Alternative GCP metrics

### Risk Management
- Position sizing based on confidence
- Stop-loss and take-profit logic
- Portfolio-level risk controls

## Technical Notes

### Thread Safety
All data structures use thread-safe locks for concurrent access:
- `DataBuffer`: Thread-safe circular buffer
- `PredictionTracker`: Thread-safe prediction logging

### Error Handling
- Graceful degradation on API failures
- Automatic retry logic for data collection
- System stops if real data is unavailable (no simulation fallback)

### Memory Management
- Configurable buffer sizes
- Automatic cleanup of old data
- Efficient data structures for real-time processing

## Research Context

This system implements methodologies from Holmberg's research on GCP correlations with financial markets, providing a real-time framework for testing these hypotheses with live data.

## License

This project is part of the GCP2 Playbox experiments and follows the same licensing terms as the parent repository.