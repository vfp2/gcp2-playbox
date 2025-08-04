# GCP Experiments Portal

A cyberpunk-styled web interface for accessing multiple Global Consciousness Project (GCP) research experiments.

## Overview

This portal provides a unified interface for various GCP research experiments, implementing established GCP methodology protocols.

## Architecture

### Main Components

1. **`gcp_experiments_portal.py`** - Main landing page and routing server
2. **`gcp_finance_analysis.py`** - Financial market correlation analysis (Experiment #6)
3. **`experiment_4_module.py`** - EGG statistical analysis wrapper (Experiment #4)
4. **`requirements.txt`** - Python dependencies

### Experiments Available

#### Experiment #4: EGG Statistical Analysis Explorer
- **Status**: ACTIVE
- **Description**: Real-time statistical analysis of GCP EGG network data
- **Features**: 
  - Stouffer Z-score methodology across filtered RNG nodes
  - Cumulative χ² deviation analysis
  - BigQuery data pipeline integration
  - Interactive time window selection
- **Access**: `/experiment-4`

#### Experiment #6: Financial Market Correlations
- **Status**: DEVELOPMENT
- **Description**: Max[Z] anomaly correlation with financial markets
- **Features**:
  - Holmberg methodology implementation (2020-2022)
  - Linear regression: `r_(t+1) = α + β×Max[Z]_t + ε_t`
  - Bootstrap null hypothesis testing
  - Threshold conditioning analysis
  - Walk-forward backtesting simulation
- **Access**: `/experiment-6`

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   - Copy `.env` file from `../4-rolling-windows/` if using BigQuery
   - Set up Google Cloud credentials for EGG data access

3. **Launch Portal**:
   ```bash
   python gcp_experiments_portal.py
   ```

4. **Access URLs**:
   - Portal: http://localhost:8050
   - EGG Analysis: http://localhost:8050/experiment-4
   - Financial Analysis: http://localhost:8050/experiment-6

## Technical Implementation

### Cyberpunk Styling
- Consistent color scheme across all experiments
- Matrix rain background animations
- Neon glow effects and pulsing borders
- Orbitron and Courier Prime fonts

### Data Analysis Framework

#### Financial Correlation Analysis
- **Synthetic Data Generation**: Realistic GCP Max[Z] and market return patterns
- **Regression Analysis**: Rolling window estimation with multivariate controls
- **Bootstrap Testing**: Null distribution generation preserving all properties except temporal structure
- **Threshold Conditioning**: Expected returns conditional on Max[Z] significance levels
- **Visualization**: 4-panel analysis dashboard

#### Statistical Methods
- Stouffer Z-score: `Z_t^(s) = Σ(Z_i,t)/√N`
- Chi-squared deviation: `Σ(Z_t^(s))² - 1`
- Bootstrap significance testing
- Linear regression with multiple controls
- Threshold-based conditional analysis

### Module Structure
- **Modular Design**: Each experiment is a separate importable module
- **Callback Registration**: Dynamic callback registration for multi-app architecture
- **Shared Styling**: Centralized cyberpunk theme components
- **Graceful Imports**: Fallback configurations for missing dependencies

## Research Foundation

### Methodology Sources
- **Established GCP protocols**: Core methodology framework
- **Holmberg (2020-2022)**: Financial correlation research series
- **Novel Market Sentiment Measure**: Backtested trading strategies
- **Nelson-Bancel**: Original GCP statistical framework

### Key Equations Implemented

**Stouffer Z-score**:
```
Z_t^(s) = Σ(Z_i,t)/√N
```

**Cumulative Chi-squared Deviation**:
```
Σ_t (Z_t^(s))² - 1
```

**Financial Regression**:
```
r_(t+1) = α + β₁×Max[Z]_t + β₂×r_t + β₃×VIX_t + ε_t
```

**Bootstrap Null Testing**: Preserves all statistical properties except Max[Z] temporal structure

## Development Roadmap

### Planned Experiments
- **Experiment #3**: Hurst Exponent Analysis
- **Network Correlation Mapping**: Global RNG node relationships
- **Real-time Event Detection**: Automated anomaly identification

### Technical Enhancements
- Real GCP data integration (replacing synthetic data)
- Historical market data APIs
- Advanced trading strategy backtesting
- Performance optimization for large datasets
- Authentication and user management

## File Structure
```
6-finance-correlations/
├── gcp_experiments_portal.py      # Main portal server
├── gcp_finance_analysis.py        # Financial analysis module
├── experiment_4_module.py         # EGG analysis wrapper
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Notes
- The EGG analysis functionality imports from `../4-rolling-windows/gcp_egg_web_app.py`
- Financial analysis currently uses synthetic data for development/testing
- All implementations follow established GCP methodology protocols
- Cyberpunk styling provides consistent user experience across experiments