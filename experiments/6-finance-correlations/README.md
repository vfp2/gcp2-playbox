# Experiment 6: GCP Finance Correlations

Analysis of correlations between Global Consciousness Project (GCP) data and financial market movements, implementing Ulf Holmberg's research methodology.

## Overview

This experiment replicates and extends Holmberg's research on using GCP Max[Z] metrics to predict market movements:

- **Holmberg (2020-2024)**: Found correlations between GCP1 Max[Z] and VIX, with out-of-sample trading simulations showing 5-14% excess returns
- **GCP2 Adaptation**: Applies the methodology to GCP2's network_coherence metric using rolling Z-score normalization

## What You Can Explore

### GCP Analysis Portal (Experiment 4)

The main visualization app lets you explore cumulative chi-squared deviation from randomness using either GCP1's Stouffer Z method or GCP2's rolling Z-normalized network coherence. You can compare the global network against individual clusters or drill down to any of the 473 individual RNG devices to see if specific units show stronger anomalies. The graph plots cumulative Σ(Z² - 1) over time with a 95% significance envelope - when the line breaks outside the envelope, you're seeing statistically significant departure from randomness. You can scrub through 25+ years of data, adjust bin sizes, and toggle between data sources to visually compare how GCP1 eggs vs GCP2 devices behave during the same events.

### Holmberg Financial Analysis

The Holmberg dashboard replicates Ulf Holmberg's Max[Z] market correlation research. It computes daily peak |Z| from GCP2's network coherence, then runs Pearson correlations against VIX levels, VIX changes, and SPY returns - you can also plug in any custom ticker to test the hypothesis on crypto, commodities, or individual stocks. The lag analysis chart shows whether GCP anomalies lead or follow market movements by up to 5 days. The backtesting simulation lets you test a simple strategy: go long SPY when Max[Z] exceeds a percentile threshold (P50-P95), hold for 1-10 days, and compare against buy-and-hold. You can set a custom start date to run proper out-of-sample tests and see if Holmberg's reported 5-14% excess returns hold up.

## Applications

### 1. Holmberg Analysis Dashboard (`holmberg_dashboard.py`)

Interactive web dashboard for Holmberg-style analysis:

```bash
python holmberg_dashboard.py
# Opens at http://localhost:8052
```

**Features:**
- Loads GCP2 network coherence data from CSV files
- Computes daily Max[Z] using rolling Z-score normalization
- Correlates with VIX and SPY returns
- Lag analysis (does GCP lead market movements?)
- Threshold-based backtesting simulation
- Interactive visualizations

### 2. Real-Time Predictor (`main.py serve`)

Real-time market prediction system:

```bash
python main.py serve
# Opens portal at http://localhost:8050
# Dashboard at http://localhost:8050/experiment-6/
```

**Features:**
- Collects live GCP data from global-mind.org
- Fetches market data via Alpaca API
- Real-time Max[Z] calculation
- Direction predictions (up/down)
- Performance tracking

### 3. Command-Line Analysis (`gcp2_holmberg_analysis.py`)

In the parent `gcp2.net-rng-data-downloaded/` directory:

```bash
cd ../gcp2.net-rng-data-downloaded
python gcp2_holmberg_analysis.py --data-dir . --months 6
```

## Holmberg Methodology

### GCP1 (Original Holmberg)
1. Raw RNG data → Z-scores per egg → Stouffer Z = Sum(z)/sqrt(N)
2. Max[Z] = daily max of |Stouffer Z|
3. Correlate Max[Z] with SPY returns, VIX levels, VIX changes

### GCP2 (Adaptation)
1. GCP2's `network_coherence` is NOT a Z-score - it's a pre-whitened coherence metric
2. Rolling Z-score: z_t = (nc_t - rolling_mean) / rolling_std (1-hour window)
3. Daily Max[Z] = max(|rolling_z|) per day
4. Correlate with VIX/SPY as in original methodology

## Key Findings (Holmberg 2020-2024)

- Max[Z] shows weak but significant correlation with VIX (r ≈ -0.049, p = 0.04)
- Out-of-sample trading simulations beat baseline by 5-14% annually
- Effect strongest during high-volatility periods
- Lag analysis suggests GCP may lead market by 1-2 days

## Research Papers

- `Holmberg2020.pdf` - Initial Max[Z] vs market correlation study
- `Holmberg2021.pdf` - Extended analysis with trading simulations
- `Holmberg (2022).pdf` - Cognitive Entropy Shift Model
- `A_Novel_Market_Sentiment_Measure_*.pdf` (2024) - Journal of Economic Studies publication
- Holmberg (2025) - "Can Consciousness Nudge Randomness?" - forthcoming in J. Scientific Exploration

## Data Sources

- **GCP1**: BigQuery (`gcpingcp.eggs_us.basket`) - 1998-2025
- **GCP2**: CSV files in `gcp2.net-rng-data-downloaded/network/` - March 2024+
- **Market**: yfinance (SPY, VIX) or Alpaca API

## Requirements

```bash
pip install dash dash-bootstrap-components plotly pandas numpy scipy yfinance
# For real-time predictor:
pip install alpaca-trade-api python-dotenv
```

## Configuration

For Alpaca API (real-time predictor), create `.env`:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```
