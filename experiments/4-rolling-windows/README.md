# Experiment #4: Rolling Windows Analysis of GCP EGG Data

## Overview

This experiment implements a sophisticated web-based analysis tool for exploring Global Consciousness Project (GCP) EGG data using rolling windows and statistical analysis. The application provides an interactive interface to analyze GCP v1.0 egg data stored in BigQuery, reproducing Nelson-style statistical analysis with real-time parameter adjustment.

## Key Features

- **Interactive Web Interface**: Dash-based web application with real-time parameter controls
- **Rolling Window Analysis**: Configurable time windows from 1 minute to 90 days
- **Statistical Methodology**: Implements correct GCP methodology:
  - Stouffer Z across eggs: Z_t(s) = ΣZ_i/√N (dynamic N based on active eggs)
  - χ² based on Stouffer Z: (Z_t(s))² (distributed as χ²(1) under null hypothesis)
  - Cumulative deviation: Σ((Z_t(s))² - 1) to detect departure from randomness
- **BigQuery Integration**: Direct connection to GCP v1.0 egg data in BigQuery
- **Caching System**: Local disk cache for improved performance
- **Broken Egg Filtering**: Automatic detection and filtering of malfunctioning eggs
- **Pseudo-Entropy Mode**: Optional random data generation for baseline comparison

## BigQuery Data Structure

### Project Configuration
- **GCP Project**: `gcpingcp` (default)
- **Dataset**: `eggs_us`
- **Main Table**: `basket` (raw second-level table)
- **Baseline Table**: `baseline_individual_nozeros`

### Baseline Individual Table SQL
The baseline_individual_nozeros table structure can be viewed at:
[BigQuery Console - baseline_individual_nozeros table](https://console.cloud.google.com/bigquery?sq=32174415983:577ae871d3e740dbabb420a077e1bb2d)

This table contains the processed GCP v1.0 egg data with the following key characteristics:
- **Expected Values**: μ=100, σ=7.0712 (published GCP values)
- **Data Range**: August 3, 1998 to July 31, 2025
- **Egg Columns**: 108 active eggs (filtered from 116 total, excluding broken eggs)
- **Time Resolution**: Second-level granularity

### Data Processing Pipeline
The application uses the `gcpingcp` repository's BigQuery inserter to populate the baseline_individual_nozeros table. The data processing includes:

1. **ASCII Conversion**: Raw egg values converted to ASCII representation
2. **Z-Score Calculation**: Normalized using published expected values
3. **Stouffer Z Aggregation**: Dynamic aggregation across active eggs
4. **χ² Deviation**: Statistical deviation from null hypothesis

## Installation & Setup

### Prerequisites
- Python 3.8+
- Google Cloud Platform account with BigQuery access
- Service account credentials for BigQuery access

### Dependencies
```bash
pip install -r requirements.txt
```

Required packages:
- `dash` - Web application framework
- `plotly` - Interactive plotting
- `google-cloud-bigquery` - BigQuery client
- `google-cloud-bigquery-storage` - BigQuery storage client
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `diskcache` - Local caching
- `python-dotenv` - Environment variable management

### Environment Configuration
Create a `.env` file with:
```
GCP_PROJECT=gcpingcp
GCP_DATASET=eggs_us
GCP_TABLE=basket
BASELINE_TABLE=baseline_individual_nozeros
```

### Service Account Setup
Place your BigQuery service account JSON file as `bigquery_service_account.json` in the experiment directory.

## Usage

### Starting the Application
```bash
python gcp_egg_web_app.py
```

The application will start on `http://localhost:8051`

### Interface Controls

#### Time Selection
- **Date Picker**: Select analysis start date (1998-2025 range)
- **Time Picker**: Select analysis start time (UTC)
- **Window Length**: Adjustable from 60 seconds to 90 days
- **Bin Count**: Number of time bins for analysis (1-30,000)

#### Analysis Options
- **Filter Broken Eggs**: Toggle to exclude malfunctioning eggs
- **Pseudo Entropy**: Generate random baseline data for comparison
- **Clear Cache**: Refresh BigQuery data cache

#### Visualization
- **Interactive Plot**: Real-time χ² cumulative deviation graph
- **Cyberpunk Theme**: Modern dark interface with neon accents
- **Live Readouts**: Current parameter values and statistics

### Default Analysis
The application defaults to analyzing the 9/11 event period:
- **Start Date**: September 11, 2001
- **Start Time**: 12:35 UTC (8:35 AM EDT)
- **Window Length**: 15,000 seconds (~4.2 hours)
- **Bin Count**: 15,000 bins

## Statistical Methodology

### GCP Protocol Implementation
Following standard GCP methodology:

1. **Individual Egg Z-Scores**: Z_i = (ASCII_i - 100) / 7.0712
2. **Stouffer Z Aggregation**: Z_t(s) = ΣZ_i/√N (dynamic N)
3. **χ² Calculation**: χ² = (Z_t(s))²
4. **Cumulative Deviation**: Σ(χ² - 1)

### Quality Control
- **Broken Egg Detection**: Automatic identification of eggs with zero variance
- **Dynamic N Calculation**: Adjusts for missing or inactive eggs
- **Null Value Handling**: Robust handling of missing data points
- **Division-by-Zero Protection**: Safe mathematical operations

## Data Sources

### GCP v1.0 Egg Data
The application connects to the `gcpingcp` BigQuery project containing processed GCP v1.0 egg data. This data represents the output of random number generators (RNGs) distributed globally as part of the Global Consciousness Project.

### Baseline Individual Table
The `baseline_individual_nozeros` table contains:
- **108 Active Eggs**: Filtered from original 116 eggs
- **Second-Level Timestamps**: Precise time resolution
- **Processed Values**: ASCII-converted and normalized data
- **Quality Flags**: Indicators for data quality and egg status

## Performance Features

### Caching System
- **Local Disk Cache**: 2GB cache limit for query results
- **Parameter-Based Keys**: Unique cache entries for each analysis configuration
- **Cache Invalidation**: Manual cache clearing option

### Query Optimization
- **BigQuery Storage API**: Efficient data retrieval
- **Dynamic SQL Generation**: Optimized queries based on parameters
- **Batch Processing**: Efficient handling of large time windows

## Technical Architecture

### Application Structure
```
gcp_egg_web_app.py
├── Configuration & Constants
├── SQL Query Builder
├── BigQuery Integration
├── Data Processing Pipeline
├── Dash Web Interface
├── Interactive Callbacks
└── Visualization Engine
```

### Key Components
- **SQL Builder**: Dynamic query generation for BigQuery
- **Data Processor**: Statistical calculations and aggregations
- **Web Interface**: Interactive Dash components
- **Cache Manager**: Local data caching system
- **Visualization**: Plotly-based interactive charts

## Research Applications

This tool enables researchers to:
- **Reproduce Historical Analyses**: Verify published GCP findings
- **Explore New Time Periods**: Analyze any date range in the dataset
- **Parameter Sensitivity**: Test different window sizes and bin counts
- **Quality Assessment**: Evaluate data quality across different periods
- **Baseline Comparison**: Compare real data against random baselines

## References

- **GCP v1.0 Protocol**: Standardized egg data processing
- **Nelson Analysis**: Statistical framework for consciousness research
- **BigQuery Integration**: Google Cloud Platform data warehousing

## License

This experiment is part of the GCP2 Playbox research environment for exploring Global Consciousness Project data and methodologies. 