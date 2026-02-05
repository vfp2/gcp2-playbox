# GCP2 Data Analysis Context

**Last Updated**: February 2026

## Summary

This folder contains downloaded GCP2 (Global Consciousness Project 2.0) random number generator data and analysis scripts. The goal is to replicate and extend Ulf Holmberg's research on correlations between RNG network anomalies and financial markets.

---

## Data Overview

### Downloaded Files (gitignored - stored on Google Drive)

**Device Data** (ZIP files in `devices/`):
- Google Drive: https://drive.google.com/drive/folders/1-z9hRcNrqrM4iH5xpkjG1ttgEZewmI9U?usp=sharing
- 473 devices total
- 451 devices have historical data (all past data)
- 471 devices have latest data (current month)
- 22 devices have no data on GCP2 servers
- Files: `device_{id}/*_History.csv.zip` and `device_{id}/*_Latest.csv.zip`

**Network Data** (ZIP files in `network/`):
- Google Drive: https://drive.google.com/drive/folders/1VmC1sCiHaXEn_Oqw81rvWQVCgG7j9Wul?usp=sharing
- 327 monthly files (March 2024 - present)
- Pre-March 2024: No network data exists on GCP2
- Structure: `network/{cluster_name}/{year}/*_{year}_{month}.csv.zip`

### Tracked Files

- `progress.json` - Download progress tracker
- `holmberg_analysis_merged.csv` - Merged GCP2 + market data from analysis
- `probe/` - Test downloads from initial exploration
- `gcp2_bulk_download.py` - Playwright browser automation to download data
- `gcp2_holmberg_analysis.py` - Holmberg methodology replication for GCP2

---

## GCP2 Network Coherence: Official Formula

From Plonka et al. (2026) "Correlations between onsite and global networks of random number generators during group healing meditations" (EXPLORE journal):

### Network Coherence (Amplitude) — Per Second

```
C(t) = (1 / √R(t)) × Σᵣ(Z²ᵣ,ₜ - 1)
```

Where:
- Each RNG outputs 10,000 bits/second, first 200 bits summed → expected μ=100, σ=√50≈7.07
- **Z** = (output - μ) / σ (normalized to standard normal)
- **Z² - 1** centers the chi-squared at 0 (mean of Z² is 1)
- **R(t)** = number of active RNGs at time t
- Sum across all RNGs, divide by √R for scaling

### Cumulative Network Coherence

```
Cumulative = Σₜ C(t)
```

The cumulative sum over time detects departure from randomness.

### What This Means for Our Code

The `network_coherence` column in GCP2 CSV files **is already the computed C(t)** — the per-second Network Coherence (Amplitude). Therefore:

| Approach | Correct? | Notes |
|----------|----------|-------|
| `cumsum(network_coherence)` | ✓ Yes | Directly gives cumulative Network Coherence |
| Rolling Z normalization | ✗ No | Double-processes an already-normalized metric |

### Distribution Properties (observed)

| Property | Expected (theory) | Observed |
|----------|-------------------|----------|
| Mean | 0 | ~0 |
| Variance | 2 | ~2.0 |
| Skewness | 2√2 ≈ 2.83 | ~2.8 |
| Kurtosis | 12 | ~12 |

These match chi-squared(1) shifted by -1, confirming the formula.

### GCP1 vs GCP2 Methodology Comparison

| Step | GCP1 (Stouffer method) | GCP2 (Amplitude method) |
|------|------------------------|-------------------------|
| 1 | Z per egg | Z per RNG |
| 2 | Combine: Stouffer Z = Σ(Z)/√N | Square each: Z² |
| 3 | Square: (Stouffer Z)² | Subtract 1: Z² - 1 |
| 4 | Subtract 1: (Stouffer Z)² - 1 | Combine: Σ(Z² - 1)/√N |
| 5 | Cumsum | Cumsum |

Both detect departure from randomness via chi-squared methodology, but combine individual RNG outputs differently.

---

## Holmberg Research Summary

### Publications

| Year | Title | Key Finding |
|------|-------|-------------|
| 2020 | Stock Returns and the Mind | Initial Max[Z]-market correlation |
| 2021 | Revisiting Stock Returns and the Mind | Validation with outlier removal |
| 2022 | Validating the GCP data hypothesis using internet search data | Google Trends validation |
| 2024 | A Novel Market Sentiment Measure | VIX correlation, out-of-sample 5-14% excess returns |
| 2025 | Can Consciousness Nudge Randomness? | Personal RNG experiment (forthcoming) |

### Holmberg's Max[Z] Calculation (GCP1)

```
1. Raw RNG data → 200 bits/second/device from ~70 devices
2. Z-scores: z = (observed - 100) / 7.0712  (published GCP values)
3. Stouffer Z per second: Z = Sum(z) / sqrt(N)
4. Daily Max[Z] = max(|Z|) over 24h window
```

### 2024 VIX Study Findings

- Max[Z] and VIX show significant relationship
- Model explains ~1% of VIX variance
- Out-of-sample portfolios beat baseline by 5-14% annually

---

## Replication Analysis Results (2016-2022)

Using GCP1 data (BigQuery) and Alpaca market data:

| Correlation | r | p-value |
|-------------|---|---------|
| Max[Z] vs VIX Level | -0.049 | 0.04 |
| Max[Z] vs VIX Change | -0.014 | 0.58 |
| Max[Z] vs SPY Return | -0.001 | 0.98 |

**Key finding**: No significant correlation in 2016-2022 data. However:
- GCP1 network degraded from 60-70 eggs to 23-42 eggs in this period
- Holmberg's strongest results were from 1999-2015 when network was at full strength
- Top 10% extreme days show borderline effect (p = 0.077)

---

## Scripts

### gcp2_bulk_download.py

Browser automation using Playwright to download from https://gcp2.net/data-results/data-download

```bash
# First time: opens browser for manual login
python3 gcp2_bulk_download.py --phase probe       # Test single download
python3 gcp2_bulk_download.py --phase device      # Download all device data
python3 gcp2_bulk_download.py --phase network     # Download all network data
python3 gcp2_bulk_download.py --phase both        # Download everything
python3 gcp2_bulk_download.py --scan              # Rebuild progress from files
```

### gcp2_holmberg_analysis.py

Adapts Holmberg methodology for GCP2's network_coherence metric.

```bash
python3 gcp2_holmberg_analysis.py --data-dir . --months 12
```

Key adaptations:
- Rolling Z-score normalization of network_coherence
- Fetches market data via yfinance (SPY, VIX)
- Permutation tests, lag analysis, threshold analysis

---

## CSV Format

### Network Data (Global Network)

```csv
epoch_time_utc,network_coherence,active_devices
1709251200,-0.5423,42
1709251201,0.2134,42
...
```

### Device Data

```csv
epoch_time_utc,device_coherence
1709251200,0.1234
1709251201,-0.0567
...
```

---

## Tasks for New Session

### Task 1: Rolling Windows Visualization (experiment 4)

Modify `/experiments/4-rolling-windows/gcp_egg_web_app.py` to:
1. Add GCP2 data overlay capability for same time periods
2. Toggle between GCP1, GCP2, or both
3. Select individual devices, clusters, or all
4. Display GCP2's network_coherence alongside GCP1's Stouffer Z
5. Proper mathematical comparison (normalize GCP2 nc to comparable scale)

The webapp currently uses BigQuery for GCP1 data. GCP2 data should be loaded from the ZIP files in this folder (assume they exist on the backend server).

### Task 2: Finance Correlations App (experiment 6)

Analyze and rebuild `/experiments/6-finance-correlations/`:
1. Ensure all Holmberg PDFs are present (2020, 2021, 2022, 2024 - check for 2025)
2. Enable both GCP1 and GCP2 data sources
3. Implement Holmberg's backtesting methodology
4. Reproduce his out-of-sample portfolio tests

Current issues: The app is "not very well functioning" - may need significant rebuild.

---

## Data Access Notes

- **GCP1 data**: Available via BigQuery (`gcpingcp` project, `eggs_us` dataset)
- **GCP2 data**: ZIP files in this folder (devices/ and network/ subfolders)
- **Market data**: Alpaca API (may return 503) or yfinance as fallback
- Python's `zipfile` module can read ZIPs directly into memory without extraction

---

## References

- GCP2.net: https://gcp2.net/
- GCP1 Historical: https://noosphere.princeton.edu/gcpdata.html
- Holmberg 2024: https://doi.org/10.1108/JES-11-2023-0663
- Plonka et al. 2026: https://doi.org/10.1016/j.explore.2025.103312 (Network Coherence formula)
- Global-mind.org recipes: https://global-mind.org/recipe.html
