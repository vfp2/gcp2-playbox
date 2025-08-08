"""GCP Real-Time Market Predictor package.

Per Scott Wilber (see canon.yaml), directional signals are derived from anomaly
intensity in Global Consciousness Project sensor aggregates. This package
implements a configurable, real-time pipeline to compute Max[Z] anomaly scores,
map them to market direction predictions, and visualize live performance.
"""

__all__ = [
    "config",
    "core",
    "data",
    "web",
    "utils",
    "simulators",
]




