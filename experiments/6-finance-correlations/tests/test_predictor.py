from __future__ import annotations

from app.config import AppConfig, RuntimeConfig
from app.core.buffers import CircularBuffer
from app.core.predict import MarketTick, Predictor, SensorSample
from app.core.tracker import PredictionTracker


def make_config() -> AppConfig:
    runtime = RuntimeConfig()
    cfg = AppConfig(env=type("E", (), {"DASH_HOST": "", "DASH_PORT": 0, "LOG_LEVEL": "INFO", "SIMULATE": True, "ALPACA_API_KEY": None, "ALPACA_SECRET_KEY": None, "ALPACA_BASE_URL": ""})(), runtime=runtime)  # type: ignore
    return cfg


def test_predictor_threshold_mapping() -> None:
    cfg = make_config()
    cfg.runtime.thresholds.up_threshold = 1.0
    cfg.runtime.thresholds.down_threshold = 1.0
    cfg.runtime.thresholds.min_confidence = 0.5
    sb: CircularBuffer[SensorSample] = CircularBuffer(capacity=100)
    mb: CircularBuffer[MarketTick] = CircularBuffer(capacity=100)
    tracker = PredictionTracker()
    pred = Predictor(cfg, sb, mb, tracker.record)

    # Add sensor data that yields score ~1.0
    sb.add(SensorSample(values=[100.0, 107.0712]))
    # Run one iteration of mapping directly
    direction, conf = pred._map_score(1.0)  # noqa: SLF001
    assert direction in {"UP", "HOLD"}
    assert conf >= 0.5




