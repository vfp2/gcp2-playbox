from __future__ import annotations

from app.core.buffers import CircularBuffer
from app.core.predict import SensorSample
from app.data.gcp_collector import GcpCollector
from app.config import AppConfig, RuntimeConfig


def make_config() -> AppConfig:
    runtime = RuntimeConfig()
    cfg = AppConfig(env=type("E", (), {"DASH_HOST": "", "DASH_PORT": 0, "LOG_LEVEL": "INFO", "ALPACA_API_KEY": None, "ALPACA_SECRET_KEY": None, "ALPACA_BASE_URL": ""})(), runtime=runtime)  # type: ignore
    return cfg


def test_parse_active_and_samples() -> None:
    buf: CircularBuffer[SensorSample] = CircularBuffer(capacity=10)
    collector = GcpCollector(make_config(), buf)
    text = "\n".join([
        "12,1,2,3",
        "13,1000,100,101,102",
        "13,1001,99,98,100",
    ])
    collector._parse_csv(text)  # type: ignore[attr-defined]
    snaps = buf.snapshot()
    assert len(snaps) == 2
    assert snaps[0].timestamp == 1000
    assert snaps[0].value.values == [100.0, 101.0, 102.0]




