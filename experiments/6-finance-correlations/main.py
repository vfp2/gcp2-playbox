from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Optional

import typer

from app.config import AppConfig, load_config
from app.core.buffers import CircularBuffer
from app.core.predict import MarketTick, Predictor, SensorSample
from app.core.tracker import PredictionTracker
from app.data.gcp_collector import GcpCollector
from app.data.market_collector import MarketCollector
from app.utils.logging import setup_logging
from app.web.server import serve as serve_web


app = typer.Typer(add_completion=False)


@app.command()
def serve(host: Optional[str] = typer.Option(None), port: Optional[int] = typer.Option(None)) -> None:
    cfg = load_config()
    setup_logging(cfg.env.LOG_LEVEL)
    serve_web(host=host, port=port)


@app.command()
def run(log_level: str = typer.Option("INFO")) -> None:
    cfg = load_config()
    setup_logging(log_level)

    sensor_buffer: CircularBuffer[SensorSample] = CircularBuffer(capacity=cfg.runtime.sensor_buffer_size)
    market_buffer: CircularBuffer[MarketTick] = CircularBuffer(capacity=cfg.runtime.market_buffer_size)
    tracker = PredictionTracker()
    predictor = Predictor(cfg, sensor_buffer, market_buffer, tracker.record)
    gcp = GcpCollector(cfg, sensor_buffer)
    market = MarketCollector(cfg, market_buffer)

    stop_event = threading.Event()

    def handle_signal(signum, frame):  # noqa: ANN001, D401
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    gcp.start()
    market.start()
    predictor.start()
    typer.echo("Running. Press Ctrl+C to stop.")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        predictor.stop()
        gcp.stop()
        market.stop()


@app.command()
def backtest(
    start_utc: str = typer.Argument(..., help="ISO8601 UTC start, e.g. 2024-08-01T13:30:00"),
    end_utc: str = typer.Argument(..., help="ISO8601 UTC end, e.g. 2024-08-01T20:00:00"),
    speed: float = typer.Option(20.0, help="Replay speed multiplier (e.g., 20x)"),
) -> None:
    """Run headless backtest that replays historical GCP and Alpaca market data.

    Uses identical Max[Z] calculation and mapping thresholds as live mode; only
    the clock is accelerated. Results are emitted into the same buffers used by
    the dashboard and predictor.
    """
    from app.config import load_config
    from app.core.buffers import CircularBuffer
    from app.core.predict import MarketTick, SensorSample, Predictor
    from app.core.tracker import PredictionTracker
    from app.data.backtest import BacktestParams, BacktestRunner

    from datetime import datetime, timezone

    cfg = load_config()
    sb: CircularBuffer[SensorSample] = CircularBuffer(cfg.runtime.sensor_buffer_size)
    mb: CircularBuffer[MarketTick] = CircularBuffer(cfg.runtime.market_buffer_size)

    tracker = PredictionTracker()
    predictor = Predictor(cfg, sb, mb, tracker.record)

    runner = BacktestRunner(cfg, sb, mb)
    s = datetime.fromisoformat(start_utc).replace(tzinfo=timezone.utc)
    e = datetime.fromisoformat(end_utc).replace(tzinfo=timezone.utc)
    params = BacktestParams(
        start_utc=s,
        end_utc=e,
        replay_speed=max(1.0, float(speed)),
        window_size_sec=cfg.runtime.method.window_size,
        horizon_sec=cfg.runtime.horizon_sec,
    )

    predictor.start()
    runner.run_async(params)
    try:
        # Wait until done
        while runner.is_running():
            typer.echo(f"Progress: {runner.progress()}")
            time.sleep(1.0)
    finally:
        predictor.stop()
    # Print brief stats
    stats = tracker.stats()
    for sym, st in stats.items():
        typer.echo(f"{sym}: total={st.total} correct={st.correct} acc={st.accuracy:.2%}")



if __name__ == "__main__":
    app()




