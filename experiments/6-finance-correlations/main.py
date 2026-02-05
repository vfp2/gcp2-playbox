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


app = typer.Typer(add_completion=False)


@app.command()
def serve(host: Optional[str] = typer.Option(None), port: Optional[int] = typer.Option(None)) -> None:
    cfg = load_config()
    setup_logging(cfg.env.LOG_LEVEL)
    # Serve the portal on a single port; the dashboard is mounted under /experiment-6/
    from gcp_experiments_portal import app as portal_app
    h = host or cfg.env.DASH_HOST
    p = cfg.env.DASH_PORT if port is None else port
    typer.echo(f"Serving portal on http://{h}:{p} (dashboard at /experiment-6/)")
    portal_app.run(host=h, port=p, debug=False)


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


if __name__ == "__main__":
    app()




