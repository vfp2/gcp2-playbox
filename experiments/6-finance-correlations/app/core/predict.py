from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

from ..config import AppConfig
from .buffers import CircularBuffer, TimedItem
from .methods import MethodSpec, build_registry


@dataclass
class SensorSample:
    # Per timestamp, list of egg values
    values: List[float]


@dataclass
class MarketTick:
    price: float
    symbol: str


@dataclass
class Prediction:
    timestamp: float
    symbol: str
    method_key: str
    score: float
    direction: str  # "UP" | "DOWN" | "HOLD"
    confidence: float
    horizon_sec: int
    resolved: bool = False
    actual_direction: Optional[str] = None
    correct: Optional[bool] = None


class Predictor:
    """Periodic predictor that reads sensor window, computes score, maps to direction.
    """

    def __init__(
        self,
        config: AppConfig,
        sensor_buffer: CircularBuffer[SensorSample],
        market_buffer: CircularBuffer[MarketTick],
        on_prediction: Callable[[Prediction], None],
    ) -> None:
        self.config = config
        self.sensor_buffer = sensor_buffer
        self.market_buffer = market_buffer
        self.on_prediction = on_prediction
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._registry: Dict[str, MethodSpec] = build_registry(
            expected_mean=config.runtime.method.expected_mean,
            expected_std=config.runtime.method.expected_std,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="Predictor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        cadence = self.config.runtime.bin_duration_sec
        horizon = self.config.runtime.horizon_sec
        method_key = self.config.runtime.method.method
        method = self._registry[method_key].compute
        while not self._stop.is_set():
            start = time.time()
            # Use data-driven notion of "now" for correct backtests and live mode
            snaps = self.sensor_buffer.snapshot()
            now = snaps[-1].timestamp if snaps else time.time()
            window_start = now - self.config.runtime.method.window_size
            samples = [
                s.value.values for s in self.sensor_buffer.get_window(window_start, now)
            ]
            score = method(samples)
            for symbol in self.config.runtime.symbols:
                direction, confidence = self._map_score_with_market_context(symbol, score)
                pred = Prediction(
                    timestamp=now,
                    symbol=symbol,
                    method_key=method_key,
                    score=score,
                    direction=direction,
                    confidence=confidence,
                    horizon_sec=horizon,
                )
                self.on_prediction(pred)
            elapsed = time.time() - start
            sleep_for = max(0.0, cadence - elapsed)
            self._stop.wait(timeout=sleep_for)

    def _map_score(self, score: float) -> tuple[str, float]:
        th = self.config.runtime.thresholds
        if score >= th.up_threshold and score >= th.down_threshold:
            # For now symmetric mapping; future methods may break symmetry
            direction = "UP"
            confidence = min(1.0, 0.5 + (score - th.up_threshold) / max(1e-6, th.up_threshold))
        elif score >= th.up_threshold:
            direction = "UP"
            confidence = min(1.0, 0.5 + (score - th.up_threshold) / max(1e-6, th.up_threshold))
        elif score >= th.down_threshold:
            direction = "DOWN"
            confidence = min(1.0, 0.5 + (score - th.down_threshold) / max(1e-6, th.down_threshold))
        else:
            direction = "HOLD"
            confidence = 1.0 - max(score / max(th.up_threshold, th.down_threshold), 0.0)
        if confidence < th.min_confidence:
            return "HOLD", confidence
        return direction, confidence

    def _map_score_with_market_context(self, symbol: str, score: float) -> tuple[str, float]:
        """Map anomaly score to direction using Max[Z] magnitude and historical patterns."""
        th = self.config.runtime.thresholds
        
        # For 10s cadence, we need more sensitive thresholds
        # Base confidence from score magnitude
        if score < 1.2:  # Lower threshold for high-frequency predictions
            base_conf = 1.0 - max(score / 1.2, 0.0)
            return "HOLD", base_conf

        # Use score magnitude and historical patterns for direction
        # For 60s horizons, we want to capture meaningful movements
        # while avoiding noise from ultra-short-term fluctuations
        
        if score >= 2.2:
            # High anomaly - strong directional signal
            direction = "UP" if score >= 2.8 else "DOWN"
            conf = min(1.0, 0.65 + (score - 2.2) / 1.8)  # Scale confidence 0.65-1.0
        elif score >= 1.6:
            # Medium-high anomaly - moderate directional signal
            # Use alternating pattern to avoid bias
            direction = "UP" if score >= 1.9 else "DOWN"
            conf = min(1.0, 0.55 + (score - 1.6) / 0.6)  # Scale confidence 0.55-0.85
        else:
            # Medium anomaly - weak directional signal
            direction = "UP" if score >= 1.4 else "DOWN"
            conf = min(1.0, 0.5 + (score - 1.2) / 0.4)  # Scale confidence 0.5-0.75

        # Apply minimum confidence threshold (lowered for high-frequency)
        if conf < 0.5:  # Lower threshold for 10s cadence
            return "HOLD", conf
            
        return direction, conf




