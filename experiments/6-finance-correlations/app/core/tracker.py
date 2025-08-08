from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from .predict import Prediction


@dataclass
class SymbolStats:
    total: int = 0
    correct: int = 0
    up: int = 0
    down: int = 0

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0


class PredictionTracker:
    """Track predictions and mark outcomes when horizon elapses."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._predictions: List[Prediction] = []
        self._stats: Dict[str, SymbolStats] = defaultdict(SymbolStats)

    def record(self, p: Prediction) -> None:
        with self._lock:
            self._predictions.append(p)
            s = self._stats[p.symbol]
            s.total += 1
            if p.direction == "UP":
                s.up += 1
            elif p.direction == "DOWN":
                s.down += 1

    def try_resolve(self, symbol: str, price_then: float, price_now: float, now_ts: float) -> None:
        with self._lock:
            for p in self._predictions:
                if p.symbol != symbol or p.resolved:
                    continue
                if now_ts - p.timestamp >= p.horizon_sec:
                    # For 60s horizons, use percentage change threshold to avoid noise
                    # Small movements (<0.1%) are considered neutral
                    price_change_pct = (price_now - price_then) / price_then * 100
                    
                    if abs(price_change_pct) < 0.1:
                        # Very small movement - mark as neutral/incorrect
                        actual = "HOLD"
                        p.correct = False  # Neutral movements don't count as correct
                    else:
                        # Significant movement - use direction
                        actual = "UP" if price_change_pct > 0 else "DOWN"
                        p.correct = (p.direction == actual)
                    
                    p.actual_direction = actual
                    p.resolved = True
                    if p.correct:
                        self._stats[symbol].correct += 1

    def recent(self, limit: int = 100) -> List[Prediction]:
        with self._lock:
            return list(self._predictions[-limit:])

    def stats(self) -> Dict[str, SymbolStats]:
        with self._lock:
            return {k: SymbolStats(**vars(v)) for k, v in self._stats.items()}




