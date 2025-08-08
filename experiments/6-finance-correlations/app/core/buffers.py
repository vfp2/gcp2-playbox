from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Generic, Iterable, List, Optional, Tuple, TypeVar


T = TypeVar("T")


def utc_now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


@dataclass
class TimedItem(Generic[T]):
    timestamp: float
    value: T


class CircularBuffer(Generic[T]):
    """Thread-safe circular buffer for timestamped items.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity: int = capacity
        self._buffer: Deque[TimedItem[T]] = deque(maxlen=capacity)
        self._lock = threading.RLock()

    def add(self, item: T, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else utc_now_ts()
        with self._lock:
            self._buffer.append(TimedItem(timestamp=ts, value=item))

    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def capacity(self) -> int:
        return self._capacity

    def snapshot(self) -> List[TimedItem[T]]:
        with self._lock:
            return list(self._buffer)

    def get_window(self, start_ts: float, end_ts: Optional[float] = None) -> List[TimedItem[T]]:
        end = end_ts if end_ts is not None else utc_now_ts()
        with self._lock:
            return [item for item in self._buffer if start_ts <= item.timestamp <= end]




