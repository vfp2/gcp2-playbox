from __future__ import annotations

import random
import time
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")


def exponential_backoff(
    attempt: int,
    base_seconds: float,
    cap_seconds: float,
) -> float:
    return min(cap_seconds, base_seconds * (2 ** max(0, attempt - 1)))


def with_retries(
    func: Callable[[], T],
    max_attempts: int,
    base_seconds: float,
    cap_seconds: float,
    jitter_fraction: float = 0.1,
) -> T:
    attempt = 1
    while True:
        try:
            return func()
        except Exception:  # noqa: BLE001
            if attempt >= max_attempts:
                raise
            delay = exponential_backoff(attempt, base_seconds, cap_seconds)
            jitter = delay * jitter_fraction * (2 * random.random() - 1)
            time.sleep(max(0.0, delay + jitter))
            attempt += 1




