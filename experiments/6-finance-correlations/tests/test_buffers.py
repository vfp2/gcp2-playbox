from __future__ import annotations

import time

from app.core.buffers import CircularBuffer


def test_circular_buffer_basic() -> None:
    buf: CircularBuffer[int] = CircularBuffer(capacity=3)
    buf.add(1, timestamp=1.0)
    buf.add(2, timestamp=2.0)
    buf.add(3, timestamp=3.0)
    assert buf.size() == 3
    buf.add(4, timestamp=4.0)
    assert buf.size() == 3
    snaps = buf.snapshot()
    assert [x.value for x in snaps] == [2, 3, 4]


def test_get_window() -> None:
    buf: CircularBuffer[int] = CircularBuffer(capacity=10)
    for i in range(10):
        buf.add(i, timestamp=float(i))
    w = buf.get_window(3.0, 6.0)
    assert [x.value for x in w] == [3, 4, 5, 6]




