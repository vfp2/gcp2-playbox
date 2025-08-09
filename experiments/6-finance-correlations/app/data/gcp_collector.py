from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from io import StringIO
from typing import Dict, List, Optional

import requests

from ..config import AppConfig
from ..core.buffers import CircularBuffer
from ..core.predict import SensorSample


GCP_URL = (
    "http://global-mind.org/cgi-bin/eggdatareq.pl?z=1&year={y}&month={m}&day={d}"
    "&stime={sh}:{sm}:{ss}&etime={eh}:{em}:{es}"
)


@dataclass
class GcpState:
    active_eggs: List[int]


class GcpCollector:
    """Collector for GCP CSV endpoint with field types 12 and 13.
    
    Backtesting extension: provide optional `start_ts` and `end_ts` to
    `start()` to replay historical samples quickly. During backtest, we
    paginate by short windows and advance until `end_ts`, emitting samples
    with their original timestamps into the buffer.
    """

    def __init__(self, config: AppConfig, buffer: CircularBuffer[SensorSample]) -> None:
        self.config = config
        self.buffer = buffer
        self._state = GcpState(active_eggs=[])
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, start_ts: Optional[float] = None, end_ts: Optional[float] = None, realtime_interval_sec: int = 5) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Stash backtest window on the instance for the thread
        self._bt_start_ts = start_ts  # type: ignore[attr-defined]
        self._bt_end_ts = end_ts      # type: ignore[attr-defined]
        self._rt_interval = realtime_interval_sec  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._run, name="GcpCollector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        session = requests.Session()
        session.headers.update({"User-Agent": "GCP2-RealTime-Predictor/0.1"})
        # Determine mode
        bt_start = getattr(self, "_bt_start_ts", None)
        bt_end = getattr(self, "_bt_end_ts", None)
        realtime_interval = getattr(self, "_rt_interval", 5)

        if bt_start is not None and bt_end is not None and bt_start < bt_end:
            # Backtest mode: iterate from start to end in minute chunks
            cursor = datetime.fromtimestamp(bt_start, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(bt_end, tz=timezone.utc)
            while not self._stop.is_set() and cursor < end_dt:
                try:
                    y, m, d = cursor.year, cursor.month, cursor.day
                    # Pull a 60s window from cursor
                    start_dt = cursor
                    end_window = min(cursor + timedelta(seconds=60), end_dt)
                    url = GCP_URL.format(
                        y=y,
                        m=str(m).zfill(2),
                        d=str(d).zfill(2),
                        sh=str(start_dt.hour).zfill(2),
                        sm=str(start_dt.minute).zfill(2),
                        ss=str(start_dt.second).zfill(2),
                        eh=str(end_window.hour).zfill(2),
                        em=str(end_window.minute).zfill(2),
                        es=str(end_window.second).zfill(2),
                    )
                    resp = session.get(url, timeout=self.config.runtime.network_timeout_sec)
                    resp.raise_for_status()
                    self._parse_csv(resp.text)
                except Exception:
                    pass
                # Advance cursor fast (e.g., 10 minutes per real second)
                cursor += timedelta(seconds=60)
                self._stop.wait(timeout=0.1)
            return

        # Real-time mode
        while not self._stop.is_set():
            try:
                now = datetime.now(timezone.utc)
                y, m, d = now.year, now.month, now.day
                # Pull a short window ending now
                end = now
                start = now - timedelta(seconds=60)
                url = GCP_URL.format(
                    y=y,
                    m=str(m).zfill(2),
                    d=str(d).zfill(2),
                    sh=str(start.hour).zfill(2),
                    sm=str(start.minute).zfill(2),
                    ss=str(start.second).zfill(2),
                    eh=str(end.hour).zfill(2),
                    em=str(end.minute).zfill(2),
                    es=str(end.second).zfill(2),
                )
                resp = session.get(url, timeout=self.config.runtime.network_timeout_sec)
                resp.raise_for_status()
                self._parse_csv(resp.text)
            except Exception:
                # Skip errors; continue loop
                pass
            self._stop.wait(timeout=realtime_interval)

    def _parse_csv(self, text: str) -> None:
        reader = csv.reader(StringIO(text))
        for row in reader:
            if not row:
                continue
            try:
                field_type = int(row[0])
            except Exception:
                continue
            if field_type == 12:
                self._parse_active_eggs(row)
            elif field_type == 13:
                self._parse_sample(row)

    def _parse_active_eggs(self, row: List[str]) -> None:
        # Format: 12,egg1,egg2,... eggN
        eggs: List[int] = []
        for token in row[1:]:
            try:
                eggs.append(int(token))
            except Exception:
                continue
        self._state.active_eggs = eggs

    def _parse_sample(self, row: List[str]) -> None:
        # Format: 13,unix_ts,v1,v2,... per-egg values
        if len(row) < 3:
            return
        try:
            ts = float(row[1])
            values = [float(x) for x in row[2:] if x != ""]
        except Exception:
            return
        self.buffer.add(SensorSample(values=values), timestamp=ts)

    # ───────────────────────────── accessors ─────────────────────────────
    def get_active_eggs(self) -> List[str]:
        """Return current active egg identifiers as user-friendly names.

        Example: ["egg_1", "egg_2", ...]
        """
        try:
            return [f"egg_{e}" for e in list(self._state.active_eggs)]
        except Exception:
            return []




