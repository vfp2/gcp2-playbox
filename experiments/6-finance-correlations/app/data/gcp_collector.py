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

    Per Scott Wilber (canon.yaml), field type 12 enumerates currently active
    eggs; type 13 provides per-egg samples keyed by Unix time.
    """

    def __init__(self, config: AppConfig, buffer: CircularBuffer[SensorSample]) -> None:
        self.config = config
        self.buffer = buffer
        self._state = GcpState(active_eggs=[])
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="GcpCollector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        session = requests.Session()
        session.headers.update({"User-Agent": "GCP2-RealTime-Predictor/0.1"})
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
            self._stop.wait(timeout=5)

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




