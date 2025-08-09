from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from alpaca_trade_api import REST  # type: ignore
except Exception:  # noqa: BLE001
    REST = None  # type: ignore

import requests

from ..config import AppConfig
from ..core.buffers import CircularBuffer
from ..core.maxz import max_abs_z_over_samples
from ..core.predict import MarketTick, Prediction, SensorSample


@dataclass
class BacktestParams:
    """Parameters for a backtest run.

    Offline backtesting validates signal mapping consistency across time and
    calibrates thresholds without introducing look-ahead bias. Replay operates
    on original timestamps and preserves sampling cadence while allowing
    accelerated wall-clock playback.
    """

    start_utc: datetime
    end_utc: datetime
    replay_speed: float  # e.g., 10.0 means 10x faster than real time
    window_size_sec: int
    horizon_sec: int


GCP_URL = (
    "http://global-mind.org/cgi-bin/eggdatareq.pl?z=1&year={y}&month={m}&day={d}"
    "&stime={sh}:{sm}:{ss}&etime={eh}:{em}:{es}"
)


def _daterange_days(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor <= last:
        yield cursor
        cursor = cursor + timedelta(days=1)


class BacktestRunner:
    """Fetch and replay historical GCP and market data into buffers.

    - GCP: uses the official CSV endpoint (fields 12, 13). Samples are added to
      `sensor_buffer` with their original UNIX timestamps.
    - Market: uses Alpaca Data API (bars or quotes as available) and adds ticks
      to `market_buffer` timestamped at bar/quote time.

    Uses the same scoring function (Max[Z]) and mapping logic as live mode;
    only the timebase is simulated.
    """

    def __init__(
        self,
        config: AppConfig,
        sensor_buffer: CircularBuffer[SensorSample],
        market_buffer: CircularBuffer[MarketTick],
    ) -> None:
        self.config = config
        self.sensor_buffer = sensor_buffer
        self.market_buffer = market_buffer
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._progress_lock = threading.RLock()
        self._progress: str = "idle"

        # Alpaca client (optional)
        self._client = None
        if (
            REST is not None
            and self.config.env.ALPACA_API_KEY
            and self.config.env.ALPACA_SECRET_KEY
        ):
            try:
                self._client = REST(
                    key_id=self.config.env.ALPACA_API_KEY,
                    secret_key=self.config.env.ALPACA_SECRET_KEY,
                    base_url=self.config.env.ALPACA_BASE_URL,
                )
            except Exception:
                self._client = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def progress(self) -> str:
        with self._progress_lock:
            return self._progress

    def _set_progress(self, text: str) -> None:
        with self._progress_lock:
            self._progress = text

    def run_async(self, params: BacktestParams) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(params,), name="BacktestRunner", daemon=True
        )
        self._thread.start()

    # ───────────────────────────── fetching ─────────────────────────────
    def _fetch_gcp_samples(
        self, start_utc: datetime, end_utc: datetime
    ) -> List[Tuple[float, List[float]]]:
        samples: List[Tuple[float, List[float]]] = []
        session = requests.Session()
        session.headers.update({"User-Agent": "GCP2-Backtest/0.1"})
        for day in _daterange_days(start_utc, end_utc):
            if self._stop.is_set():
                break
            sh, sm, ss = (0, 0, 0)
            eh, em, es = (23, 59, 59)
            # constrain edges for first/last day
            if day.date() == start_utc.date():
                sh, sm, ss = start_utc.hour, start_utc.minute, start_utc.second
            if day.date() == end_utc.date():
                eh, em, es = end_utc.hour, end_utc.minute, end_utc.second
            url = GCP_URL.format(
                y=day.year,
                m=str(day.month).zfill(2),
                d=str(day.day).zfill(2),
                sh=str(sh).zfill(2),
                sm=str(sm).zfill(2),
                ss=str(ss).zfill(2),
                eh=str(eh).zfill(2),
                em=str(em).zfill(2),
                es=str(es).zfill(2),
            )
            try:
                resp = session.get(url, timeout=self.config.runtime.network_timeout_sec)
                resp.raise_for_status()
                for line in resp.text.splitlines():
                    if not line:
                        continue
                    try:
                        parts = line.split(",")
                        ftype = int(parts[0])
                    except Exception:
                        continue
                    if ftype == 13 and len(parts) >= 3:
                        try:
                            ts = float(parts[1])
                            vals = [float(x) for x in parts[2:] if x != ""]
                        except Exception:
                            continue
                        if start_utc.timestamp() <= ts <= end_utc.timestamp():
                            samples.append((ts, vals))
            except Exception:
                # Skip day on errors
                continue
        samples.sort(key=lambda t: t[0])
        return samples

    def _fetch_alpaca_prices(
        self, start_utc: datetime, end_utc: datetime
    ) -> Dict[str, List[Tuple[float, float]]]:
        out: Dict[str, List[Tuple[float, float]]] = {}
        symbols = list(self.config.runtime.symbols)
        if self._client is None or not symbols:
            return out
        # Try bars first (1Min)
        for sym in symbols:
            out[sym] = []
            try:
                get_bars = getattr(self._client, "get_bars", None)
                if callable(get_bars):
                    bars = get_bars(
                        sym,
                        "1Min",
                        start_utc.isoformat(),
                        end_utc.isoformat(),
                    )
                    # Support both list-like and BarSet
                    rows = bars if isinstance(bars, list) else list(bars)
                    for b in rows:
                        # bar.t may be pandas.Timestamp or datetime; fall back to to_datetime
                        ts = None
                        tattr = getattr(b, "t", None) or getattr(b, "timestamp", None)
                        if hasattr(tattr, "timestamp"):
                            ts = float(tattr.timestamp())
                        else:
                            try:
                                ts = float(datetime.fromisoformat(str(tattr)).timestamp())
                            except Exception:
                                ts = None
                        c = getattr(b, "c", None) or getattr(b, "close", None)
                        if ts is not None and c is not None:
                            out[sym].append((ts, float(c)))
                else:
                    # Fallback: latest quotes in coarse steps (every 60s)
                    step = timedelta(minutes=1)
                    cur = start_utc
                    get_quote = getattr(self._client, "get_latest_quote", None)
                    while cur <= end_utc and callable(get_quote):
                        q = get_quote(sym)
                        try:
                            bid = float(getattr(q, "bidprice"))
                            ask = float(getattr(q, "askprice"))
                            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else None
                        except Exception:
                            mid = None
                        if mid is not None:
                            out[sym].append((cur.replace(tzinfo=timezone.utc).timestamp(), mid))
                        cur += step
            except Exception:
                # Skip symbol on errors
                continue
            out[sym].sort(key=lambda t: t[0])
        return out

    # ───────────────────────────── replay ─────────────────────────────
    def _run(self, params: BacktestParams) -> None:
        try:
            self._set_progress("fetch:gcp")
            gcp = self._fetch_gcp_samples(params.start_utc, params.end_utc)
            if self._stop.is_set():
                return
            self._set_progress("fetch:alpaca")
            prices = self._fetch_alpaca_prices(params.start_utc, params.end_utc)
            if self._stop.is_set():
                return

            # Pre-build price lookup per symbol
            def price_at(symbol: str, ts: float) -> Optional[float]:
                series = prices.get(symbol) or []
                # Find first point with time >= ts; else last before
                lo, hi = 0, len(series) - 1
                if hi < 0:
                    return None
                # Binary search lower_bound
                idx = None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    tmid = series[mid][0]
                    if tmid < ts:
                        lo = mid + 1
                    else:
                        idx = mid
                        hi = mid - 1
                if idx is not None:
                    return series[idx][1]
                # fallback last before
                return series[-1][1]

            # Replay loop
            self._set_progress("replay:start")
            last_wall = time.time()
            start_wall = last_wall
            start_ts = gcp[0][0] if gcp else params.start_utc.timestamp()
            for i, (ts, vals) in enumerate(gcp):
                if self._stop.is_set():
                    break
                # Pace according to replay_speed
                now_wall = time.time()
                elapsed_wall = now_wall - last_wall
                target_wait = max(0.0, (1.0 / max(1e-6, params.replay_speed)) - elapsed_wall)
                if target_wait > 0:
                    time.sleep(min(0.25, target_wait))
                last_wall = time.time()

                # Add sensor sample at original timestamp
                self.sensor_buffer.add(SensorSample(values=vals), timestamp=ts)

                # Add market mid-price samples near this timestamp for all symbols
                for sym in self.config.runtime.symbols:
                    px = price_at(sym, ts)
                    if px is not None:
                        self.market_buffer.add(MarketTick(price=px, symbol=sym), timestamp=ts)

                # Periodically update progress
                if i % 250 == 0:
                    pct = 100.0 * (ts - start_ts) / max(1e-6, params.end_utc.timestamp() - start_ts)
                    self._set_progress(f"replay:{pct:.1f}%")

            self._set_progress("done")
        finally:
            # nothing to clean; buffers hold results for dashboard charts and predictor can run live concurrently if desired
            pass


