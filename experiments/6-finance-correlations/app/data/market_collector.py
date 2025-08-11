from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Tuple
import math
import hashlib
import re

try:
    from alpaca_trade_api import REST  # type: ignore
except Exception:  # noqa: BLE001
    REST = None  # type: ignore

from ..config import AppConfig
from ..core.buffers import CircularBuffer
from ..core.predict import MarketTick


@dataclass
class PriceState:
    last_price: Dict[str, float]


@dataclass
class ExchangeInfo:
    symbol: str
    exchange: str
    is_open: bool
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    trading_hours: str = "9:30 AM - 4:00 PM ET"


class MarketCollector:
    """Collect market prices from Alpaca.

    Backtesting extension: start(start_ts, end_ts, speed) replays historical
    prices between the timestamps at accelerated speed. Per Scott Wilber
    (canon.yaml), design prioritizes determinism and low latency; we fetch
    bars up-front and stream them in timestamp order, avoiding jitter.
    """

    def __init__(self, config: AppConfig, buffer: CircularBuffer[MarketTick]) -> None:
        self.config = config
        self.buffer = buffer
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = PriceState(last_price={})
        self._exchange_info: Dict[str, ExchangeInfo] = {}
        
        # No simulation parameters; relies on live market data

        self._client = None
        if REST is not None and self.config.env.ALPACA_API_KEY and self.config.env.ALPACA_SECRET_KEY:
            self._client = REST(
                key_id=config.env.ALPACA_API_KEY,
                secret_key=config.env.ALPACA_SECRET_KEY,
                base_url=config.env.ALPACA_BASE_URL,
            )
        # Validation client
        self._validation_client = None
        if REST is not None and config.env.ALPACA_API_KEY and config.env.ALPACA_SECRET_KEY:
            try:
                self._validation_client = REST(
                    key_id=config.env.ALPACA_API_KEY,
                    secret_key=config.env.ALPACA_SECRET_KEY,
                    base_url=config.env.ALPACA_BASE_URL,
                )
            except Exception:
                self._validation_client = None
        
        # Initialize exchange info
        self._init_exchange_info()

    def start(self, start_ts: Optional[float] = None, end_ts: Optional[float] = None, speed: float = 60.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Backtest parameters (if provided)
        self._bt_start_ts = start_ts  # type: ignore[attr-defined]
        self._bt_end_ts = end_ts      # type: ignore[attr-defined]
        self._bt_speed = float(speed) if speed and speed > 0 else 60.0  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._run, name="MarketCollector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        bt_start = getattr(self, "_bt_start_ts", None)
        bt_end = getattr(self, "_bt_end_ts", None)
        bt_speed = getattr(self, "_bt_speed", 60.0)

        if bt_start is not None and bt_end is not None and bt_start < bt_end:
            # Backtest mode
            try:
                self._run_backtest(bt_start, bt_end, bt_speed)
            except Exception:
                # Fall back to idle if backtest fails
                pass
            return

        # Live mode
        while not self._stop.is_set():
            try:
                # Update market status
                self._update_market_status()
                for symbol in self.config.runtime.symbols:
                    price = self._get_latest_price(symbol)
                    if price is not None:
                        self._state.last_price[symbol] = price
                        self.buffer.add(MarketTick(price=price, symbol=symbol))
            except Exception:
                pass
            self._stop.wait(timeout=1)

    # ───────────────────────────── backtest helpers ─────────────────────────────
    def _run_backtest(self, start_ts: float, end_ts: float, speed: float) -> None:
        """Fetch historical bars and replay as ticks between start/end.

        We aim for predictable playback. Sleep per step is scaled by `speed`:
        a 60-second bar sleeps ~1.0/speed seconds (e.g., speed=600 → 0.1s/bar).

        Per Scott Wilber (canon.yaml), batch-fetch upfront to avoid runtime
        stalls and stream deterministically.
        """
        if self._client is None:
            return
        from datetime import datetime as _dt
        import time as _time
        import pytz  # type: ignore

        start_dt = _dt.fromtimestamp(start_ts, tz=timezone.utc)
        end_dt = _dt.fromtimestamp(end_ts, tz=timezone.utc)

        # Prefer minute bars; fallback to barset for older clients
        def _fetch_bars(sym: str):
            try:
                get_bars = getattr(self._client, "get_bars", None)
                if callable(get_bars):
                    # Alpaca v2 style
                    return list(get_bars(sym, "1Min", start_dt, end_dt))
                # Older API
                get_barset = getattr(self._client, "get_barset", None)
                if callable(get_barset):
                    bs = get_barset(sym, "minute", start=start_dt, end=end_dt)
                    return list(bs[sym]) if bs and sym in bs else []
            except Exception:
                return []
            return []

        # Build per-symbol iterators
        symbol_to_bars: Dict[str, List] = {}
        for s in self.config.runtime.symbols:
            symbol_to_bars[s] = _fetch_bars(s)

        # Merge by time in a simple round-robin per symbol; if timestamps exist use them
        # For v2 bars, bar.t is a pandas.Timestamp; for older, bar.t is datetime
        # Normalize to POSIX seconds
        def _bar_ts(bar) -> float:
            t = getattr(bar, "t", None) or getattr(bar, "time", None)
            try:
                if hasattr(t, "timestamp"):
                    return float(t.timestamp())
                # pandas Timestamp
                return float(_dt.fromisoformat(str(t)).replace(tzinfo=pytz.UTC).timestamp())
            except Exception:
                return 0.0

        def _bar_price(bar) -> Optional[float]:
            for attr in ("c", "close", "price"):
                v = getattr(bar, attr, None)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        continue
            return None

        # Flatten all bars into a single list
        merged: List[Tuple[float, str, float]] = []
        for sym, bars in symbol_to_bars.items():
            for b in bars:
                ts = _bar_ts(b)
                pr = _bar_price(b)
                if pr is not None and start_ts <= ts <= end_ts:
                    merged.append((ts, sym, pr))
        merged.sort(key=lambda x: x[0])

        if not merged:
            return

        # Playback
        # Compute average step to scale sleep (fallback 60s)
        avg_step = 60.0
        if len(merged) >= 2:
            deltas = [merged[i+1][0] - merged[i][0] for i in range(len(merged)-1)]
            good = [d for d in deltas if d > 0]
            if good:
                avg_step = sum(good) / len(good)
        sleep_per_step = max(0.005, avg_step / max(1.0, speed))

        for ts, sym, pr in merged:
            if self._stop.is_set():
                break
            try:
                self._state.last_price[sym] = pr
                self.buffer.add(MarketTick(price=pr, symbol=sym), timestamp=ts)
            except Exception:
                pass
            # Accelerated wait
            self._stop.wait(timeout=sleep_per_step)

    def _update_market_status(self) -> None:
        """Update market status for all symbols."""
        is_open = self._is_market_open()
        for symbol in self.config.runtime.symbols:
            if symbol in self._exchange_info:
                self._exchange_info[symbol].is_open = is_open

    def _get_latest_price(self, symbol: str) -> Optional[float]:
        if self._client is None:
            return None
        try:
            t = self._client.get_latest_trade(symbol)
            px = getattr(t, "price", None)
            if px is not None and float(px) > 0:
                return float(px)
        except Exception:
            return None
        return None

    def _init_exchange_info(self) -> None:
        """Initialize exchange information for all symbols.
        
        Note: This is a simplified implementation. For full Alpaca API integration:
        1. Use self._client.get_asset(symbol) to get real exchange info
        2. Use self._client.get_clock() to get real market status
        3. Use pytz for proper timezone handling
        4. Handle holidays and special market hours
        """
        for symbol in self.config.runtime.symbols:
            # Default exchange info (NYSE/NASDAQ for most ETFs)
            exchange = "NYSE" if symbol in ["SPY", "IVV", "VOO", "VXX"] else "NASDAQ"
            
            # Try to get real exchange info if available
            if self._client is not None:
                try:
                    # TODO: Implement real Alpaca API calls
                    # asset = self._client.get_asset(symbol)
                    # exchange = asset.exchange if hasattr(asset, 'exchange') else exchange
                    pass
                except Exception:
                    pass
            
            self._exchange_info[symbol] = ExchangeInfo(
                symbol=symbol,
                exchange=exchange,
                is_open=self._is_market_open(),
                trading_hours="9:30 AM - 4:00 PM ET"
            )

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        # Use Alpaca clock if available; otherwise, report closed.
        if self._client is None:
            return False
        try:
            clock = getattr(self._client, "get_clock", None)
            if callable(clock):
                c = clock()
                is_open = getattr(c, "is_open", None)
                return bool(is_open) if isinstance(is_open, bool) else False
        except Exception:
            return False
        return False

    def get_exchange_info(self, symbol: str) -> Optional[ExchangeInfo]:
        """Get exchange information for a symbol."""
        return self._exchange_info.get(symbol)

    def get_all_exchange_info(self) -> Dict[str, ExchangeInfo]:
        """Get exchange information for all symbols."""
        return self._exchange_info.copy()

    # ───────────────────────────── dynamic symbols ─────────────────────────────
    def add_symbol(self, symbol: str) -> bool:
        """Validate and add a new symbol at runtime.

        Returns True if added, False if invalid/not found.

        Validate via Alpaca if available; otherwise reject.

        Per Scott Wilber (canon.yaml), we prioritize deterministic, low-latency
        operation; dynamic additions should be validated quickly and integrated
        without disrupting ongoing calculations.
        """
        s = (symbol or "").strip().upper()
        if not s:
            return False
        if s in self.config.runtime.symbols:
            return True

        # Basic validation: letters, numbers, dots/hyphens common in tickers
        if not re.fullmatch(r"[A-Z][A-Z0-9\.\-]{0,9}", s):
            return False

        exists = True
        asset_exch: Optional[str] = None
        client = self._client or self._validation_client
        if client is not None:
            try:
                # Prefer asset lookup for existence and exchange
                asset = client.get_asset(s)
                if asset is not None and getattr(asset, "symbol", None):
                    exists = True
                    asset_exch = getattr(asset, "exchange", None)
                else:
                    exists = False
            except Exception:
                # Fallback to quote lookup
                try:
                    quote = client.get_latest_quote(s)
                    exists = bool(quote)
                except Exception:
                    exists = False
        else:
            # No validation backend available → reject
            exists = False

        if not exists:
            return False

        # Update runtime symbols
        self.config.runtime.symbols.append(s)

        # No simulation parameters to initialize

        # Add default exchange info entry
        # Choose exchange: prefer Alpaca asset-derived if available
        exchange = asset_exch or ("NYSE" if s in ["SPY", "IVV", "VOO", "VXX"] else "NASDAQ")
        self._exchange_info[s] = ExchangeInfo(
            symbol=s,
            exchange=exchange,
            is_open=self._is_market_open(),
            trading_hours="9:30 AM - 4:00 PM ET",
        )
        return True





