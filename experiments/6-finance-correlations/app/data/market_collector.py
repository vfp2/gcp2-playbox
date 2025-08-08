from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
import math
import hashlib

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
    def __init__(self, config: AppConfig, buffer: CircularBuffer[MarketTick]) -> None:
        self.config = config
        self.buffer = buffer
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = PriceState(last_price={})
        self._exchange_info: Dict[str, ExchangeInfo] = {}
        
        # Deterministic per-symbol simulation parameters (so lines are distinct)
        self._sim_params: Dict[str, Dict[str, float]] = {}
        for s in config.runtime.symbols:
            h = int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
            # map hash to stable params
            base_offset = (h % 4000) / 100.0 - 20.0  # [-20, +20]
            amp1 = 1.0 + ((h >> 3) % 300) / 100.0    # [1.0, 4.0]
            amp2 = 0.2 + ((h >> 9) % 80) / 100.0     # [0.2, 1.0]
            phase1 = ((h >> 15) % 628) / 100.0       # [0, 6.28]
            phase2 = ((h >> 21) % 628) / 100.0       # [0, 6.28]
            self._sim_params[s] = {
                "base_offset": base_offset,
                "amp1": amp1,
                "amp2": amp2,
                "phase1": phase1,
                "phase2": phase2,
            }

        self._client = None
        if not config.runtime.dev_mode and REST is not None:
            self._client = REST(
                key_id=config.env.ALPACA_API_KEY,
                secret_key=config.env.ALPACA_SECRET_KEY,
                base_url=config.env.ALPACA_BASE_URL,
            )
        
        # Initialize exchange info
        self._init_exchange_info()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="MarketCollector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
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
            self._stop.wait(timeout=2)

    def _update_market_status(self) -> None:
        """Update market status for all symbols."""
        is_open = self._is_market_open()
        for symbol in self.config.runtime.symbols:
            if symbol in self._exchange_info:
                self._exchange_info[symbol].is_open = is_open

    def _get_latest_price(self, symbol: str) -> Optional[float]:
        if self._client is None:
            # Simulation: symbol-specific deterministic waveform so lines are distinct
            t = datetime.now(timezone.utc).timestamp()
            p = self._sim_params.get(symbol) or {"base_offset": 0.0, "amp1": 2.0, "amp2": 0.5, "phase1": 0.0, "phase2": 0.0}
            base = 400.0 + p["base_offset"]
            slow = p["amp1"] * math.sin(t / 90.0 + p["phase1"])  # slow component
            fast = p["amp2"] * math.sin(t / 6.0 + p["phase2"])   # fast wiggle
            micro = 0.3 * math.sin(t / 2.0 + p["phase1"]/2.0)
            return base + slow + fast + micro
        try:
            # Using quotes endpoint for latest bid/ask midpoint
            quote = self._client.get_latest_quote(symbol)
            bid = float(quote.bidprice)
            ask = float(quote.askprice)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
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
        if self._client is None:
            # Simulation mode - assume market is open during business hours
            now = datetime.now(timezone.utc)
            
            # Convert to ET (Eastern Time)
            # ET is UTC-5 (EST) or UTC-4 (EDT) - using EDT for simplicity
            # In production, this should use pytz for proper timezone handling
            et_offset = -4  # EDT offset (UTC-4)
            et_hour = now.hour + et_offset
            if et_hour < 0:
                et_hour += 24
            elif et_hour >= 24:
                et_hour -= 24
            et_minute = now.minute
            
            # Market hours: 9:30 AM - 4:00 PM ET
            market_start = 9 * 60 + 30  # 9:30 AM in minutes
            market_end = 16 * 60  # 4:00 PM in minutes
            current_time = et_hour * 60 + et_minute
            
            # Check if it's a weekday (Monday = 0, Sunday = 6)
            weekday = now.weekday()
            is_weekday = weekday < 5  # Monday to Friday
            
            return is_weekday and market_start <= current_time < market_end
        
        try:
            # Try to get market clock from Alpaca API
            # This is a placeholder - actual implementation depends on Alpaca API
            # clock = self._client.get_clock()
            # return clock.is_open if hasattr(clock, 'is_open') else True
            return True
        except Exception:
            return True

    def get_exchange_info(self, symbol: str) -> Optional[ExchangeInfo]:
        """Get exchange information for a symbol."""
        return self._exchange_info.get(symbol)

    def get_all_exchange_info(self) -> Dict[str, ExchangeInfo]:
        """Get exchange information for all symbols."""
        return self._exchange_info.copy()




