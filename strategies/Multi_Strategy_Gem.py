"""
Enhanced AdvancedMultiStrategy
------------------------------
This file builds on the previously updated multi-strategy by adding:

1. CircuitBreaker for managing repeated failures
2. MetricsCollector for Prometheus-based metrics
3. Thread-safety in PositionManager with a Lock
4. Additional methods in AdvancedMultiStrategy:
    - setup_strategy_logging()
    - analyze_market_structure (now cached)
    - invalidate_caches()
    - cleanup_resources()
    - adjust_for_backtest()
    - _disable_realtime_checks()
    - handle_rate_limiting()
    - emergency_exit()
    - check_strategy_health() + sub-checks
5. Updated bot_loop_start() with health checks and metrics
6. Updated custom_stake_amount() to record gauge metrics
"""

import logging
from threading import Lock
from functools import lru_cache
from datetime import timedelta, datetime
from typing import Dict, List, Optional, Tuple
from time import time, sleep

import numpy as np
import pandas as pd
import talib.abstract as ta
from technical import qtpylib
from prometheus_client import Counter, Gauge, start_http_server

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.configuration import RunMode
from freqtrade.enums import CandleType

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Tracks consecutive failures. If threshold is exceeded, triggers circuit break.
    Resets failures after cooldown_period passes without new failures.
    """

    def __init__(self, threshold: int = 3, cooldown_period: int = 300):
        self.failures = 0
        self.threshold = threshold
        self.cooldown_period = cooldown_period  # seconds
        self._last_failure = 0

    def record_failure(self):
        self.failures += 1
        self._last_failure = time()

    def should_break(self) -> bool:
        # If enough time has passed since last failure, reset
        if time() - self._last_failure > self.cooldown_period:
            self.failures = 0
        return self.failures >= self.threshold


class MetricsCollector:
    """
    Simple Prometheus metrics collector for trades, positions, and profit.
    Extend with your own counters/gauges if desired.
    """

    def __init__(self):
        self.trade_counter = Counter("trades_total", "Total trades")
        self.position_gauge = Gauge("position_size", "Position size", ["pair"])
        self.profit_gauge = Gauge("current_profit", "Current profit")


class PositionManager:
    """
    Manages positions with thread safety and circuit breaker usage.
    """

    def __init__(self):
        self.positions: Dict[str, float] = {}
        self.risk_limits: Dict[str, float] = {}
        self._lock = Lock()
        self.circuit_breaker = CircuitBreaker()

    def validate_position(self, pair: str, size: float) -> bool:
        """
        Example method: check if a new position violates risk limits.
        """
        if pair not in self.risk_limits:
            logger.debug(f"No specific risk limit found for {pair}.")
            return True
        if (self.positions.get(pair, 0) + size) > self.risk_limits[pair]:
            logger.warning(f"Position size for {pair} exceeds configured limit.")
            return False
        return True

    def update_position(self, pair: str, size: float) -> None:
        """
        Thread-safe update of position sizes, with circuit breaker on error.
        """
        with self._lock:
            try:
                self.positions[pair] = self.positions.get(pair, 0) + size
            except Exception as e:
                self.circuit_breaker.record_failure()
                raise


class AdvancedMultiStrategy(IStrategy):
    """
    Advanced Multi-Strategy with Additional Features:
    - Circuit Breaker
    - Metrics Collection
    - Thread-safe PositionManager
    - Enhanced logging, caching, cleanup, etc.
    """

    INTERFACE_VERSION: int = 3
    VERSION: str = "1.3.0"
    AUTHOR: str = "YourName"
    STRATEGY_DESCRIPTION: str = (
        "Advanced multi-strategy with circuit breaker, metrics, "
        "thread-safe PositionManager, and enhanced functionality."
    )

    minimal_required_freqtrade_version: str = "2023.9"
    tags = ["Advanced", "Performance", "ExitLogic"]
    categories = ["Trend Following", "Mean Reversion"]

    # Timeframes and settings
    timeframe = "5m"
    informative_timeframe = "1h"
    startup_candle_count: int = 400
    process_only_new_candles = True
    can_short = True
    trailing_stop = True

    # ROI / Stoploss / Trailing
    minimal_roi = {"0": 0.05, "30": 0.025, "60": 0.015, "120": 0.01}
    stoploss = -0.025
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "emergency_exit": "market",
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    max_open_trades = 5
    position_adjustment_enable = True
    max_entry_position_adjustment = 3
    max_dca_multiplier = 2.0

    # Hyperopt parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")
    short_rsi = IntParameter(60, 90, default=70, space="sell")

    atr_period = IntParameter(10, 30, default=14, space="buy")
    bb_period = IntParameter(15, 50, default=20, space="buy")
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, space="buy")

    ema_short = IntParameter(3, 15, default=5, space="buy")
    ema_long = IntParameter(15, 50, default=21, space="buy")

    high_volatility_threshold = 0.05  # Example for dynamic timeframe

    # Instantiate PositionManager
    position_manager = PositionManager()

    def __init__(self, config: dict) -> None:
        """
        Constructor to initialize metrics, set up logging, etc.
        """
        super().__init__(config)
        self._cache_timestamp = time()
        self.metrics = MetricsCollector()
        self.setup_strategy_logging()

        # Start Prometheus server, with port defined in config or defaulting to 8080
        prometheus_config = config.get("prometheus", {})
        prometheus_port = prometheus_config.get("prometheus_port", 8080)
        start_http_server(prometheus_port)

    def setup_strategy_logging(self):
        """Configure strategy-specific logging to a file."""
        handler = logging.FileHandler("strategy_debug.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def bot_loop_start(self, **kwargs) -> None:
        """Enhanced bot loop start with health checks and metrics updates."""
        super().bot_loop_start(**kwargs)

        # Perform health checks
        health_status = self.check_strategy_health()
        if not all(health_status.values()):
            logger.warning(f"Health check failed: {health_status}")

        # Update position metrics
        for pair, size in self.position_manager.positions.items():
            self.metrics.position_gauge.labels(pair=pair).set(size)

        # Perform any existing logic
        self.adjust_timeframes()
        self.adapt_to_market_regime()
        self.update_market_conditions()

        # Handle potential rate limiting
        if hasattr(self, "exchange"):
            self.handle_rate_limiting(self.exchange.name)

    @lru_cache(maxsize=100)
    def analyze_market_structure(self) -> Dict:
        """
        Cached analysis of market structure.
        Replace with actual logic or keep the placeholders from earlier code.
        """
        return {
            "liquidity_score": self.calculate_liquidity_score("GLOBAL"),
            "market_regime": self.detect_market_regime(),
            "correlation_matrix": None,
            "volatility_surface": None,
        }

    def invalidate_caches(self):
        """Invalidate all strategy caches."""
        self._cache_timestamp = time()
        self.analyze_market_structure.cache_clear()

    def cleanup_resources(self):
        """Clean up resources when strategy stops."""
        self.invalidate_caches()
        self.position_manager = None
        logger.info("Strategy resources cleaned up")

    def adjust_for_backtest(self) -> None:
        """Modify certain behaviors for backtesting mode."""
        if self.config.get("runmode") == RunMode.BACKTEST:
            logger.info("Adjusting strategy for backtesting")
            self._disable_realtime_checks()

    def _disable_realtime_checks(self):
        """Disable real-time checks that are irrelevant for backtests."""
        self.analyze_market_impact = lambda *args, **kwargs: {"slippage_estimate": 0.0}

    def handle_rate_limiting(self, exchange_name: str) -> None:
        """Handle exchange rate limits if necessary."""
        if hasattr(self, "exchange") and self.exchange.has.get("rateLimit"):
            sleep(self.exchange.rateLimit / 1000)

    def emergency_exit(self, pair: str, reason: str) -> None:
        """Close all positions for a pair due to an emergency."""
        logger.warning(f"Emergency exit triggered for {pair}: {reason}")
        try:
            trades = Trade.get_trades_proxy(pair=pair, is_open=True)
            for trade in trades:
                self.force_exit(trade.pair, reason)

            # Trigger circuit breaker to prevent further trades
            self.position_manager.circuit_breaker.record_failure()

        except Exception as e:
            logger.error(f"Emergency exit failed: {e}")

    def check_strategy_health(self) -> Dict[str, bool]:
        """Check critical aspects of strategy health."""
        return {
            "exchange_connection": self._check_exchange_connection(),
            "data_feed": self._check_data_feed_health(),
            "position_manager": not self.position_manager.circuit_breaker.should_break(),
            "risk_limits": self._check_risk_limits(),
        }

    def _check_exchange_connection(self) -> bool:
        """Check if the exchange connection is functional."""
        try:
            self.exchange.fetch_balance()
            return True
        except Exception:
            return False

    def _check_data_feed_health(self) -> bool:
        """Confirm that data feed is returning valid pairs."""
        try:
            pairs = self.dp.current_whitelist()
            return len(pairs) > 0
        except Exception:
            return False

    def _check_risk_limits(self) -> bool:
        """Ensure current positions don't exceed risk limits."""
        return all(
            pos <= self.position_manager.risk_limits.get(pair, float("inf"))
            for pair, pos in self.position_manager.positions.items()
        )

    def informative_pairs(self) -> List[Tuple[str, str]]:
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:

        try:
            # Volume Ratio
            dataframe["_temp_vol_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)
            dataframe["volume_ratio"] = np.where(
                dataframe["_temp_vol_sma"] > 0,
                dataframe["volume"] / dataframe["_temp_vol_sma"],
                0,
            )

            # EMA
            dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=self.ema_short.value)
            dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.ema_long.value)

            # Bollinger Bands
            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(dataframe),
                window=self.bb_period.value,
                stds=self.bb_std.value,
            )
            dataframe["bb_lowerband"] = bollinger["lower"]
            dataframe["bb_middleband"] = bollinger["mid"]
            dataframe["bb_upperband"] = bollinger["upper"]

            # RSI
            dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

            # ATR
            dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period.value)

            # Custom Indicators
            dataframe["trend_strength"] = self.calculate_trend_strength(dataframe)
            dataframe["volatility_state"] = self.calculate_volatility_state(dataframe)
            dataframe = self.custom_indicators(dataframe)

        except Exception as e:
            logger.error(
                f"Error calculating indicators for {metadata.get('pair', 'unknown')}: {e}"
            )
            raise
        finally:
            # Clean up temporary columns, ensure this runs even if there's an error
            dataframe.drop(
                columns=["_temp_vol_sma"],
                inplace=True,
                errors="ignore",
            )

        return dataframe

    def calculate_trend_strength(self, dataframe: pd.DataFrame) -> pd.Series:
        short_ema = dataframe["ema_short"]
        long_ema = dataframe["ema_long"]
        ratio = np.where(long_ema != 0, (short_ema - long_ema) / long_ema, 0)
        rsi_normalized = np.where(dataframe["rsi"] != 0, (dataframe["rsi"] - 50) / 50, 0)
        return pd.Series((ratio + rsi_normalized) / 2, index=dataframe.index)

    def calculate_volatility_state(self, dataframe: pd.DataFrame) -> pd.Series:
        vol = dataframe["atr"] / dataframe["close"]
        vol_ma = vol.rolling(window=100).mean()
        vol_std = vol.rolling(window=100).std()

        vol_state = pd.Series("normal", index=dataframe.index)
        high_mask = vol > (vol_ma + vol_std)
        low_mask = vol < (vol_ma - vol_std)

        vol_state.loc[high_mask] = "high"
        vol_state.loc[low_mask] = "low"
        return vol_state

    def custom_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["va_rsi"] = np.where(
            dataframe["close"] != 0,
            dataframe["rsi"] * (1 + dataframe["atr"] / dataframe["close"]),
            dataframe["rsi"],
        )
        dataframe["vol_price_conf"] = (
            dataframe["volume_ratio"]
            * ((dataframe["close"] - dataframe["ema_long"]) / dataframe["close"]).fillna(
                0