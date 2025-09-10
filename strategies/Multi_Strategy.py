"""
================================================================================
File: advanced_multi_strategy.py
================================================================================

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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce, lru_cache
from threading import Lock
from typing import Dict, List, Optional, Tuple
import cProfile
import pstats
from prometheus_client import Counter, Gauge
import semantic_version
from time import time, sleep

# TA-Lib / Pandas-TA
import talib.abstract as ta
from technical import qtpylib

# Freqtrade
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from freqtrade.strategy.interface import Timeframe
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.configuration import RunMode  # for backtest check

logger = logging.getLogger(__name__)


# =============================================================================
# Additional Classes
# =============================================================================

class CircuitBreaker:
    """
    Tracks consecutive failures. If threshold is exceeded, triggers circuit break.
    Resets failures after cooldown_period passes without new failures.
    """
    def __init__(self):
        self.failures = 0
        self.threshold = 3
        self.cooldown_period = 300  # seconds
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
        self.trade_counter = Counter('trades_total', 'Total trades')
        self.position_gauge = Gauge('position_size', 'Position size', ['pair'])
        self.profit_gauge = Gauge('current_profit', 'Current profit')


# =============================================================================
# PositionManager Class
# =============================================================================

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


# =============================================================================
# AdvancedMultiStrategy Class
# =============================================================================

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

    # ROI / Stoploss / Trailing
    minimal_roi = {
        "0": 0.05,
        "30": 0.025,
        "60": 0.015,
        "120": 0.01
    }
    stoploss = -0.025
    trailing_stop = True
    trailing_stop_positive = 0.005
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True

    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'emergency_exit': 'market'
    }
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

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

    def __init__(self):
        """
        Constructor to initialize metrics, set up logging, etc.
        """
        super().__init__()
        self._cache_timestamp = time()
        self.metrics = MetricsCollector()
        self.setup_strategy_logging()

    def setup_strategy_logging(self):
        """Configure strategy-specific logging to a file."""
        handler = logging.FileHandler('strategy_debug.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # =========================================================================
    # Example: At the start of each iteration
    # =========================================================================
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
        if hasattr(self, 'exchange'):
            self.handle_rate_limiting(self.exchange.name)

    # =========================================================================
    # Cached Market Structure
    # =========================================================================
    @lru_cache(maxsize=100)
    def analyze_market_structure(self) -> Dict:
        """
        Cached analysis of market structure.
        Replace with actual logic or keep the placeholders from earlier code.
        """
        return {
            'liquidity_score': self.calculate_liquidity_score("GLOBAL"),
            'market_regime': self.detect_market_regime(),
            'correlation_matrix': None,
            'volatility_surface': None,
        }

    def invalidate_caches(self):
        """Invalidate all strategy caches."""
        self._cache_timestamp = time()
        self.analyze_market_structure.cache_clear()

    # =========================================================================
    # Resource Cleanup
    # =========================================================================
    def cleanup_resources(self):
        """Clean up resources when strategy stops."""
        self.invalidate_caches()
        self.position_manager = None
        logger.info("Strategy resources cleaned up")

    # =========================================================================
    # Adjust for Backtesting
    # =========================================================================
    def adjust_for_backtest(self) -> None:
        """Modify certain behaviors for backtesting mode."""
        # The config object is usually accessible as self.config
        if self.config.get('runmode') == RunMode.BACKTEST:
            logger.info("Adjusting strategy for backtesting")
            self._disable_realtime_checks()

    def _disable_realtime_checks(self):
        """Disable real-time checks that are irrelevant for backtests."""
        self.analyze_market_impact = lambda *args, **kwargs: {'slippage_estimate': 0.0}

    # =========================================================================
    # Rate Limiting Handling
    # =========================================================================
    def handle_rate_limiting(self, exchange_name: str) -> None:
        """Handle exchange rate limits if necessary."""
        if hasattr(self, 'exchange') and self.exchange.has.get('rateLimit'):
            sleep(self.exchange.rateLimit / 1000)

    # =========================================================================
    # Emergency Exit Handling
    # =========================================================================
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

    # =========================================================================
    # Strategy Health Checks
    # =========================================================================
    def check_strategy_health(self) -> Dict[str, bool]:
        """Check critical aspects of strategy health."""
        return {
            'exchange_connection': self._check_exchange_connection(),
            'data_feed': self._check_data_feed_health(),
            'position_manager': not self.position_manager.circuit_breaker.should_break(),
            'risk_limits': self._check_risk_limits(),
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
            pos <= self.position_manager.risk_limits.get(pair, float('inf'))
            for pair, pos in self.position_manager.positions.items()
        )

    # =========================================================================
    # Populate Indicators / Entry / Exit
    # (Retaining prior logic with references to advanced features)
    # =========================================================================

    def informative_pairs(self) -> List[Tuple[str, str]]:
        if not self.validate_timeframes():
            logger.warning("Incompatible timeframes detected.")
        pairs = self.dp.current_whitelist()
        return [(pair, self.informative_timeframe) for pair in pairs]

    def validate_timeframes(self) -> bool:
        if not hasattr(self.dp, "validate_timeframes"):
            return True
        return self.dp.validate_timeframes(
            timeframe=self.timeframe,
            informative_timeframes=[self.informative_timeframe],
        )

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        columns_to_drop = [col for col in dataframe.columns if col.startswith("_temp_")]
        dataframe.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        try:
            dataframe['_temp_vol_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
            dataframe['volume_ratio'] = np.where(
                dataframe['_temp_vol_sma'] > 0,
                dataframe['volume'] / dataframe['_temp_vol_sma'],
                0
            )

            dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=self.ema_short.value)
            dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=self.ema_long.value)

            bollinger = qtpylib.bollinger_bands(
                qtpylib.typical_price(dataframe),
                window=self.bb_period.value,
                stds=self.bb_std.value
            )
            dataframe['bb_lowerband'] = bollinger['lower']
            dataframe['bb_middleband'] = bollinger['mid']
            dataframe['bb_upperband'] = bollinger['upper']

            dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
            dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
            dataframe['trend_strength'] = self.calculate_trend_strength(dataframe)
            dataframe['volatility_state'] = self.calculate_volatility_state(dataframe)

            dataframe = self.custom_indicators(dataframe)
        except Exception as e:
            logger.error(f"Error calculating indicators for {metadata.get('pair', 'unknown')}: {e}")
            raise

        dataframe.drop(columns=['_temp_vol_sma'], inplace=True, errors='ignore')
        return dataframe

    def calculate_trend_strength(self, dataframe: pd.DataFrame) -> pd.Series:
        short_ema = dataframe['ema_short']
        long_ema = dataframe['ema_long']
        ratio = np.where(long_ema != 0, (short_ema - long_ema) / long_ema, 0)
        rsi_normalized = np.where(dataframe['rsi'] != 0, (dataframe['rsi'] - 50) / 50, 0)
        return pd.Series((ratio + rsi_normalized) / 2, index=dataframe.index)

    def calculate_volatility_state(self, dataframe: pd.DataFrame) -> pd.Series:
        vol = dataframe['atr'] / dataframe['close']
        vol_ma = vol.rolling(window=100).mean()
        vol_std = vol.rolling(window=100).std()

        vol_state = pd.Series('normal', index=dataframe.index)
        high_mask = vol > (vol_ma + vol_std)
        low_mask = vol < (vol_ma - vol_std)

        vol_state.loc[high_mask] = 'high'
        vol_state.loc[low_mask] = 'low'
        return vol_state

    def custom_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['va_rsi'] = np.where(
            dataframe['close'] != 0,
            dataframe['rsi'] * (1 + dataframe['atr'] / dataframe['close']),
            dataframe['rsi']
        )
        dataframe['vol_price_conf'] = (
            dataframe['volume_ratio'] *
            ((dataframe['close'] - dataframe['ema_long']) / dataframe['close']).fillna(0)
        )
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        long_cond = (
            (dataframe['ema_short'] > dataframe['ema_long']) &
            (dataframe['trend_strength'] > 0.5) &
            (dataframe['rsi'] < self.buy_rsi.value) &
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['close'] < dataframe['bb_lowerband']) &
            (dataframe['volatility_state'] == 'low')
        )
        dataframe.loc[long_cond, 'enter_long'] = 1

        if self.can_short:
            short_cond = (
                (dataframe['ema_short'] < dataframe['ema_long']) &
                (dataframe['trend_strength'] < -0.5) &
                (dataframe['rsi'] > self.short_rsi.value) &
                (dataframe['close'] > dataframe['bb_upperband'])
            )
            dataframe.loc[short_cond, 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        long_exit = (
            (dataframe['rsi'] > self.sell_rsi.value) |
            (dataframe['ema_short'] < dataframe['ema_long']) |
            (dataframe['close'] > dataframe['bb_upperband'])
        )
        dataframe.loc[long_exit, 'exit_long'] = 1

        if self.can_short:
            short_exit = (
                (dataframe['rsi'] < 40) |
                (dataframe['close'] < dataframe['bb_lowerband'])
            )
            dataframe.loc[short_exit, 'exit_short'] = 1

        return dataframe

    # =========================================================================
    # Trade Exit Management and Additional Features (Retained)
    # =========================================================================
    def should_exit_trade(self, trade: Trade, current_rate: float) -> bool:
        return (
            self.check_technical_exit(trade, current_rate) or
            self.check_risk_based_exit(trade) or
            self.check_correlation_based_exit(trade) or
            self.check_portfolio_based_exit(trade)
        )

    def check_technical_exit(self, trade: Trade, current_rate: float) -> bool:
        return False

    def check_risk_based_exit(self, trade: Trade) -> bool:
        return False

    def check_correlation_based_exit(self, trade: Trade) -> bool:
        return False

    def check_portfolio_based_exit(self, trade: Trade) -> bool:
        return False

    # =========================================================================
    # Market Impact Analysis
    # =========================================================================
    def analyze_market_impact(self, pair: str, order_size: float) -> Dict:
        return {
            'slippage_estimate': self.estimate_slippage(pair, order_size),
            'liquidity_score': self.calculate_liquidity_score(pair),
            'spread_analysis': self.analyze_spread(pair),
            'order_book_depth': self.analyze_order_book_depth(pair),
        }

    def estimate_slippage(self, pair: str, order_size: float) -> float:
        return 0.0

    def calculate_liquidity_score(self, pair: str) -> float:
        return 0.0

    def analyze_spread(self, pair: str) -> float:
        return 0.0

    def analyze_order_book_depth(self, pair: str) -> float:
        return 0.0

    # =========================================================================
    # Dynamic Timeframe Adjustment
    # =========================================================================
    def adjust_timeframes(self) -> None:
        current_volatility = self.get_market_volatility()
        if current_volatility > self.high_volatility_threshold:
            self.timeframe = self.get_higher_timeframe()
            logger.info(f"Timeframe updated to {self.timeframe} due to high volatility.")

    def get_market_volatility(self) -> float:
        return 0.01

    def get_higher_timeframe(self) -> str:
        return "15m"

    # =========================================================================
    # Risk Management
    # =========================================================================
    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs
    ) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        atr_sl_multiplier = 2.0
        atr_sl = last_candle['atr'] * atr_sl_multiplier

        if current_profit > 0.02:
            return current_profit * 0.5

        return -atr_sl / current_rate

    def custom_stake_amount(
        self, pair: str, current_time: datetime, current_rate: float,
        proposed_stake: float, min_stake: Optional[float], max_stake: float,
        leverage: float, entry_tag: Optional[str], side: str, **kwargs
    ) -> float:
        """
        Enhanced stake amount calculation with metrics.
        """
        try:
            # We can call super() if we want the previously-coded logic from your last iteration:
            adjusted_stake = super().custom_stake_amount(
                pair, current_time, current_rate, proposed_stake,
                min_stake, max_stake, leverage, entry_tag, side, **kwargs
            )
            # Update metrics with the final stake amount
            self.metrics.position_gauge.labels(pair=pair).set(adjusted_stake)

            return adjusted_stake
        except Exception as e:
            logger.error(f"Stake calculation error: {e}")
            return min_stake or 0

    def confirm_trade_entry(
        self, pair: str, order_type: str, amount: float,
        rate: float, time_in_force: str, current_time: datetime,
        entry_tag: Optional[str], side: str, **kwargs
    ) -> bool:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if last_candle['trend_strength'] < 0.3 or last_candle['volume_ratio'] < 1.2:
                return False
        else:
            if last_candle['trend_strength'] > -0.3:
                return False

        return True

    # =========================================================================
    # Market Analysis and Regime
    # =========================================================================
    def detect_market_regime(self) -> str:
        return "neutral"

    def update_market_conditions(self) -> None:
        pairs = self.dp.current_whitelist()
        for pair in pairs:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.informative_timeframe)

    def adapt_to_market_regime(self) -> None:
        regime = self.detect_market_regime()
        self.update_parameters(regime)
        self.adjust_risk_levels(regime)
        self.modify_indicators(regime)

    def update_parameters(self, regime: str) -> None:
        pass

    def adjust_risk_levels(self, regime: str) -> None:
        pass

    def modify_indicators(self, regime: str) -> None:
        pass

    # =========================================================================
    # Risk Scoring
    # =========================================================================
    def calculate_risk_score(self, pair: str) -> float:
        weights = {
            'volatility': 0.3,
            'liquidity': 0.2,
            'correlation': 0.2,
            'trend': 0.15,
            'sentiment': 0.15
        }
        score = 0.0
        for factor, weight in weights.items():
            score += weight * self.get_risk_factor(factor, pair)
        return score

    def get_risk_factor(self, factor: str, pair: str) -> float:
        return 1.0

    # =========================================================================
    # Helper Methods
    # =========================================================================
    def log_trade_info(self, pair: str, current_rate: float,
                       current_profit: float, trade: Trade) -> None:
        logger.info(f"""
            Pair: {pair}
            Current Rate: {current_rate}
            Profit: {current_profit}
            Entry Tag: {trade.enter_tag}
            Duration (minutes): {trade.duration_in_minutes}
        """)

    # =========================================================================
    # Testing and Exchange-Specific
    # =========================================================================
    def test_strategy_components(self) -> bool:
        return True

    def validate_exchange_compatibility(self) -> bool:
        required_features = [
            'fetchOHLCV',
            'createOrder',
            'fetchOrder',
            'fetchBalance',
            'cancelOrder'
        ]
        if not hasattr(self, 'exchange') or not self.exchange:
            return False
        return all(self.exchange.has.get(feature, False) for feature in required_features)

    def monitor_strategy_performance(self) -> Dict:
        return {
            'win_rate': None,
            'profit_factor': None,
            'sharpe_ratio': None,
            'max_drawdown': None
        }