"""
EuroEdgeSimple Strategy - Optimized Version

Based on 30-day backtest analysis, this version addresses:
- Over-trading issues (16+ trades/day → target 5-8/day)
- Poor exit signal performance (17.4% win rate)
- Missing trend confirmation
- Aggressive exit conditions

Key Improvements:
1. Stricter entry conditions with trend confirmation
2. Simplified exit logic focused on ROI targets
3. Better regime filtering
4. Reduced position adjustment frequency
"""

import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import reduce

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy, 
    CategoricalParameter, 
    DecimalParameter, 
    IntParameter,
    informative
)
from freqtrade.persistence import Trade
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta


logger = logging.getLogger(__name__)


class EuroEdge(IStrategy):
    """
    Optimized version based on backtest analysis
    
    Target Performance:
    - 5-8 trades per day (vs 16+ in original)
    - >50% ROI exits (vs 13% in original)
    - >40% win rate (vs 27% in original)
    """
    
    INTERFACE_VERSION: int = 3
    
    # Basic strategy settings
    timeframe = '5m'
    informative_timeframe = '1h'
    
    # Position management
    can_short = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Position adjustment settings (reduced frequency)
    position_adjustment_enable = True
    max_entry_position_adjustment = 1  # Reduced from 2
    
    # Risk management - more conservative
    startup_candle_count: int = 30
    
    # ROI table - DYNAMIC based on regime (will be overridden in custom_roi)
    minimal_roi = {
        "0": 0.06,    # Moderate default
        "30": 0.04,   # 4% after 30 min
        "60": 0.02,   # 2% after 1 hour
        "120": 0.01,  # 1% after 2 hours
        "240": 0.005, # 0.5% after 4 hours
        "480": 0      # Break even after 8 hours
    }
    
    # Stop loss - DISABLED (using ATR-based dynamic stoploss)
    stoploss = -0.99  # Disabled - using custom_stoploss() with ATR
    
    # Trailing stop - ENABLED for gains - TIGHTENED for better expectancy
    trailing_stop = True
    trailing_stop_positive = 0.008  # Start trailing at 0.8% profit (was 1.5%)
    trailing_stop_positive_offset = 0.012  # Trail by 1.2% (was 2%)
    trailing_only_offset_is_reached = True  # Only trail after reaching positive offset
    
    # Hyperopt spaces - OPTIMIZED PARAMETERS FROM HYPEROPT
    buy_params = {
        "adx_period": 14,
        "atr_period": 16,         # Optimized: was 14
        "bb_period": 18,          # Optimized: was 20
        "ema_fast": 31,           # Optimized: was 34
        "ema_slow": 103,          # Optimized: was 150 - more responsive
        "entry_cooldown": 2,      # Optimized: was 3 - faster entry
        "highvol_threshold": 3.5, # Optimized: was 2.5 - more conservative
        "min_volume_ratio": 1.4,  # Optimized: was 1.1 - stricter volume filter
        "range_adx_threshold": 29.0, # Optimized: was 25 - higher ADX threshold
        "rsi_period": 18,         # Optimized: was 14 - longer RSI period
        "trend_confirmation": False, # Optimized: was True - disabled for faster entries
    }

    sell_params = {
        "exit_rsi_threshold": 80,  # Higher threshold
        "exit_bb_threshold": True,
        "simplified_exits": True
    }    # Optimized parameters based on hyperopt results (-8.48% vs -12.52% baseline)
    ema_fast = IntParameter(20, 50, default=31, space="buy")      # Optimized: was 34
    ema_slow = IntParameter(100, 200, default=103, space="buy")   # Optimized: was 150
    rsi_period = IntParameter(10, 20, default=18, space="buy")    # Optimized: was 14
    atr_period = IntParameter(10, 20, default=16, space="buy")    # Optimized: was 14
    bb_period = IntParameter(15, 25, default=18, space="buy")     # Optimized: was 20
    adx_period = IntParameter(10, 20, default=14, space="buy")

    # New parameters for better filtering - OPTIMIZED VALUES
    trend_confirmation = CategoricalParameter([True, False], default=True, space="buy")  # ENABLED for higher TF confirmation
    min_volume_ratio = DecimalParameter(1.0, 1.5, default=1.4, decimals=1, space="buy")  # Optimized: was 1.1
    entry_cooldown = IntParameter(3, 6, default=4, space="buy")                          # INCREASED to reduce trade frequency

    # Regime thresholds - OPTIMIZED VALUES
    highvol_threshold = DecimalParameter(2.0, 4.0, default=3.5, decimals=1, space="buy")   # Optimized: was 2.5
    range_adx_threshold = DecimalParameter(20, 30, default=29.0, decimals=0, space="buy")  # Optimized: was 25
    
    # Exit parameters
    exit_rsi_threshold = IntParameter(75, 85, default=80, space="sell")
    exit_bb_threshold = CategoricalParameter([True, False], default=True, space="sell")
    simplified_exits = CategoricalParameter([True, False], default=True, space="sell")
    
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        
        # Load environment variables
        self.dry_run = config.get('dry_run', True)
        self.base_risk_pct = float(os.getenv('BASE_RISK_PCT', '0.4'))  # Reduced
        self.max_total_risk_pct = float(os.getenv('MAX_TOTAL_RISK_PCT', '2.5'))  # Reduced
        self.atr_sl_multiplier = float(os.getenv('ATR_SL_MULTIPLIER', '1.8'))  # TIGHTENED from 2.5
        
        # Trade tracking with cooldown
        self.open_trades_risk = 0.0
        self.last_regime = "UNKNOWN"
        self.last_entry_times = {}  # Track last entry per pair
        
        # Portfolio heat tracking
        self.total_portfolio_risk = 0.0
        self.max_portfolio_risk = 5.0  # Max 5% total portfolio risk
        
        # Regime-specific ROI tables - LESS GREEDY in BULL for better expectancy
        self.roi_tables = {
            'BULL': {
                "0": 0.04,    # Less greedy (was 8%)
                "20": 0.03,   # 3% after 20 min
                "60": 0.02,   # 2% after 1 hour
                "120": 0.01,  # 1% after 2 hours
                "240": 0.005, # 0.5% after 4 hours
                "480": 0      # Break even after 8 hours
            },
            'RANGE': {
                "0": 0.03,    # Conservative in range markets
                "30": 0.02,
                "60": 0.015,
                "120": 0.01,
                "240": 0.005,
                "480": 0
            },
            'BEAR': {
                "0": 0.02,    # Very conservative in bear markets
                "30": 0.015,
                "60": 0.01,
                "120": 0.005,
                "240": 0.002,
                "480": 0
            },
            'HIGHVOL': {
                "0": 0.02,    # Quick exits in high volatility
                "15": 0.01,
                "30": 0.005,
                "60": 0,
            }
        }
        
        logger.info("EuroEdge strategy initialized with ATR-based stoploss and regime-aware ROI")
    
    def _market_ok(self, current_time: datetime) -> bool:
        """BTC/ETH regime filter - only trade when major markets are trending up"""
        try:
            for ref in ['BTC/USDT', 'ETH/USDT']:
                try:
                    df, _ = self.dp.get_analyzed_dataframe(ref, '1h')
                    if df is not None and len(df) > 0:
                        last = df.iloc[-1]
                        if not ((last['ema_fast_1h'] > last['ema_slow_1h']) and (last['adx_1h'] > 20)):
                            return False
                except Exception:
                    continue  # Skip if data not available for this pair
            return True
        except Exception:
            return True  # fail-open in backtests without ref data
    
    def informative_pairs(self):
        """Define informative pairs for higher timeframe analysis"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs
    
    @informative('1h')
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate indicators for 1h timeframe (trend confirmation)"""
        # Trend indicators
        dataframe['ema_fast_1h'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow_1h'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['rsi_1h'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx_1h'] = ta.ADX(dataframe, timeperiod=14)
        
        # Trend direction
        dataframe['trend_up_1h'] = dataframe['ema_fast_1h'] > dataframe['ema_slow_1h']
        dataframe['trend_strong_1h'] = dataframe['adx_1h'] > 25
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate indicators for 5m timeframe"""
        
        # Basic indicators
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period.value)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period.value)
        
        # Bollinger Bands
        bb = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period.value, stds=2)
        dataframe['bb_lower'] = bb['lower']
        dataframe['bb_middle'] = bb['mid']
        dataframe['bb_upper'] = bb['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Volume analysis
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # ATR percentage
        dataframe['atr_pct'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # EMA slope (trend strength)
        dataframe['ema_slope'] = (dataframe['ema_fast'] - dataframe['ema_fast'].shift(5)) / dataframe['ema_fast'] * 100
        
        # MOMENTUM INDICATORS FOR GAINS
        dataframe['momentum'] = (dataframe['close'] - dataframe['close'].shift(10)) / dataframe['close'].shift(10) * 100
        dataframe['price_above_ema_fast'] = dataframe['close'] > dataframe['ema_fast']
        dataframe['price_above_ema_slow'] = dataframe['close'] > dataframe['ema_slow']
        
        # BULL MARKET DETECTION
        dataframe['strong_bull'] = (
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['ema_slope'] > 0.1) &  # Strong upward slope
            (dataframe['momentum'] > 2) &     # Strong momentum
            (dataframe['rsi'] < 70)           # Not overbought
        )
        
        # Market regime detection - improved for gains
        dataframe['regime'] = self._detect_market_regime_optimized(dataframe)
        
        return dataframe
    
    def _detect_market_regime_optimized(self, dataframe: DataFrame) -> pd.Series:
        """Improved market regime detection with stricter criteria"""
        regime = pd.Series(index=dataframe.index, dtype=str)
        
        # High volatility detection - stricter
        high_vol_condition = dataframe['atr_pct'] > self.highvol_threshold.value
        
        # Trend detection - improved
        trend_up = (dataframe['ema_fast'] > dataframe['ema_slow']) & (dataframe['ema_slope'] > 0.05)
        trend_down = (dataframe['ema_fast'] < dataframe['ema_slow']) & (dataframe['ema_slope'] < -0.05)
        trend_strong = dataframe['adx'] > 30  # Stricter
        
        # Range detection - more conservative
        range_condition = (
            (dataframe['adx'] < self.range_adx_threshold.value) & 
            (abs(dataframe['ema_slope']) < 0.02) &
            (~high_vol_condition)
        )
        
        # Assign regimes with priority
        regime.loc[high_vol_condition] = 'HIGHVOL'
        regime.loc[trend_up & trend_strong & (~high_vol_condition)] = 'BULL'
        regime.loc[trend_down & trend_strong & (~high_vol_condition)] = 'BEAR'
        regime.loc[range_condition & (~high_vol_condition)] = 'RANGE'
        
        # Forward fill unknown values
        regime = regime.fillna(method='ffill').fillna('RANGE')
        
        return regime
    
    def _should_enter_trade(self, pair: str, current_time: datetime) -> bool:
        """Check if enough time has passed since last entry (anti-overtrading)"""
        if pair not in self.last_entry_times:
            return True
        
        time_diff = current_time - self.last_entry_times[pair]
        cooldown_minutes = self.entry_cooldown.value * 5  # Convert to minutes
        
        return time_diff.total_seconds() >= (cooldown_minutes * 60)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Optimized entry conditions - FOCUS ON EXPECTANCY"""
        
        pair = metadata['pair']
        
        # ENABLE trend confirmation for better win rate
        trend_confirmation_enabled = True
        
        # Base conditions - stricter
        base_conditions = (
            (dataframe['volume'] > 0) &
            (dataframe['rsi'] > 35) &  # Stricter
            (dataframe['rsi'] < 65) &  # Stricter
            (dataframe['volume_ratio'] > self.min_volume_ratio.value) &
            (dataframe['atr_pct'] < 3.0)  # Avoid high volatility
        )
        
        # Bull market entries - REQUIRE 1H TREND CONFIRMATION
        bull_conditions = (
            (dataframe['regime'] == 'BULL') &
            (dataframe['close'] > dataframe['ema_fast']) &
            (dataframe['ema_fast'] > dataframe['ema_slow']) &
            (dataframe['rsi'] > 40) &
            (dataframe['rsi'] < 70) &  # Allow higher RSI for momentum
            (dataframe['bb_percent'] > 0.2) &
            (dataframe['bb_percent'] < 0.9) &  # Allow higher BB for momentum
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['momentum'] > 1) &  # Strong momentum
            (dataframe['trend_up_1h_1h']) &    # ADD: Require 1h uptrend
            (dataframe['trend_strong_1h_1h'])  # ADD: Require 1h strong trend
        )
        
        # STRONG BULL CONDITIONS - Capture explosive moves with 1H confirmation
        strong_bull_conditions = (
            (dataframe['strong_bull']) &
            (dataframe['volume_ratio'] > 1.5) &  # Strong volume
            (dataframe['ema_slope'] > 0.15) &    # Very strong trend
            (dataframe['rsi'] > 50) &            # Momentum
            (dataframe['rsi'] < 75) &            # Not extreme overbought
            (dataframe['trend_up_1h_1h']) &         # ADD: Require 1h uptrend
            (dataframe['trend_strong_1h_1h'])       # ADD: Require 1h strong trend
        )
        
        # DISABLE RANGE TRADING for now - focus on trends only
        # range_conditions = False  # REMOVED - no range trading until PF > 1
        
        # DISABLE BEAR MARKET ENTRIES - this is where we lost money!
        bear_conditions = (
            # DISABLED: No entries in bear markets
            False  # Force disable bear entries
        )
        
        # Combine all conditions - ONLY BULL ENTRIES with 1H confirmation
        entry_signal = base_conditions & (strong_bull_conditions | bull_conditions)
        
        # Portfolio heat check - prevent overexposure
        if hasattr(self, 'total_portfolio_risk') and self.total_portfolio_risk > self.max_portfolio_risk:
            logger.info(f"Portfolio risk too high ({self.total_portfolio_risk:.1f}%), blocking new entries")
            entry_signal[:] = False
        
        # Apply entry cooldown (anti-overtrading) - INCREASED cooldown
        for i in range(len(dataframe)):
            if entry_signal.iloc[i]:
                current_time = dataframe['date'].iloc[i]
                if not self._should_enter_trade(pair, current_time):
                    entry_signal.iloc[i] = False
        
        dataframe.loc[entry_signal, 'enter_long'] = 1
        
        # Never enter in HIGHVOL regime
        dataframe.loc[dataframe['regime'] == 'HIGHVOL', 'enter_long'] = 0
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Enhanced exit conditions with volatility protection"""
        
        if self.simplified_exits.value:
            # Enhanced exit logic with volatility protection
            conditions = [
                # Extreme overbought
                (dataframe['rsi'] > self.exit_rsi_threshold.value),
                
                # Regime change to high volatility (protection)
                (dataframe['regime'] == 'HIGHVOL'),
                
                # Volatility spike protection (new)
                (dataframe['atr_pct'] > 5.0),  # Exit on volatility spikes
                
                # Strong momentum reversal
                ((dataframe['momentum'] < -2) & (dataframe['rsi'] > 60)),
            ]
            
            if self.exit_bb_threshold.value:
                conditions.append(dataframe['close'] > dataframe['bb_upper'] * 1.01)  # Well above upper BB
            
        else:
            # Original exit conditions (for comparison)
            conditions = [
                (dataframe['rsi'] > 75),
                (dataframe['close'] > dataframe['bb_upper']),
                (qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])),
                (dataframe['regime'] == 'HIGHVOL'),
                ((dataframe['close'] < dataframe['ema_fast']) & 
                 (dataframe['volume_ratio'] > 1.5))
            ]
        
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'exit_long'
            ] = 1
        
        return dataframe
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ATR-based dynamic stoploss with BREAKEVEN protection for better expectancy
        """
        try:
            # BREAKEVEN + TIGHTER TRAILING for expectancy improvement
            if current_profit >= 0.006:   # +0.6%
                return -0.002             # -0.2% (near breakeven)
            if current_profit >= 0.012:   # +1.2%
                return 0.0                # hard breakeven
            
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            latest = dataframe.iloc[-1]
            
            # Calculate ATR-based stoploss
            atr_value = latest['atr']
            current_price = latest['close']
            
            # Base ATR multiplier - TIGHTENED
            base_multiplier = self.atr_sl_multiplier
            
            # Adjust multiplier based on regime (more conservative in volatile conditions)
            regime = latest['regime']
            if regime == 'HIGHVOL':
                multiplier = base_multiplier * 1.5  # Wider stops in high volatility
            elif regime == 'BEAR':
                multiplier = base_multiplier * 1.2  # Wider stops in bear markets
            elif regime == 'BULL':
                multiplier = base_multiplier * 0.8  # Tighter stops in bull markets
            else:  # RANGE
                multiplier = base_multiplier
            
            # Calculate stoploss as percentage
            atr_stoploss_pct = (atr_value * multiplier) / current_price
            
            # TIGHTER stoploss limits for better expectancy
            min_stoploss = 0.012  # 1.2% minimum (was 1.5%)
            max_stoploss = 0.055  # 5.5% maximum (was 8%)
            
            atr_stoploss_pct = max(min_stoploss, min(max_stoploss, atr_stoploss_pct))
            
            # Return as negative percentage
            return -atr_stoploss_pct
            
        except Exception as e:
            logger.error(f"Error calculating ATR stoploss for {pair}: {e}")
            return -0.05  # Fallback to 5%
    
    def custom_roi(self, pair: str, trade: 'Trade', current_time: datetime,
                  current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Regime-based dynamic ROI - aggressive in BULL, conservative in BEAR/RANGE
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            latest = dataframe.iloc[-1]
            
            # Get current regime
            regime = latest['regime']
            
            # Use regime-specific ROI table
            roi_table = self.roi_tables.get(regime, self.roi_tables['RANGE'])
            
            # Calculate minutes since trade start
            trade_duration = (current_time - trade.open_date_utc).total_seconds() / 60
            
            # Find appropriate ROI based on duration
            for duration_str, roi_value in roi_table.items():
                duration_minutes = int(duration_str)
                if trade_duration >= duration_minutes:
                    return roi_value
            
            # If no match, return the last (highest duration) ROI
            return list(roi_table.values())[-1]
            
        except Exception as e:
            logger.error(f"Error calculating custom ROI for {pair}: {e}")
            return 0.02  # Fallback to 2%

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                          entry_tag: Optional[str], side: str, **kwargs) -> float:
        """Custom entry price - slightly below market for better fills"""
        return proposed_rate * 0.999  # 0.1% below market
    
    def adjust_trade_position(self, trade: 'Trade', current_time: datetime, 
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            **kwargs) -> Optional[float]:
        """Enhanced position adjustment with stricter criteria"""
        
        # Only adjust if losing significantly AND in favorable regime
        if current_profit > -0.05:  # Only if down more than 5% (was 1.5%)
            return None
        
        # Get current market data
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        latest = dataframe.iloc[-1]
        
        # Only adjust in BULL regime (where we make money)
        regime = latest['regime']
        if regime != 'BULL':  # Only adjust in bull markets
            logger.info(f"No position adjustment for {trade.pair}: regime is {regime}")
            return None
        
        # Check portfolio risk before adjusting
        if hasattr(self, 'total_portfolio_risk') and self.total_portfolio_risk > 3.0:
            logger.info(f"No position adjustment: portfolio risk too high ({self.total_portfolio_risk:.1f}%)")
            return None
        
        # More conservative adjustment
        if trade.nr_of_successful_entries < self.max_entry_position_adjustment:
            try:
                # Smaller adjustment size
                current_stake = trade.stake_amount
                additional_stake = current_stake * 0.3  # Even smaller adjustment (was 0.5)
                
                # Update last entry time
                self.last_entry_times[trade.pair] = current_time
                
                logger.info(f"Position adjustment for {trade.pair}: +{additional_stake:.2f} USDT in {regime} regime")
                return additional_stake
            except Exception as e:
                logger.error(f"Error in position adjustment: {e}")
        
        return None
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                          time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                          side: str, **kwargs) -> bool:
        """Enhanced entry confirmation with portfolio risk management"""
        
        # Update last entry time
        self.last_entry_times[pair] = current_time
        
        # Additional safety checks
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        latest = dataframe.iloc[-1]
        
        # Don't enter if extreme conditions
        if latest['atr_pct'] > 4.0 or latest['rsi'] > 70 or latest['rsi'] < 30:
            logger.info(f"Entry rejected for {pair}: extreme conditions (ATR: {latest['atr_pct']:.1f}%, RSI: {latest['rsi']:.1f})")
            return False
        
        # Don't enter in BEAR regime (our analysis shows this destroys performance)
        if latest['regime'] == 'BEAR':
            logger.info(f"Entry rejected for {pair}: BEAR regime detected")
            return False
        
        # Portfolio heat check
        if hasattr(self, 'total_portfolio_risk') and self.total_portfolio_risk > self.max_portfolio_risk:
            logger.info(f"Entry rejected for {pair}: portfolio risk limit exceeded ({self.total_portfolio_risk:.1f}%)")
            return False
        
        logger.info(f"Entry confirmed for {pair} in {latest['regime']} regime (ATR: {latest['atr_pct']:.1f}%, RSI: {latest['rsi']:.1f})")
        return True
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str,
                **kwargs) -> float:
        """Dynamic leverage based on regime and volatility"""
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            latest = dataframe.iloc[-1]
            
            regime = latest['regime']
            atr_pct = latest['atr_pct']
            
            # Conservative leverage based on conditions
            if regime == 'BULL' and atr_pct < 2.0:
                return min(1.0, max_leverage)  # Max 1x leverage even in best conditions
            elif regime == 'RANGE' and atr_pct < 1.5:
                return min(0.5, max_leverage)  # 0.5x in range markets
            else:
                return min(0.3, max_leverage)  # Very conservative in other conditions
                
        except Exception as e:
            logger.error(f"Error calculating leverage for {pair}: {e}")
            return min(0.3, max_leverage)  # Safe fallback
