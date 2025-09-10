# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List, Optional
from functools import reduce
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from datetime import datetime
from freqtrade.persistence import Trade


class CryptoFuturesStrategy(IStrategy):
    """
    A robust futures trading strategy for BTC, ETH, and DOGE with dual-directional trading,
    combining trend-following and mean-reversion signals with advanced risk management.
    
    Author: ergs0204
    Version: 1.0
    """

    INTERFACE_VERSION = 3
    timeframe = '15m'
    minimal_roi = {"0": 0.05}  # Dynamic exits via custom stoploss

    can_short = True

    # Hard stoploss for emergency exit
    stoploss = -0.99
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False


    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Configure all required technical indicators"""
        
        # Trend Indicators
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # Momentum Indicators
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Volatility Indicators
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_sma50'] = ta.SMA(dataframe['atr'], timeperiod=50)

        # Volume Analysis
        dataframe['volume_sma20'] = ta.SMA(dataframe['volume'], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Define entry conditions for both long and short trades"""
        
        # Long Entry Criteria
        dataframe.loc[
            (
                (dataframe['ema50'] > dataframe['ema200']) &  # Primary trend
                (dataframe['adx'] > 25) &  # Strong trend confirmation
                qtpylib.crossed_above(dataframe['macd'], dataframe['macd_signal']) &
                (dataframe['rsi'] < 65) &  # Avoid overbought
                (dataframe['close'] > dataframe['bb_lower']) &  # BB bounce
                (dataframe['volume'] > dataframe['volume_sma20']) &  # Volume spike
                (dataframe['atr'] > dataframe['atr_sma50'])  # Volatility filter
            ),
            'enter_long'] = 1

        # Short Entry Criteria
        dataframe.loc[
            (
                (dataframe['ema50'] < dataframe['ema200']) &  # Primary trend
                (dataframe['adx'] > 25) &  # Strong trend confirmation
                qtpylib.crossed_below(dataframe['macd'], dataframe['macd_signal']) &
                (dataframe['rsi'] > 35) &  # Avoid oversold
                (dataframe['close'] < dataframe['bb_upper']) &  # BB rejection
                (dataframe['volume'] > dataframe['volume_sma20']) &
                (dataframe['atr'] > dataframe['atr_sma50'])
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Optional exit signals for early profit taking"""
        
        dataframe.loc[
            qtpylib.crossed_below(dataframe['macd'], dataframe['macd_signal']),
            'exit_long'] = 1

        dataframe.loc[
            qtpylib.crossed_above(dataframe['macd'], dataframe['macd_signal']),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Dynamic ATR-based trailing stoploss"""
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            return 0.10  # Fallback value

        current_atr = dataframe['atr'].iloc[-1]
        if current_atr <= 0:
            current_atr = 0.001

        # Progressive trailing stop
        if trade.is_short:
            stoploss_price = trade.min_rate + (3 * current_atr)
            stoploss = (stoploss_price - current_rate) / current_rate
        else:
            stoploss_price = trade.max_rate - (3 * current_atr)
            stoploss = (current_rate - stoploss_price) / current_rate

        # Ensure minimum 0.5% stop and valid value
        return max(abs(stoploss), 0.005)

    @property
    def protections(self):
        """Protection against extreme market conditions"""
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 5
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.2
            }
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """Dynamic leverage management"""
        return min(proposed_leverage, 3.0)