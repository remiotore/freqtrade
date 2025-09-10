# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class AdvancedFuturesStrategy(IStrategy):
    """
    Advanced Futures Trading Strategy
    
    Strategi ini dirancang untuk futures trading dengan fokus pada:
    - High Average/Cumulative/Total Profit %
    - High Win Rate %
    - Low Drawdown %
    - High Sharpe Ratio
    
    Menggunakan kombinasi indikator teknikal dan manajemen risiko yang ketat
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Timeframe untuk strategi utama
    timeframe = '5m'
    
    # Timeframe informative untuk konfirmasi trend
    informative_timeframe = '15m'

    # ROI table - Conservative approach untuk futures
    minimal_roi = {
        "0": 0.15,   # 15% profit untuk exit segera
        "30": 0.08,  # 8% setelah 30 menit
        "60": 0.05,  # 5% setelah 1 jam
        "120": 0.03, # 3% setelah 2 jam
        "240": 0.02, # 2% setelah 4 jam
        "480": 0.01  # 1% setelah 8 jam
    }

    # Stoploss - Ketat untuk mengurangi drawdown
    stoploss = -0.04  # 4% stop loss

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02    # Mulai trailing di +2%
    trailing_stop_positive_offset = 0.025  # Offset 0.5% dari entry
    trailing_only_offset_is_reached = True

    # Futures leverage (hati-hati dengan leverage tinggi)
    leverage_num = 3

    # Position sizing
    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    # Protections
    protections = [
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 50,
            "trade_limit": 2,
            "stop_duration_candles": 20,
            "only_per_pair": False
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 200,
            "trade_limit": 10,
            "stop_duration_candles": 100,
            "max_allowed_drawdown": 0.15
        },
        {
            "method": "LowProfitPairs",
            "lookback_period_candles": 400,
            "trade_limit": 8,
            "stop_duration_candles": 60,
            "required_profit": -0.02
        }
    ]

    # Hyperopt parameters
    # Entry parameters
    rsi_buy_threshold = IntParameter(25, 40, default=32, space="buy")
    rsi_sell_threshold = IntParameter(60, 75, default=68, space="sell")
    
    bb_buy_threshold = DecimalParameter(0.98, 1.02, default=1.0, space="buy")
    bb_sell_threshold = DecimalParameter(0.98, 1.02, default=1.0, space="sell")
    
    adx_threshold = IntParameter(20, 35, default=25, space="buy")
    
    # Volume parameters
    volume_factor_buy = DecimalParameter(1.2, 2.5, default=1.8, space="buy")
    volume_factor_sell = DecimalParameter(1.2, 2.5, default=1.8, space="sell")

    # EMA parameters
    ema_fast = IntParameter(8, 21, default=12, space="buy")
    ema_slow = IntParameter(21, 50, default=34, space="buy")
    
    # MACD parameters
    macd_fast = IntParameter(8, 15, default=12, space="buy")
    macd_slow = IntParameter(21, 30, default=26, space="buy")
    macd_signal = IntParameter(7, 12, default=9, space="buy")

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], 
                 side: str, **kwargs) -> float:
        return self.leverage_num

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Dapatkan data timeframe yang lebih tinggi
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], 
                                                timeframe=self.informative_timeframe)
        informative = self.populate_indicators_informative(informative, metadata)
        
        dataframe = merge_informative_pair(dataframe, informative, 
                                         self.timeframe, self.informative_timeframe, ffill=True)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_middle'] = bollinger['mid']
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']

        # EMAs
        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=self.ema_fast.value)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=self.ema_slow.value)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast.value, 
                      slowperiod=self.macd_slow.value, signalperiod=self.macd_signal.value)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # ADX untuk mengukur kekuatan trend
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe, timeperiod=14)

        # Volume indicators
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']

        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']

        # Williams %R
        dataframe['williams_r'] = ta.WILLR(dataframe, timeperiod=14)

        # Support and Resistance levels
        dataframe['pivot'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        dataframe['r1'] = 2 * dataframe['pivot'] - dataframe['low']
        dataframe['s1'] = 2 * dataframe['pivot'] - dataframe['high']

        # ATR untuk volatilitas
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = (dataframe['atr'] / dataframe['close']) * 100

        # Heikin Ashi untuk smooth trend
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        return dataframe

    def populate_indicators_informative(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Indikator untuk timeframe yang lebih tinggi (konfirmasi trend)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['rsi_higher'] = ta.RSI(dataframe, timeperiod=14)
        
        # Trend direction
        dataframe['trend_up'] = (dataframe['ema_50'] > dataframe['ema_100']) & \
                               (dataframe['close'] > dataframe['ema_50'])
        dataframe['trend_down'] = (dataframe['ema_50'] < dataframe['ema_100']) & \
                                 (dataframe['close'] < dataframe['ema_50'])
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long = [
            # Trend confirmation dari timeframe lebih tinggi
            (dataframe[f'trend_up_{self.informative_timeframe}'] == True),
            
            # RSI oversold tapi tidak terlalu dalam
            (dataframe['rsi'] > self.rsi_buy_threshold.value),
            (dataframe['rsi'] < 50),
            
            # Price near lower Bollinger Band
            (dataframe['bb_percent'] < self.bb_buy_threshold.value),
            (dataframe['bb_percent'] > 0.1),  # Tidak terlalu dekat dengan lower band
            
            # EMA crossover atau price above fast EMA
            ((dataframe['ema_fast'] > dataframe['ema_slow']) | 
             (dataframe['close'] > dataframe['ema_fast'])),
            
            # MACD bullish
            (dataframe['macd'] > dataframe['macdsignal']),
            (dataframe['macdhist'] > 0),
            
            # ADX menunjukkan trend yang kuat
            (dataframe['adx'] > self.adx_threshold.value),
            (dataframe['di_plus'] > dataframe['di_minus']),
            
            # Volume confirmation
            (dataframe['volume_ratio'] > self.volume_factor_buy.value),
            
            # Price above 200 EMA (long term trend)
            (dataframe['close'] > dataframe['ema_200']),
            
            # Stochastic tidak overbought
            (dataframe['stoch_k'] < 80),
            
            # Williams %R oversold
            (dataframe['williams_r'] < -20),
            (dataframe['williams_r'] > -80),
            
            # Volatilitas tidak terlalu tinggi
            (dataframe['atr_percent'] < 5),
            
            # Heikin Ashi bullish
            (dataframe['ha_close'] > dataframe['ha_open'])
        ]

        conditions_short = [
            # Trend confirmation dari timeframe lebih tinggi
            (dataframe[f'trend_down_{self.informative_timeframe}'] == True),
            
            # RSI overbought tapi tidak terlalu dalam
            (dataframe['rsi'] < self.rsi_sell_threshold.value),
            (dataframe['rsi'] > 50),
            
            # Price near upper Bollinger Band
            (dataframe['bb_percent'] > self.bb_sell_threshold.value),
            (dataframe['bb_percent'] < 0.9),  # Tidak terlalu dekat dengan upper band
            
            # EMA crossover atau price below fast EMA
            ((dataframe['ema_fast'] < dataframe['ema_slow']) | 
             (dataframe['close'] < dataframe['ema_fast'])),
            
            # MACD bearish
            (dataframe['macd'] < dataframe['macdsignal']),
            (dataframe['macdhist'] < 0),
            
            # ADX menunjukkan trend yang kuat
            (dataframe['adx'] > self.adx_threshold.value),
            (dataframe['di_minus'] > dataframe['di_plus']),
            
            # Volume confirmation
            (dataframe['volume_ratio'] > self.volume_factor_sell.value),
            
            # Price below 200 EMA (long term trend)
            (dataframe['close'] < dataframe['ema_200']),
            
            # Stochastic tidak oversold
            (dataframe['stoch_k'] > 20),
            
            # Williams %R overbought
            (dataframe['williams_r'] > -20),
            (dataframe['williams_r'] < -80),
            
            # Volatilitas tidak terlalu tinggi
            (dataframe['atr_percent'] < 5),
            
            # Heikin Ashi bearish
            (dataframe['ha_close'] < dataframe['ha_open'])
        ]

        # Long entries
        dataframe.loc[
            (
                reduce(lambda x, y: x & y, conditions_long)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'long_entry')

        # Short entries
        dataframe.loc[
            (
                reduce(lambda x, y: x & y, conditions_short)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'short_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_exit_long = [
            # RSI overbought
            (dataframe['rsi'] > 70) |
            
            # Price near upper Bollinger Band
            (dataframe['bb_percent'] > 0.85) |
            
            # MACD bearish crossover
            ((dataframe['macd'] < dataframe['macdsignal']) & 
             (dataframe['macdhist'] < 0)) |
            
            # Stochastic overbought
            (dataframe['stoch_k'] > 80) |
            
            # Williams %R overbought
            (dataframe['williams_r'] > -20) |
            
            # Trend berubah di timeframe lebih tinggi
            (dataframe[f'trend_down_{self.informative_timeframe}'] == True)
        ]

        conditions_exit_short = [
            # RSI oversold
            (dataframe['rsi'] < 30) |
            
            # Price near lower Bollinger Band
            (dataframe['bb_percent'] < 0.15) |
            
            # MACD bullish crossover
            ((dataframe['macd'] > dataframe['macdsignal']) & 
             (dataframe['macdhist'] > 0)) |
            
            # Stochastic oversold
            (dataframe['stoch_k'] < 20) |
            
            # Williams %R oversold
            (dataframe['williams_r'] < -80) |
            
            # Trend berubah di timeframe lebih tinggi
            (dataframe[f'trend_up_{self.informative_timeframe}'] == True)
        ]

        # Exit long positions
        dataframe.loc[
            (
                reduce(lambda x, y: x | y, conditions_exit_long)
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'exit_long_signal')

        # Exit short positions
        dataframe.loc[
            (
                reduce(lambda x, y: x | y, conditions_exit_short)
            ),
            ['exit_short', 'exit_tag']
        ] = (1, 'exit_short_signal')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic stoploss berdasarkan ATR dan profit
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # ATR-based dynamic stop
        atr_multiplier = 2.0
        if trade.is_short:
            atr_stop = (current_rate + (last_candle['atr'] * atr_multiplier)) / trade.open_rate - 1
        else:
            atr_stop = (current_rate - (last_candle['atr'] * atr_multiplier)) / trade.open_rate - 1
        
        # Trailing stop yang lebih agresif saat profit tinggi
        if current_profit > 0.05:  # 5% profit
            trailing_stop = -0.02  # 2% trailing
        elif current_profit > 0.03:  # 3% profit
            trailing_stop = -0.025  # 2.5% trailing
        else:
            trailing_stop = self.stoploss
        
        return max(atr_stop, trailing_stop)

    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                            current_rate: float, current_profit: float, 
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        """
        Position adjustment untuk averaging down/up
        """
        if current_profit > -0.02:  # Hanya adjust jika loss tidak lebih dari 2%
            return None
            
        # Hanya adjust sekali
        if trade.nr_of_successful_entries > 1:
            return None
            
        # Check if trend masih sesuai
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if trade.is_short:
            if not last_candle[f'trend_down_{self.informative_timeframe}']:
                return None
        else:
            if not last_candle[f'trend_up_{self.informative_timeframe}']:
                return None
        
        # Adjust dengan ukuran yang lebih kecil
        return trade.stake_amount * 0.5

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Konfirmasi final sebelum entry
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if len(dataframe) < 1:
            return False
            
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Check volatilitas tidak terlalu tinggi
        if last_candle['atr_percent'] > 8:
            return False
            
        # Check volume
        if last_candle['volume_ratio'] < 1.2:
            return False
            
        return True