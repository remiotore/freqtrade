# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from functools import reduce
import json
from pathlib import Path

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


# ==========================================
# Boost Your Algo Trading with Freqtrade Parabolic SAR Strategy!
# YouTube Link: https://youtu.be/Ssvm9MVqPSE
# ==========================================


# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/config_binance_futures_ParabolicSAR.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================
"""
freqtrade hyperopt \
    --strategy ParabolicSAR \
    --config user_data/config_binance_futures_ParabolicSAR.json \
    --timeframe 30m \
    --timerange 20231001-20240501 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 10 \
    -p GRT/USDT:USDT \
    --max-open-trades 1
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy ParabolicSAR \
    --timeframe 30m \
    --timerange 20231001-20241001 \
    --breakdown month \
    -c user_data/config_binance_futures_ParabolicSAR.json \
    --max-open-trades 1 \
    --timeframe-detail 5m \
    -p GRT/USDT:USDT \
    --cache none
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/config_binance_futures_ParabolicSAR.json
"""


class ParabolicSAR(IStrategy):
            
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "30m"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.20

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured
    
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Hyperoptable parameters
    ema_length = CategoricalParameter([50, 100, 150, 200], default=200, space="buy", optimize=False)

    rsi_upper_threshold = CategoricalParameter([55, 60, 65], default=60, space="buy", optimize=True)
    rsi_lower_threshold = CategoricalParameter([45, 40, 35], default=40, space="buy", optimize=True)
    rsi_length = CategoricalParameter([7, 14, 28], default=14, space="buy", optimize=True)    

    stop_loss = DecimalParameter(0.01, 0.05, default=0.01, decimals=2, space='buy')
    risk_ratio = CategoricalParameter([1, 1.5, 2], default=2, space="buy")
    
    leverage_level = 1

    @property
    def plot_config(self):

        plot_config = {}
        plot_config['main_plot'] = {
            f'ema_{self.ema_length.value}': {'color': '#2962ff'},
            'sar': {
                'color': '#dedede',
                'type': 'scatter',
                'scatterSymbolSize': 4
            }
        }
        plot_config['subplots'] = {
            'rsi': {
                f'rsi_{self.rsi_length.value}': {'color': '#7e57c2'},
            }
        }
        
        return plot_config
    
    
    def informative_pairs(self):
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
        
        # get access to all pairs available in whitelist.
        # pairs = self.dp.current_whitelist()

        # # Assign tf to each pair so they can be downloaded and cached for strategy.
        # informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        
        return []
    
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe["sar"] = ta.SAR(dataframe)
        
        for val in self.rsi_length.range:
            dataframe[f'rsi_{val}'] = ta.RSI(dataframe, timeperiod=val)
        
        for period in self.ema_length.range:
            dataframe[f'ema_{period}'] = ta.EMA(dataframe, timeperiod=period)
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['sar']) &
                (qtpylib.crossed_above(dataframe[f'rsi_{self.rsi_length.value}'], self.rsi_upper_threshold.value)) &
                (dataframe['close'] > dataframe[f'ema_{self.ema_length.value}']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1
        
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['sar']) &
                (qtpylib.crossed_below(dataframe[f'rsi_{self.rsi_length.value}'], self.rsi_lower_threshold.value)) &
                (dataframe['close'] < dataframe[f'ema_{self.ema_length.value}']) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1
        
        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
                
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        return dataframe
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        
        take_profit = self.stop_loss.value * self.risk_ratio.value

        if current_profit > take_profit: 
            return 'Take Profit'
        
        if current_profit < -self.stop_loss.value: 
            return 'Stop Loss'
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level