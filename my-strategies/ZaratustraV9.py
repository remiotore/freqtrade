# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from freqtrade.persistence import Trade
from typing import Dict, Optional, Union, Tuple
from freqtrade.strategy import IStrategy, merge_informative_pair, IntParameter
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class ZaratustraV9(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    inf_times = ["5m", "15m", "30m",]
    
    can_short = True
    
    use_exit_signal = False

    exit_profit_only = True
    exit_profit_offset = 0.01
    
    # ROI table:
    minimal_roi = {
        "0": 1.0,
    }

    # Stoploss:
    stoploss = -0.296

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.013
    trailing_stop_positive_offset = 0.071
    trailing_only_offset_is_reached = True

    # Hyperparameters
    base_leverage = IntParameter(1, 50, default=8, space="buy")
    
    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for inf_time in self.inf_times:
            for pair in pairs:
                informative_pairs.append((pair, inf_time))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.dp:
            return dataframe

        for inf_time in self.inf_times:
            informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_time)
            
            informative['adx'] = ta.ADX(informative)
            informative['pdi'] = ta.PLUS_DI(informative)
            informative['mdi'] = ta.MINUS_DI(informative)

            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
            informative['bbu'] = bollinger['upper']
            informative['bbm'] = bollinger['mid']
            informative['bbl'] = bollinger['lower']

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_time, ffill=True)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['signal'] = macd['macdsignal']

        dataframe['min'] = (dataframe['close'] == dataframe['close'].rolling(15).min()).astype(int)
        dataframe['max'] = (dataframe['close'] == dataframe['close'].rolling(15).max()).astype(int)
        dataframe['min_check'] = dataframe['min'].rolling(5).apply(lambda x: x.all(), raw=True).fillna(0).astype(int)
        dataframe['max_check'] = dataframe['max'].rolling(5).apply(lambda x: x.all(), raw=True).fillna(0).astype(int)

        return dataframe

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return self.base_leverage.value
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # ADX
                (dataframe['adx_30m'] > 25) &
                (dataframe['adx_15m'] > 25) &
                (dataframe['adx_5m']  > 25) &
                # Directional Indicator
                (dataframe['pdi_30m'] > 25) &
                (dataframe['pdi_15m'] > 25) &
                (dataframe['pdi_5m']  > 25) &
                # Bollinger Bands
                (dataframe['close_30m'] > dataframe['bbm_30m']) &
                (dataframe['close_15m'] > dataframe['bbm_15m']) &
                (dataframe['close_5m']  > dataframe['bbm_5m']) &
                # MACD
                (dataframe['macd']   > dataframe['signal']) & 
                (dataframe['signal'] > 0) & 
                (dataframe['macd']   > 0)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend')

        dataframe.loc[
            (
                # ADX
                (dataframe['adx_30m'] > 25) &
                (dataframe['adx_15m'] > 25) &
                (dataframe['adx_5m']  > 25) &
                # Directional Indicator
                (dataframe['mdi_30m'] > 25) &
                (dataframe['mdi_15m'] > 25) &
                (dataframe['mdi_5m']  > 25) &
                # Bollinger Bands
                (dataframe['close_30m'] < dataframe['bbm_30m']) &
                (dataframe['close_15m'] < dataframe['bbm_15m']) &
                (dataframe['close_5m']  < dataframe['bbm_5m']) &
                # MACD
                (dataframe['macd']   < dataframe['signal']) & 
                (dataframe['signal'] < 0) & 
                (dataframe['macd']   < 0)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe