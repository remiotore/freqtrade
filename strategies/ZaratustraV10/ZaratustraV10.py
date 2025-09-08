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


class ZaratustraV10(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    inf_times = ["5m", "15m", "30m",]
    can_short = True
    use_exit_signal = False
    exit_profit_only = True
    exit_profit_offset = 0.5
    
    # Buy hyperspace params:
    buy_params = {
        "base_leverage": 10,
    }

    # ROI table:
    minimal_roi = {
        "0": 1.0
    }

    # Stoploss:
    stoploss = -0.20

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 10

    # Hyperparameters
    base_leverage = IntParameter(1, 50, default=3, space="buy")
    
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

            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_time, ffill=True)

        return dataframe

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return self.base_leverage.value
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['adx_30m'] > dataframe['mdi_30m']) &
                (dataframe['adx_15m'] > dataframe['mdi_15m']) &
                (dataframe['adx_5m']  > dataframe['mdi_5m']) &
                (dataframe['pdi_30m'] > dataframe['mdi_30m']) &
                (dataframe['pdi_15m'] > dataframe['mdi_15m']) &
                (dataframe['pdi_5m']  > dataframe['mdi_5m'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Bullish trend')

        dataframe.loc[
            (
                (dataframe['adx_30m'] > dataframe['pdi_30m']) &
                (dataframe['adx_15m'] > dataframe['pdi_15m']) &
                (dataframe['adx_5m']  > dataframe['pdi_5m']) &
                (dataframe['mdi_30m'] > dataframe['pdi_30m']) &
                (dataframe['mdi_15m'] > dataframe['pdi_15m']) &
                (dataframe['mdi_5m']  > dataframe['pdi_5m'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Bearish trend')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe