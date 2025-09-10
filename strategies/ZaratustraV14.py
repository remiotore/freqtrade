# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
from freqtrade.constants import Config
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, IntParameter, DecimalParameter
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Dict, List, Optional, Union, Tuple
import talib.abstract as ta
from technical import qtpylib



class ZaratustraV14(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = False
    exit_profit_only = True
    
    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.16

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.107
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 9

    # Hyper Parameters
    base_atr = DecimalParameter(0.00, 1.00, default=0.2, decimals=2, space="buy")

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {}
        plot_config['subplots'] = {
            "DI": {
                'dx' : {'color': 'yellow'},
                'adx': {'color': 'orange'},
                'pdi': {'color': 'green'},
                'mdi': {'color': 'red'},
            },
            "ATR": {
                'atr': {'color': 'red'}
            }
        }

        return plot_config

    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10
        
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['atr']  = ta.ATR(dataframe)
        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['mdi']) &
                (dataframe['pdi'] > dataframe['mdi']) &
                (dataframe['atr'] > self.base_atr.value)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['bb_upperband'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long Bollinger enter')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['pdi']) &
                (dataframe['mdi'] > dataframe['pdi']) &
                (dataframe['atr'] > self.base_atr.value)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['close'], dataframe['bb_lowerband'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short Bollinger enter')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe