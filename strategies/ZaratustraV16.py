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



class ZaratustraV16(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = False
    
    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.15

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.107
    trailing_only_offset_is_reached = True

    # Max Open Trades:
    max_open_trades = 7

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
            "AROON": {
                'aup': { 'color': 'green' },
                'ado': { 'color': 'red' }
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['dx']  = ta.DX(dataframe)
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['pdi'] = ta.PLUS_DI(dataframe)
        dataframe['mdi'] = ta.MINUS_DI(dataframe)
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe[['aup', 'ado']] = ta.AROON(dataframe)[['aroonup','aroondown']]
        dataframe[['bbl','bbm','bbu']] = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=8, stds=2)[['lower','mid','upper']]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['close'], dataframe['bbu'])
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long Bollinger enter')

        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['close'], dataframe['bbl'])
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short Bollinger enter')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['mdi']) &
                (dataframe['adx'] > dataframe['mdi']) &
                (dataframe['pdi'] > dataframe['mdi']) &
                (dataframe['aup'] > dataframe['ado']) &
                (dataframe['aup'] > 50) &
                (dataframe['mfi'] > 50)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')

        dataframe.loc[
            (
                (dataframe['dx']  > dataframe['pdi']) &
                (dataframe['adx'] > dataframe['pdi']) &
                (dataframe['mdi'] > dataframe['pdi']) &
                (dataframe['ado'] > dataframe['aup']) &
                (dataframe['ado'] > 50) &
                (dataframe['mfi'] < 50)
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe
    
    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10