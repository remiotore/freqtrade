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



class ZaratustraV20(IStrategy):
    # Parameters
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = True
    
    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.99

    # Max Open Trades:
    max_open_trades = -1

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {}
        plot_config['subplots'] = {
            'DI' : {
                'DX' : {'color' : 'yellow'},
                'ADX': {'color' : 'orange'},
                'PDI': {'color' : 'green'},
                'MDI': {'color' : 'red'},
            },
        }

        return plot_config

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['DX']  = ta.SMA(ta.DX(dataframe)       * dataframe['volume'] ) / ta.SMA(dataframe['volume'])
        dataframe['ADX'] = ta.SMA(ta.ADX(dataframe)      * dataframe['volume'] ) / ta.SMA(dataframe['volume'])
        dataframe['PDI'] = ta.SMA(ta.PLUS_DI(dataframe)  * dataframe['volume'] ) / ta.SMA(dataframe['volume'])
        dataframe['MDI'] = ta.SMA(ta.MINUS_DI(dataframe) * dataframe['volume'] ) / ta.SMA(dataframe['volume'])
        dataframe['DDI'] = abs(dataframe['PDI'] - dataframe['MDI'])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Positive trend
                (dataframe['PDI'] > dataframe['MDI']) &
                # ADX between PDI and MDI
                (dataframe['ADX'] > dataframe['MDI']) &
                (dataframe['ADX'] < dataframe['PDI']) &
                # Main event!
                (qtpylib.crossed_above(dataframe['DX'], dataframe['ADX']))
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'Long DI enter')

        dataframe.loc[
            (
                # Negative trend
                (dataframe['MDI'] > dataframe['PDI']) &
                # ADX between PDI and MDI
                (dataframe['ADX'] > dataframe['PDI']) &
                (dataframe['ADX'] < dataframe['MDI']) &
                # Main event!
                (qtpylib.crossed_above(dataframe['DX'], dataframe['ADX']))
            ),
            ['enter_short', 'enter_tag']
        ] = (1, 'Short DI enter')
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # End of the event!
                (qtpylib.crossed_below(dataframe['DX'], dataframe['ADX'])) |
                # Bad moments for trading...
                (dataframe['DX'] <= dataframe['MDI']) |
                (dataframe['DX'] <= dataframe['PDI']) |
                # Trend reversion...
                (dataframe['DX'] <= dataframe['DX'].shift(1))
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'Long DI exit')

        dataframe.loc[
            (
                # End of the event!
                (qtpylib.crossed_below(dataframe['DX'], dataframe['ADX'])) |
                # Bad moments for trading...
                (dataframe['DX'] <= dataframe['MDI']) |
                (dataframe['DX'] <= dataframe['PDI']) |
                # Trend reversion...
                (dataframe['DX'] <= dataframe['DX'].shift(1))
            ),
            ['exit_long', 'exit_tag']
        ] = (1, 'Long DI exit')

        return dataframe
    
    def leverage(self, pair: str, current_time: "datetime", current_rate: float, proposed_leverage: float, max_leverage: float, side: str, **kwargs,) -> float:
        return 10