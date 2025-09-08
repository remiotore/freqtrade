import datetime
import numpy as np
import pandas_ta as pta
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from technical import qtpylib
from scipy.stats import linregress
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, IntParameter, IStrategy)
from freqtrade.persistence import Trade
from typing import Optional, Tuple, Union


class SlopeV4(IStrategy):
    INTERFACE_VERSION = 3
    
    can_short = True
    timeframe = '15m'
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.1

    buy_params = {
        "minus_di": 25,
        "plus_di": 57,
        "volume_long": 2.312,
        "window": 24,  # value loaded from strategy
    }

    sell_params = {
        "volume_short": 5.143,
    }

    minimal_roi = {
        "0": 0.398,
        "99": 0.11,
        "209": 0.03,
        "284": 0
    }

    stoploss = -0.2  # value loaded from strategy

    trailing_stop = False  # value loaded from strategy
    trailing_stop_positive = None  # value loaded from strategy
    trailing_stop_positive_offset = 0.0  # value loaded from strategy
    trailing_only_offset_is_reached = False  # value loaded from strategy

    max_open_trades = -1

    window = IntParameter(1, 120, space='buy',  default=buy_params['window'], optimize=False)
    minus_di = IntParameter(1, 100, space='buy',  default=buy_params['minus_di'], optimize=True)
    plus_di = IntParameter(1, 100, space='buy',  default=buy_params['plus_di'], optimize=True)
    volume_long = DecimalParameter(0.0, 100.0, space='buy',  default=buy_params['volume_long'], optimize=True)
    volume_short = DecimalParameter(0.0, 100.0, space='sell', default=sell_params['volume_short'], optimize=True)

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
            },
            'subplots' : {
                'Directional Indicator' : {
                    'mid_di'   : { 'color' : 'black' },
                    'plus_di'  : { 'color' : 'red' },
                    'minus_di' : { 'color' : 'blue' },
                },
                'Volume %' : {
                    'volume_pct' : { 'color' : 'black' },
                },
            }
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.window.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.window.value)
        dataframe['mid_di'] = dataframe['plus_di'] - dataframe['minus_di']
        dataframe['volume_pct'] = dataframe['volume'].pct_change(self.window.value)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.volume_long.value) &
                (dataframe['minus_di']   < self.minus_di.value) &
                (dataframe['plus_di']    > self.plus_di.value) &
                (dataframe['volume']     > 0)
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['volume_pct'] > self.volume_short.value) &
                (dataframe['minus_di']   > self.minus_di.value) &
                (dataframe['plus_di']    < self.plus_di.value) &
                (dataframe['volume']     > 0)
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['minus_di'].shift(2) < dataframe['minus_di'].shift(1)) &
                (dataframe['minus_di'].shift(1) > dataframe['minus_di']) &
                (dataframe['volume'] > 0)
            ),
        'exit_long'] = 1

        dataframe.loc[
            (

                (dataframe['minus_di'].shift(2) > dataframe['minus_di'].shift(1)) &
                (dataframe['minus_di'].shift(1) < dataframe['minus_di']) &
                (dataframe['volume'] > 0)
            ),
        'exit_short'] = 1

        return dataframe