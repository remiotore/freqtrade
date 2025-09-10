


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)
from pandas_ta.utils import data


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas_ta as pd_ta


class SuperTrend_154(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 100
    }


    stoploss = -0.10

    trailing_stop = False




    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {

        'main_plot': {
            'ST_long': {'color': 'green'},
            'ST_short': {'color': 'red'}
            
        }
    }
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        periodo = 7
        atr_multiplicador = 3.0

        dataframe['ST_long'] = pd_ta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=periodo, 
                                                multiplier=atr_multiplicador)[f'SUPERTl_{periodo}_{atr_multiplicador}']
        dataframe['ST_short'] = pd_ta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=periodo, 
                                                multiplier=atr_multiplicador)[f'SUPERTs_{periodo}_{atr_multiplicador}']
    
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['ST_long'] < dataframe['close']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['ST_short'] > dataframe['close']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    