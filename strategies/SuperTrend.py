


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import pandas_ta as pd_ta


class SuperTrend(IStrategy):
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

    timeframe = '1m'


    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }


    stoploss = -0.02

    trailing_stop = False




    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

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
    
    @property
    def plot_config(self):
        return {

            'main_plot': {
                'ema12': {'color': 'green'},
                'ema50': {'color': 'red'}
            }
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        period = 7
        atr_mult = 3.0

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['ema12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['overbought'] = 70
        dataframe['oversold'] = 30

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )

        dataframe['ST_long'] = pd_ta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=period, multiplier=atr_mult)[f'SUPERTl_{period}_{atr_mult}']
        dataframe['ST_short'] = pd_ta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=period, multiplier=atr_mult)[f'SUPERTs_{period}_{atr_mult}']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (

                (dataframe['ema12'] > dataframe['ema50']) & 
                (dataframe['rsi'] < 30) &
                (dataframe['close'] < dataframe['bb_lowerband']) &

                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

            ),
            'sell'] = 1
        return dataframe
    