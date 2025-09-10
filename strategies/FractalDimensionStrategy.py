from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, List
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.persistence import Trade, Order
from freqtrade.strategy import stoploss_from_open
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import merge_informative_pair, CategoricalParameter, DecimalParameter, IntParameter, stoploss_from_open


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Optional, Union  # noqa
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import IntParameter
from freqtrade.strategy import CategoricalParameter
from pandas import DataFrame, Series
import pandas_ta as pta
import pandas as pd
import numpy
import numpy as np
from math import inf
import math
from warnings import simplefilter
from technical.indicators import ichimoku
from numpy import cos as npCos
from numpy import exp as npExp
from numpy import pi as npPi
from numpy import sqrt as npSqrt
from pandas_ta.utils import get_offset, verify_series

class FractalDimensionStrategy(IStrategy):


    minimal_roi = {
        "0": 0.1
    }


    stoploss = -0.02

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.015
    trailing_only_offset_is_reached = True

    fractal_dimension_buy = 1.1
    fractal_dimension_sell = 1.4
    rsi_buy = 50
    rsi_sell = 80
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several indicators to the given DataFrame.
        """
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe["rsi_fast"] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=25)
        dataframe['emarsi'] = ta.EMA(dataframe['rsi'], timeperiod=9)
        dataframe['emarsi2'] = ta.EMA(dataframe['rsi'], timeperiod=3)
        dataframe["hma50"] = qtpylib.hull_moving_average(dataframe["close"], window=50)
        dataframe["hma200"] = qtpylib.hull_moving_average(dataframe["close"], window=200)
        dataframe['ema9'] = ta.EMA(dataframe,timeperiod = 9)
        dataframe['ema3'] = ta.EMA(dataframe,timeperiod = 3)
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"] 

        period = 14
        smoothD = 3
        SmoothK = 3
        stochrsi = (dataframe["rsi"] - dataframe["rsi"].rolling(period).min()) / (
            dataframe["rsi"].rolling(period).max()
            - dataframe["rsi"].rolling(period).min()
        )
        dataframe["srsi_k"] = stochrsi.rolling(SmoothK).mean() * 100
        dataframe["srsi_d"] = dataframe["srsi_k"].rolling(smoothD).mean()

        weighted_bollinger = qtpylib.weighted_bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2
        )

        weighted_bollinger2 = qtpylib.weighted_bollinger_bands(
            qtpylib.typical_price(dataframe), window=10, stds=2
        )

        dataframe["bb_upperband2"] = weighted_bollinger2["upper"]
        dataframe["bb_upperband"] = weighted_bollinger["upper"]
        dataframe["bb_lowerband"] = weighted_bollinger["lower"]
        dataframe["bb_middleband"] = weighted_bollinger["mid"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )

        dataframe['fractal_dimension'] = self.calculate_fractal_dimension(dataframe)

        return dataframe

    def calculate_fractal_dimension(self, dataframe: DataFrame) -> Series:
        """
        Calculates the Fractal Dimension of the price series.
        """
        fractal_dimension = []
        for i in range(len(dataframe)):
            if i < 10:
                fractal_dimension.append(1.0)
            else:
                x = np.arange(len(dataframe.iloc[i-9:i+1]['close']))
                y = dataframe.iloc[i-9:i+1]['close']
                model = LinearRegression()
                model.fit(x.reshape(-1, 1), y)
                fractal_dimension.append(2 - model.coef_[0])
        return Series(fractal_dimension, index=dataframe.index)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        """

        buy = (

            (dataframe['rsi'] < dataframe['rsi_fast']) & (dataframe['rsi'] > dataframe['rsi_slow'] ) & (dataframe['macdhist'] > 0) 
            & (dataframe['volume'] > dataframe['volume'].shift(1).rolling(12).max()))





        dataframe.loc[
            (
                buy & ((buy).shift(1).rolling(10).max() == 0)
            ),
            'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        """
        dataframe.loc[
            (

                (dataframe['rsi'] > 96)
            ),
            'sell'] = 1

        return dataframe