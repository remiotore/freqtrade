# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TapolV1(IStrategy):

    timeframe = '5m'

    stoploss = -0.10

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.077
    trailing_stop_positive_offset = 0.175
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (

                (dataframe['ema7'] > dataframe['bb_middleband']) &
                (dataframe['macd'] > dataframe['macdsignal']) &
                (
                    (dataframe['close'] > dataframe['ema7']) |
                    (dataframe['close'].shift(1) > dataframe['ema7']) |
                    (dataframe['close'].shift(2) > dataframe['ema7'])
                )
            ),

            'buy'] = 1

        return dataframe
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    (dataframe['ema7'] < dataframe['bb_middleband']) &
                    (dataframe['macd'] < dataframe['macdsignal']) &
                    (
                        (dataframe['close'] < dataframe['ema7']) |
                        (dataframe['close'].shift(1) < dataframe['ema7'])
                    )
                ) |
                (
                    (dataframe['ema7'] > dataframe['bb_middleband']) &
                    (dataframe['macd'] > dataframe['macdsignal']) &
                    (
                        (dataframe['close'] > dataframe['bb_upperband']) |
                        (dataframe['close'].shift(1) > dataframe['bb_upperband']) 
                    )
                )
            ),
            'sell'] = 1

        return dataframe









