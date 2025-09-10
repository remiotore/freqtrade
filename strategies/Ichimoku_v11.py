from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

from technical.util import resample_to_interval, resampled_merge
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy
from technical.indicators import ichimoku

class Ichimoku_v11(IStrategy):
    """

    """

    minimal_roi = {
        "0": 100
    }

    stoploss = -1 #-0.35

    ticker_interval = '4h' #3m







    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        ichi = ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)

        dataframe['tenkan'] = ichi['tenkan_sen']
        dataframe['kijun'] = ichi['kijun_sen']
        dataframe['senkou_a'] = ichi['senkou_span_a']
        dataframe['senkou_b'] = ichi['senkou_span_b']
        dataframe['cloud_green'] = ichi['cloud_green']
        dataframe['cloud_red'] = ichi['cloud_red']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['senkou_a'])) &
                (dataframe['close'] > dataframe['senkou_a']) &
                (dataframe['close'] > dataframe['senkou_b'])
            ),
            'buy'] = 1

        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['senkou_b'])) &
                (dataframe['close'] > dataframe['senkou_a']) &
                (dataframe['close'] > dataframe['senkou_b'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
















        return dataframe
