from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class MACDCrossoverWithTrend(IStrategy):

    """
    MACDCrossoverWithTrend
    author@: Paul Csapak
    github@: https://github.com/paulcpk/freqtrade-strategies-that-work

    How to use it?

    > freqtrade download-data --timeframes 1h --timerange=20180301-20200301
    > freqtrade backtesting --export trades -s MACDCrossoverWithTrend --timeframe 1h --timerange=20180301-20200301
    > freqtrade plot-dataframe -s MACDCrossoverWithTrend --indicators1 ema100 --timeframe 1h --timerange=20180301-20200301

    """









    stoploss = -0.2

    timeframe = '1h'

    trailing_stop = False
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.04

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe['macd'] < 0) &  # MACD is below zero

                (qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])) &
                (dataframe['low'] > dataframe['ema100']) &  # Candle low is above EMA

                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_below(dataframe['macd'], 0)) | 
                (dataframe['low'] < dataframe['ema100'])  # OR price is below trend ema
            ),
            'sell'] = 1
        return dataframe
