from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class DoubleEMACrossoverWithTrend_37(IStrategy):

    """
    DoubleEMACrossoverWithTrend
    author@: Paul Csapak
    github@: https://github.com/paulcpk/freqtrade-strategies-that-work
    How to use it?
    > freqtrade download-data --timeframes 1h --timerange=20180301-20200301
    > freqtrade backtesting --export trades -s DoubleEMACrossoverWithTrend --timeframe 1h --timerange=20180301-20200301
    > freqtrade plot-dataframe -s DoubleEMACrossoverWithTrend --indicators1 ema200 --timeframe 1h --timerange=20180301-20200301
    """









    stoploss = -0.2

    timeframe = '1h'

    trailing_stop = False
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.04

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_above(dataframe['ema9'], dataframe['ema21'])) &
                (dataframe['low'] > dataframe['ema200']) &  # Candle low is above EMA

                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_below(dataframe['ema9'], dataframe['ema21'])) |
                (dataframe['low'] < dataframe['ema200']) # OR price is below trend ema
            ),
            'sell'] = 1
        return dataframe