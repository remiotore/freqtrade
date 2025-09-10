from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
import numpy  # noqa


class RSIDirectionalWithTrendSlow(IStrategy):

    """
    RSIDirectionalWithTrendSlow
    author@: Paul Csapak
    github@: https://github.com/paulcpk/freqtrade-strategies-that-work

    How to use it?

    > freqtrade download-data --timeframes 1h --timerange=20180301-20200301
    > freqtrade backtesting --export trades -s DoubleEMACrossoverWithTrend --timeframe 1h --timerange=20180301-20200301
    > freqtrade plot-dataframe -s DoubleEMACrossoverWithTrend --indicators1 ema600 --timeframe 1h --timerange=20180301-20200301

    """

    timeframe = '1h'









    stoploss = -0.2

    trailing_stop = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=10)
        dataframe['ema600'] = ta.EMA(dataframe, timeperiod=600)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_above(dataframe['rsi_slow'], 25)) &
                (dataframe['low'] > dataframe['ema600']) &  # Candle low is above EMA

                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_below(dataframe['rsi_slow'], 20)) |

                (dataframe['low'] < dataframe['ema600'])
            ),
            'sell'] = 1
        return dataframe
