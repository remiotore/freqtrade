from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.exchange import timeframe_to_minutes
import numpy  # noqa


class RSIDirectionalWithTrend(IStrategy):

    """
    RSIDirectionalWithTrend
    author@: Paul Csapak
    github@: https://github.com/paulcpk/freqtrade-strategies-that-work

    How to use it?

    > freqtrade download-data --timeframes 1h --timerange=20180301-20200301
    > freqtrade backtesting --export trades -s DoubleEMACrossoverWithTrend --timeframe 1h --timerange=20180301-20200301
    > freqtrade plot-dataframe -s DoubleEMACrossoverWithTrend --indicators1 ema100 --timeframe 1h --timerange=20180301-20200301

    """

    timeframe = '1h'









    stoploss = -0.1

    trailing_stop = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_above(dataframe['rsi'], 15)) &
                (dataframe['low'] > dataframe['ema100']) &  # Candle low is above EMA

                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_above(dataframe['rsi'], 85)) |

                (dataframe['low'] < dataframe['ema100'])
            ),
            'sell'] = 1
        return dataframe
