from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class EMAcrossover(IStrategy):

    """
    EMAPriceCrossoverWithTreshold
    """









    stoploss = -0.15

    timeframe = '1h'

    trailing_stop = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        threshold_percentage = 1
        dataframe['ema800'] = ta.EMA(dataframe, timeperiod=800)
        dataframe['ema_threshold'] = dataframe['ema800'] * (100 - threshold_percentage) / 100

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_above(dataframe['close'], dataframe['ema800'])) &

                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (qtpylib.crossed_below(dataframe['close'], dataframe['ema_threshold']))
            ),
            'sell'] = 1
        return dataframe