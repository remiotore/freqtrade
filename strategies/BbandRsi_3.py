
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class BbandRsi_3(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs

    """



    minimal_roi = {
        "0": 0.1
    }

    stoploss = -0.25

    ticker_interval = '1h'

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] < 30) &
                    (dataframe['close'] < dataframe['bb_lowerband'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 70)

            ),
            'sell'] = 0
        return dataframe
