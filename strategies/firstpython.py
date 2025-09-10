
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class firstpython(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

    https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/BbandRsi.cs

    """



    minimal_roi = {
        "0": 0.11078,
        "33": 0.06816,
        "68": 0.02844,
        "165": 0
    }

    trailing_stop = True
    trailing_stop_positive = 0.0102
    trailing_stop_positive_offset = 0.03701
    trailing_only_offset_is_reached = False

    stoploss = -0.32745

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators.
        Can be a copy of the corresponding method from the strategy,
        or will be loaded from the strategy.
        Must align to populate_indicators used (either from this File, or from the strategy)
        Only used when --spaces does not include buy
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband']) &


                (dataframe['rsi'] < 40)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators.
        Can be a copy of the corresponding method from the strategy,
        or will be loaded from the strategy.
        Must align to populate_indicators used (either from this File, or from the strategy)
        Only used when --spaces does not include sell
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_upperband']) 





            ),
            'sell'] = 1
        return dataframe
