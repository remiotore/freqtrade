
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class adx_bb_rsi2(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

        https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AdxMomentum.cs

    """



    minimal_roi = {
        "0": 0.16083,
        "33": 0.04139,
        "85": 0.01225,
        "197": 0

    }

    stoploss = -0.32237

    timeframe = '1h'

    startup_candle_count: int = 20

    trailing_stop = True
    trailing_stop_positive = 0.1195
    trailing_stop_positive_offset = 0.1568
    trailing_only_offset_is_reached = True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']




        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['adx'] > 47) &
                (dataframe['fastd'] < 41)&
                (dataframe['close'] < dataframe['bb_lowerband'])




            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['adx'] > 67) &
                (dataframe['mfi'] > 97) &

                (dataframe['fastd'] < 74)







            ),
            'sell'] = 1
        return dataframe
