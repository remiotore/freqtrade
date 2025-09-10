
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib



class ADXMomentum_3(IStrategy):
    """

    author@: Gert Wohlgemuth

    converted from:

        https://github.com/sthewissen/Mynt/blob/master/src/Mynt.Core/Strategies/AdxMomentum.cs

    """



    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.25

    timeframe = '1h'

    startup_candle_count: int = 20


    buy_adx = IntParameter(15, 35, default=25, space="buy")
    sell_adx = IntParameter(15, 35, default=25, space="sell")
    adx_timeperiod = IntParameter(7, 21, default=14, space="buy")

    buy_plus_di = IntParameter(15, 35, default=25, space="buy")
    sell_minus_di = IntParameter(15, 35, default=25, space="sell")
    plus_di_timeperiod = IntParameter(20, 35, default=25, space="buy")
    minus_di_timeperiod = IntParameter(20, 35, default=25, space="buy")

    buy_mom = IntParameter(0, 20, default=0, space="buy")
    sell_mom = IntParameter(-20, 0, default=0, space="sell")
    mom_timeperiod = IntParameter(15, 30, default=25, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=self.adx_timeperiod.value)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.plus_di_timeperiod.value)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.minus_di_timeperiod.value)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['mom'] = ta.MOM(dataframe, timeperiod=self.mom_timeperiod.value)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > self.buy_adx.value) &
                    (dataframe['mom'] > self.buy_mom.value) &
                    (dataframe['plus_di'] > self.buy_plus_di.value) &
                    (dataframe['plus_di'] > dataframe['minus_di'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['adx'] > self.sell_adx.value) &
                    (dataframe['mom'] < self.sell_mom.value) &
                    (dataframe['minus_di'] > self.sell_minus_di.value) &
                    (dataframe['plus_di'] < dataframe['minus_di'])

            ),
            'sell'] = 1
        return dataframe
