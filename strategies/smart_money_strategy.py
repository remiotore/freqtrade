import numpy
import talib.abstract as ta
from pandas import DataFrame
from technical.indicators import chaikin_money_flow
from freqtrade.strategy import (DecimalParameter, IStrategy, IntParameter)











class smart_money_strategy(IStrategy):

    minimal_roi = {
        "0": 10
    }

    stoploss = -99

    timeframe = '1h'
    sell_profit_only = True
    sell_profit_offset = 0.01

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['cmf'] = chaikin_money_flow(dataframe, period=20)

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['ema_200']) &
                    (dataframe['mfi'] < 35) &
                    (dataframe['cmf'] < -0.07)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['ema_200']) &
                    (dataframe['mfi'] > 70) &
                    (dataframe['cmf'] > 0.20)
            ),
            'sell'] = 1

        return dataframe

class smart_money_strategy_hyperopt(IStrategy):

    minimal_roi = {
        "0": 10
    }

    stoploss = -99

    timeframe = '1h'
    sell_profit_only = True
    sell_profit_offset = 0.01

    buy_mfi = IntParameter(20, 60, default=35, space="buy")
    buy_cmf = DecimalParameter(-0.4, -0.01, decimals=2, default=-0.07, space="buy")

    sell_mfi = IntParameter(50, 95, default=70, space="sell")
    sell_cmf = DecimalParameter(0.1, 0.6, decimals=2, default=0.2, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['cmf'] = chaikin_money_flow(dataframe, period=20)

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['ema_200']) &
                    (dataframe['mfi'] < self.buy_mfi.value) &
                    (dataframe['cmf'] < self.buy_cmf.value)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['ema_200']) &
                    (dataframe['mfi'] > self.sell_mfi.value) &
                    (dataframe['cmf'] > self.sell_cmf.value)
            ),
            'sell'] = 1
        return dataframe
