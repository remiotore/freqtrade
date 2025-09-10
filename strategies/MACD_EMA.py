
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class MACD_EMA(IStrategy):
   
    EMA_LONG_TERM = 200



    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }

    stoploss = -0.25

    timeframe = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['ema_{}'.format(self.EMA_LONG_TERM)] = ta.EMA(
            dataframe, timeperiod=self.EMA_LONG_TERM
        )


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']) &
                ((dataframe['close'] > dataframe['ema_{}'.format(self.EMA_LONG_TERM)]))

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                     qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']) &
                (dataframe['close'] < dataframe['ema_{}'.format(self.EMA_LONG_TERM)])

            ),
            'sell'] = 1
        return dataframe
