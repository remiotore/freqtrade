
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class TuplaBollinger_345(IStrategy):
   
    EMA_LONG_TERM = 200



    minimal_roi = {
        "0": 0.9,
        "1": 0.05,
        "10": 0.04,
        "15": 0.5
    }

    stoploss = -0.25

    timeframe = '5h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        bollinger_inner = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['inner_lowerband'] = bollinger_inner['lower']
        dataframe['bb_middleband'] = bollinger_inner['mid']
        dataframe['inner_upperband'] = bollinger_inner['upper']

        bollinger_outer = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['outer_lowerband'] = bollinger_outer['lower']

        dataframe['outer_upperband'] = bollinger_outer['upper']

        dataframe['ema_{}'.format(self.EMA_LONG_TERM)] = ta.EMA(
            dataframe, timeperiod=self.EMA_LONG_TERM
        )


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] < dataframe['inner_lowerband']) &
                    (dataframe['close'].shift(1) < dataframe['close'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['inner_upperband']) &
                    (dataframe['close'].shift(1) > dataframe['close'])

            ),
            'sell'] = 1
        return dataframe
