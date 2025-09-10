
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib



class BBRSI_815(IStrategy):



    minimal_roi = {
        "0": 0.17139,
        "10": 0.07792,
        "66": 0.03513,
        "130": 0
    }

    stoploss = -0.25


    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > 30) &
                    (dataframe['close'] < dataframe['bb_lowerband'])

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['bb_middleband'])

            ),
            'sell'] = 1
        return dataframe
