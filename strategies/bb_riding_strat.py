
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class bb_riding_strat(IStrategy):



    minimal_roi = {
        "0": 0.3 #0.26   
    }






    stoploss = -0.235

    timeframe = '30m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        boll = ta.BBANDS(dataframe, nbdevup=2.0, nbdevdn=2.0, timeperiod=20)  #set timeperiod to your time period
        dataframe['bb_lower'] = boll['lowerband']
        dataframe['bb_middle'] = boll['middleband']
        dataframe['bb_upper'] = boll['upperband']




        dataframe["bb_width"] = (
            (dataframe["bb_upper"] - dataframe["bb_lower"]) / dataframe["bb_middle"]
        )
        print(metadata)
        print(dataframe[["date","close","bb_upper","bb_middle","bb_lower","bb_width"]].tail(25))

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['close'] > dataframe['bb_upper']) 

            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

            ),
            'sell'] = 1
        return dataframe





















































