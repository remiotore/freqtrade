from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta

class teststrat_fvg_bpercent(IStrategy):
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        bollinger = ta.BBANDS(dataframe, timeperiod=200)
        dataframe['bb_percent_b'] = (dataframe['close'] - bollinger['lowerband']) / (bollinger['upperband'] - bollinger['lowerband'])

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=50)


        dataframe['fvg'] = dataframe['close'].diff().abs() > (dataframe['close'] * 0.01)  # Example threshold: 1% price jump

        dataframe['vwap'] = ta.VWAP(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &
                (dataframe['bb_percent_b'] < 0) &
                (dataframe['fvg']) &  # Buy in FVG zones
                (dataframe['close'] < dataframe['vwap'])  # Buy in Discount zones
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) &
                (dataframe['bb_percent_b'] > 1) &
                (dataframe['fvg']) &  # Sell in FVG zones
                (dataframe['close'] > dataframe['vwap'])  # Sell in Premium zones
            ),
            'sell'] = 1
        return dataframe
