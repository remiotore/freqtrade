# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
from technical.util import resample_to_interval, resampled_merge


class MultiMACD(IStrategy):
    """

    author@: Gert Wohlgemuth

    based on work from Creslin

    """
    minimal_roi = {
        "0": 0.01
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.05

    # Optimal timeframe for the strategy
    timeframe = '4h'

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:


        # resample our dataframes
        dataframe_short = resample_to_interval(dataframe, self.get_ticker_indicator() * 2)
        dataframe_long = resample_to_interval(dataframe, self.get_ticker_indicator() * 8)

        # compute our MACD's
        dataframe_short['macd'] = ta.macd(dataframe_short, fastperiod=12, slowperiod=26, signalperiod=7)
        dataframe_long['macd'] = ta.macd(dataframe_long, fastperiod=12, slowperiod=26, signalperiod=7

        # merge dataframe back together
        dataframe = resampled_merge(dataframe, dataframe_short)
        dataframe = resampled_merge(dataframe, dataframe_long)


        dataframe_short['macd'] = macd['macd']
        dataframe_short['macdsignal'] = macd['macdsignal']
        
        dataframe_long['macd'] = macd['macd']
        dataframe_long['macdsignal'] = macd['macdsignal']
        dataframe.fillna(method='ffill', inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                 dataframe_long['macd']  > dataframe_long['macdsignal']
 
            
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
               #(dataframe['rsi'] > dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*2)]) &
                #(dataframe['rsi'] > dataframe['resample_{}_rsi'.format(self.get_ticker_indicator()*8)])
            ),
            'sell'] = 1
        return dataframe
