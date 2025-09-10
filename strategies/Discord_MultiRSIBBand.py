# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
# --------------------------------
import talib.abstract as ta
from technical.util import resample_to_interval, resampled_merge


class MultiRSIBBand(IStrategy):
    """

    author@: Gert Wohlgemuth

    based on work from Creslin

    """
    minimal_roi = {
    "180": 0.0,    # Sell after 3h minutes if the profit is not negative
    "30": 0.04,   # Sell after 30 minutes if there is at least 1% profit
    "20": 0.04,   # Sell after 20 minutes if there is at least 2% profit
    "0":  0.05    # Sell immediately if there is at least 5% profit
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.20
    trailing_stop = True
    trailing_stop_positive = 0.006
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # resample our dataframes
        dataframe_short = resample_to_interval(dataframe, self.get_ticker_indicator() * 1)
        dataframe_long = resample_to_interval(dataframe, self.get_ticker_indicator() * 12)

        # compute our RSI's
        dataframe_short['rsi'] = ta.RSI(dataframe_short, timeperiod=14)
        dataframe_long['rsi'] = ta.RSI(dataframe_long, timeperiod=14)

        # merge dataframe back together
        dataframe = resampled_merge(dataframe, dataframe_short)
        dataframe = resampled_merge(dataframe, dataframe_long)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe.fillna(method='ffill', inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

            (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 12)] > (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 12)].shift(2))) &
            
            (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 1)] > 50) &
            (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 1)].shift(1) <= 50)
            
             
#             #RSI 5 min increased from below 35 to above 35
#            (dataframe['rsi'] > 35) &
#
#            (dataframe['rsi'].shift(1) <= 35)






            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (

            ),
            'sell'] = 1
        return dataframe
