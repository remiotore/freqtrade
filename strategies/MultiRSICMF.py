
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame

import talib.abstract as ta


class MultiRSICMF(IStrategy):
    """

    author@: Gert Wohlgemuth

    based on work from Creslin

    """
    minimal_roi = {
        "0": 0.01
    }

    stoploss = -0.05

    ticker_interval = '5m'

    def get_ticker_indicator(self):
        return int(self.ticker_interval[:-1])

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:



        from technical.util import resample_to_interval
        from technical.util import resampled_merge
        from technical.indicators import cmf
        from technical.indicators import osc

        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe_short = resample_to_interval(dataframe, self.get_ticker_indicator() * 2)
        dataframe_long = resample_to_interval(dataframe, self.get_ticker_indicator() * 8)

        dataframe_short['rsi'] = ta.RSI(dataframe_short, timeperiod=14)
        dataframe_long['rsi'] = ta.RSI(dataframe_long, timeperiod=14)
        dataframe_short['cmf'] = cmf(dataframe_short, 14)
        dataframe_long['cmf'] = cmf(dataframe_long, 14)

        dataframe_short['osc'] = osc(dataframe_short, 14)
        dataframe_long['osc'] = osc(dataframe_long, 14)

        dataframe = resampled_merge(dataframe, dataframe_short)
        dataframe = resampled_merge(dataframe, dataframe_long)

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe.fillna(method='ffill', inplace=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (

                    (dataframe['sma5'] >= dataframe['sma200']) &
                    (dataframe['rsi'] < (dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 8)] - 20)) &
                    (dataframe['resample_{}_cmf'.format(self.get_ticker_indicator() * 8)] > 0)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        dataframe.loc[
            (
                    (dataframe['rsi'] > dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 2)]) &
                    (dataframe['rsi'] > dataframe['resample_{}_rsi'.format(self.get_ticker_indicator() * 8)])
            ),
            'sell'] = 1
        return dataframe
