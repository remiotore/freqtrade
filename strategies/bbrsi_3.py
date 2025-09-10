# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy


class BBRSI(IStrategy):

    # Minimal ROI designed for the strategy
    minimal_roi = {   
        "0": 0.25027387240605425,
        "17": 0.06199470854285548,
        "53": 0.010104345522763993,
        "169": 0
    }


    # Optimal stoploss designed for the strategy
    stoploss = -0.08069594551891693

    # Optimal ticker interval for the strategy
    ticker_interval = '1h'

    # Optional order type mapping
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional time in force for orders
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        #dataframe['bb_lowerband1'] = bollinger['lower']
        dataframe['bb_middleband1'] = bollinger['mid']

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=4)
        dataframe['bb_lowerband4'] = bollinger['lower']

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband4']) &
                (dataframe['rsi'] > 13)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['close'] > dataframe['bb_middleband1'])
            ),
            'sell'] = 1
        return dataframe
