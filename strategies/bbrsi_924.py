

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical.indicator_helpers import fishers_inverse
from freqtrade.strategy.interface import IStrategy


class bbrsi_924(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    minimal_roi = {
        "0": 0.07916501011003861,
        "28": 0.035159901969879184,
        "83": 0.016727420189894475,
        "131": 0
    }

    stoploss = -0.4275704128061333

    ticker_interval = '1h'

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['rsi'] = ta.RSI(dataframe)



        bollinger3 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
        dataframe['bb_lowerband3'] = bollinger3['lower']
        dataframe['bb_middleband3'] = bollinger3['mid']
        dataframe['bb_upperband3'] = bollinger3['upper']

        bollinger1 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband1'] = bollinger1['lower']
        dataframe['bb_middleband1'] = bollinger1['mid']
        dataframe['bb_upperband1'] = bollinger1['upper']

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
                (dataframe['rsi'] > 5) &
                (dataframe["close"] < dataframe['bb_lowerband3'])
            ) ,
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
                (dataframe['rsi'] > 88) &
                (dataframe["close"] > dataframe['bb_upperband1'])
            ) ,
            'sell'] = 1
        return dataframe
