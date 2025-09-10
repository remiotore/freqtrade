

import math
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, merge_informative_pair



import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class TenderEnter(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://github.com/freqtrade/freqtrade/blob/develop/docs/bot-optimization.md

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """


    INTERFACE_VERSION = 2

    custom_stops = {}








    minimal_roi = {
        "0": 0.21296,
        "94": 0.13203,
        "190": 0.04443,
        "374": 0
    }









    stoploss = -0.25933







    trailing_stop = True
    trailing_stop_positive = 0.25571
    trailing_stop_positive_offset = 0.35142
    trailing_only_offset_is_reached = True

    timeframe = '15m'
    inf_tf = '15m' #timeframe of second line

    process_only_new_candles = False

    use_sell_signal = False
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 102

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    

    
    plot_config = {

        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {

            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """























































































































































































        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
    
        dataframe.loc[(
            self.compareFields(dataframe, 'close', 1, 1017) &
            self.compareFields(dataframe, 'close', 2, 1017) &
            self.compareFields(dataframe, 'volume', 1, 65) &
            self.compareFields(dataframe, 'volume', 2, 65) &
            (dataframe['volume'] > 0)),'buy'] = 1
        return dataframe

    def compareFields(self, dt, fieldname, shift, ratio=1.034):
        return dt[fieldname].shift(shift)/dt[fieldname] > ratio/1000

















    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (


                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 0
        return dataframe


















