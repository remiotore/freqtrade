


import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class DoubleBollingerStrategy(IStrategy):
    
    buy_sma20_drop_ratio = DecimalParameter(0.6, 0.8, decimals=1, default=0.8, space="buy")
    sell_sma5_higher_ratio = DecimalParameter(0.9, 1.3, decimals=1, default=1.3, space="sell")


    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 0.579,
        "3653": 0.373,
        "19881": 0.161,
        "41906": 0
    }

    stoploss = -0.048

    trailing_stop = True
    trailing_stop_positive = 0.294
    trailing_stop_positive_offset = 0.385
    trailing_only_offset_is_reached = True

    timeframe = '1d'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
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



        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']        
        dataframe['macdhist'] = macd['macdhist']
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband20'] = bollinger['lower']
        
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=5, stds=2)
        dataframe['bb_lowerband5'] = bollinger['lower']   
        
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
     
        conditions = []

        conditions.append(dataframe['close'] < dataframe['bb_lowerband20'])
        conditions.append(dataframe['close'] > dataframe['bb_lowerband5'])
        conditions.append(dataframe['bb_lowerband20'] > dataframe['bb_lowerband5'])


        conditions.append(dataframe['macdhist'].shift(1) < 0)
        conditions.append(dataframe['macdhist'] > dataframe['macdhist'].shift(1))
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe   
      

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        conditions = []

        conditions.append(dataframe['close'] > dataframe['sma5'])
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe







    