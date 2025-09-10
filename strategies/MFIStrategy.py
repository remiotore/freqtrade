

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import numpy
import tabulate

from freqtrade.strategy.interface import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class MFIStrategy(IStrategy):
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


    minimal_roi = {
        "60": 1.00,
        "30": 1.00,
        "0": 1.00
    }


    stoploss = -1

    trailing_stop = False




    timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False


    startup_candle_count: int = 14

    order_types = {
        'buy': 'market',
        'sell': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {

        'main_plot': {
            'sma5' : {'color' : 'blue'},
            'sma34' : {'color' : 'green'}


        },
        'subplots': {

            "MFI": {
                'mfi': {'color': 'red'},
                'rv': {'color': 'black'}
            },
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



        high   = dataframe['high']
        low    = dataframe['low']
        close  = dataframe['close']
        volume = dataframe['volume']
        dataframe['mfi'] = ta.MFI(high, low, close, volume, 14)

        dataframe['signal'] = 0
        dataframe.loc[
            (
                    (dataframe['mfi'] <= 20)
            ),
            'signal'] = 1
        dataframe.loc[
            (
                (dataframe['mfi'] >= 80)
            ),
            'signal'] = -1
        dataframe['signal'] = dataframe['signal'].diff()
        dataframe.loc[
            (
                (dataframe['mfi'] > 20) &
                (dataframe['mfi'] < 80)
            ),
            'signal'] = 0

        dataframe['rv'] = (volume / dataframe['volume'].rolling(14).max()) * 100
































































































        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)

        dataframe['sma34'] = ta.SMA(dataframe, timeperiod=34)
































































        """

        if self.dp:
            if self.dp.runmode in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['signal'] == 1 ) &
                (dataframe['rv'] > 20)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (dataframe['signal'] == -1) &
                (dataframe['rv'] > 80)
            ),
            'sell'] = 1
        return dataframe
