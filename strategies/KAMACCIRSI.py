

from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class KAMACCIRSI(IStrategy):
    """
    author@: werkkrew
    github@: https://github.com/werkkrew/freqtrade-strategies

    Strategy using 3 indicators with fully customizable parameters and full hyperopt support
    including indicator periods as well as cross points.

    There is nothing groundbreaking about this strategy, how it works, or what it does. 
    It was mostly an experiment for me to learn Freqtrade strategies and hyperopt development.

    Default hyperopt defined parameters below were done on 60 days of data from Kraken against 20 BTC pairs
    using the SharpeHyperOptLoss loss function.

    Suggestions and improvements are welcome!

    Supports selling via strategy, as well as ROI and Stoploss/Trailing Stoploss

    Indicators Used:
    KAMA "Kaufman Adaptive Moving Average" (Short Duration)
    KAMA (Long Duration)
    CCI "Commodity Channel Index"
    RSI "Relative Strength Index"

    Buy Strategy:
        kama-cross OR kama-slope 
            kama-short > kama-long
            kama-long-slope > 1
        cci-enabled? 
            cci > X 
        rsi-enabled?
            rsi > Y 

    Sell Strategy:
        kama-cross OR kama-slope 
            kama-short < kama-long
            kama-long-slope < 1
        cci-enabled?
            cci < A
        rsi-enabled?
            rsi < B

    Ideas and Todo:
        - Add informative pairs to help decision (e.g. BTC/USD to inform other */BTC pairs)
    """


    INTERFACE_VERSION = 2

    """
    HYPEROPT SETTINGS
    The following is set by Hyperopt, or can be set by hand if you wish:

    - minimal_roi table
    - stoploss
    - trailing stoploss
    - for buy/sell separate
        - kama-trigger = cross, slope
        - kama-short timeperiod
        - kama-long timeperiod
        - cci period
        - cci upper / lower threshold
        - rsi period
        - rsi upper / lower threshold

    PASTE OUTPUT FROM HYPEROPT HERE
    """

    buy_params = {
        'cci-enabled': True,
        'cci-limit': 198,
        'cci-period': 18,
        'kama-long-period': 46,
        'kama-short-period': 11,
        'kama-trigger': 'cross',
        'rsi-enabled': False,
        'rsi-limit': 72,
        'rsi-period': 5
    }

    sell_params = {
        'sell-cci-enabled': False,
        'sell-cci-limit': -144,
        'sell-cci-period': 18,
        'sell-kama-long-period': 41,
        'sell-kama-short-period': 5,
        'sell-kama-trigger': 'cross',
        'sell-rsi-enabled': False,
        'sell-rsi-limit': 69,
        'sell-rsi-period': 12
    }

    minimal_roi = {
        "0": 0.11599,
        "18": 0.03112,
        "34": 0.01895,
        "131": 0
    }

    stoploss = -0.32982

    trailing_stop = True
    trailing_stop_positive = 0.28596
    trailing_stop_positive_offset = 0.29771
    trailing_only_offset_is_reached = True

    """
    END HYPEROPT
    """

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = False

    process_only_new_candles = False


    startup_candle_count: int = 72
    
    """
    Not currently being used for anything, thinking about implementing this later.
    """
    def informative_pairs(self):

        informative_pairs = [(f"{self.config['stake_currency']}/USD", self.timeframe)]
        return informative_pairs

    """
    Populate all of the indicators we need (note: indicators are separate for buy/sell)
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['buy-cci'] = ta.CCI(dataframe, timeperiod=self.buy_params['cci-period'])
        dataframe['sell-cci'] = ta.CCI(dataframe, timeperiod=self.sell_params['sell-cci-period'])

        dataframe['buy-rsi'] = ta.RSI(dataframe, timeperiod=self.buy_params['rsi-period'])
        dataframe['sell-rsi'] = ta.RSI(dataframe, timeperiod=self.sell_params['sell-rsi-period'])

        dataframe['buy-kama-short'] = ta.KAMA(dataframe, timeperiod=self.buy_params['kama-short-period'])
        dataframe['buy-kama-long'] = ta.KAMA(dataframe, timeperiod=self.buy_params['kama-long-period'])
        dataframe['buy-kama-long-slope'] = (dataframe['buy-kama-long'] / dataframe['buy-kama-long'].shift())

        dataframe['sell-kama-short'] = ta.KAMA(dataframe, timeperiod=self.sell_params['sell-kama-short-period'])
        dataframe['sell-kama-long'] = ta.KAMA(dataframe, timeperiod=self.sell_params['sell-kama-long-period'])
        dataframe['sell-kama-long-slope'] = (dataframe['sell-kama-long'] / dataframe['sell-kama-long'].shift())

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        if self.buy_params['rsi-enabled']:
            conditions.append(dataframe['buy-rsi'] > self.buy_params['rsi-limit'])
        if self.buy_params['cci-enabled']:
            conditions.append(dataframe['buy-cci'] > self.buy_params['cci-limit'])
        if self.buy_params['kama-trigger'] == 'cross':
            conditions.append(dataframe['buy-kama-short'] > dataframe['buy-kama-long'])
        if self.buy_params['kama-trigger'] == 'slope':
            conditions.append(dataframe['buy-kama-long'] > 1)

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        if self.sell_params['sell-rsi-enabled']:
            conditions.append(dataframe['sell-rsi'] < self.sell_params['sell-rsi-limit'])
        if self.sell_params['sell-cci-enabled']:
            conditions.append(dataframe['sell-cci'] < self.sell_params['sell-cci-limit'])
        if self.sell_params['sell-kama-trigger'] == 'cross':
            conditions.append(dataframe['sell-kama-short'] < dataframe['sell-kama-long'])
        if self.sell_params['sell-kama-trigger'] == 'slope':
            conditions.append(dataframe['sell-kama-long'] < 1)

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe
    