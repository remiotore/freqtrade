 
# --- Do not remove these libs ---
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce

# --------------------------------


class Nemesis4(IStrategy):

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
        "120": 0.02,
        "60":  0.03,
        "30":  0.04,
        "20":  0.05,
        "0":  0.06
    }

    # minimal_roi = {
    #    "0":  100
    # }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -1

    trailing_only_offset_is_reached = True
    trailing_stop = True
    trailing_stop_positive = 0.00301
    trailing_stop_positive_offset = 0.00459

    # Optimal ticker interval for the strategy
    ticker_interval = '5m'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # define macd
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # define point 0 and a
        dataframe['0'] = dataframe['close'].tail(500).min()
        dataframe['a'] = dataframe['close'].tail(500).min()
        dataframe['diff0A'] = dataframe['a'] - dataframe['0']

        # define gkl
        dataframe['500gkl'] = dataframe['diff0A'] * 0.5
        dataframe['559gkl'] = dataframe['diff0A'] * 0.441
        dataframe['618gkl'] = dataframe['diff0A'] * 0.382
        dataframe['667gkl'] = dataframe['diff0A'] * 0.233

        # define bc and zl
        dataframe['b'] = 0
        dataframe['c'] = 0
        dataframe['zl1618'] = 0
        dataframe['zl1809'] = 0
        dataframe['zl2'] = 0
        dataframe['diffBC'] = 0

        # define bc
        dataframe['500bc'] = 0
        dataframe['559bc'] = 0
        dataframe['618bc'] = 0
        dataframe['667bc'] = 0
        dataframe['sequenceActivated'] = False

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        update point a, c and gkl bc when sequence is finished or destroyed
        check if sequence is activated, if y set the variable sequenceActivated = True
        check if there is a valid point b, if y set b if not set b 0, if b still in range maybe update it.
        """
        # update 0
        if((dataframe['close'].tail(500).min() < dataframe['close']).any()):
            dataframe['0'] = dataframe['close'].tail(500).min()
            # update gkl when low/0 changes
            self.update_gkl(dataframe)

        # update a
        if((dataframe['close'].tail(500).max() < dataframe['close']).any() and (dataframe['b'] == 0).any()):
            dataframe['a'] = dataframe['close'].tail(500).max()
            # update gkl when high/a changes
            self.update_gkl(dataframe)

        # set b in gkl
        if((dataframe['close'] < dataframe['500gkl']).any() & (dataframe['close'] > dataframe['667gkl']).any() and dataframe['b'] == 0):
            dataframe['b'] = dataframe['close']

        # update b if a new low inside gkl is generated
        if((dataframe['b'] > 0).any() & (dataframe['close'] < dataframe['b']).any()):
            dataframe['b'] = dataframe['close']

        # create bc and activate sequence
        if((dataframe['b'] > 0).any() & (dataframe['close'] > dataframe['a']).any()):
            dataframe['sequenceActivated'] = True
            dataframe['c'] = dataframe['close']
            dataframe['zl1618'] = (dataframe['a'] - dataframe['0']) * 1.618 + dataframe['b']
            dataframe['zl1809'] = (dataframe['a'] - dataframe['0']) * 1.809 + dataframe['b']
            dataframe['zl2'] = (dataframe['a'] - dataframe['0']) * 2 + dataframe['b']
            self.update_bc(dataframe)

        # destroy sequence
        if ((dataframe['close'] < dataframe['b']).any() & (dataframe['667gkl']).any()):
            dataframe['sequenceActivated'] = False
            dataframe['b'] = dataframe['close']

        dataframe.loc[
            (
                (dataframe['sequenceActivated']) &
                (dataframe['close'] <= dataframe['500bc'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                dataframe['close'] >= dataframe['c']
            ),
            'sell'] = 1

        return dataframe

    # update gkl
    def update_gkl(self, dataframe):
        dataframe['0'] = dataframe['close'].tail(500).min()
        dataframe['a'] = dataframe['close'].tail(500).min()
        dataframe['diff0A'] = dataframe['a'] - dataframe['0']
        dataframe['500gkl'] = dataframe['diff0A'] * 0.5
        dataframe['559gkl'] = dataframe['diff0A'] * 0.441
        dataframe['618gkl'] = dataframe['diff0A'] * 0.382
        dataframe['667gkl'] = dataframe['diff0A'] * 0.233

    # update bc
    def update_bc(self, dataframe):
        dataframe['diffBC'] = dataframe['c'] - dataframe['b']
        dataframe['500bc'] = dataframe['diffBC'] * 0.5
        dataframe['559bc'] = dataframe['diffBC'] * 0.441
        dataframe['618bc'] = dataframe['diffBC'] * 0.382
        dataframe['667bc'] = dataframe['diffBC'] * 0.233
