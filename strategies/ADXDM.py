
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from user_data.strategies import Config



class ADXDM(IStrategy):
    """
    Simple strategy based on ADX value and DM+/DM- crossing

    How to use it?
    > python3 ./freqtrade/main.py -s ADXDM
    """


    buy_params = {
        "buy_adx": 60.0,
        "buy_bb_enabled": False,
        "buy_bb_gain": 0.02,
        "buy_mfi": 6.0,
        "buy_mfi_enabled": True,
        "buy_period": 12,
    }

    buy_adx = DecimalParameter(20, 60, decimals=0, default=60, space="buy")
    buy_mfi = DecimalParameter(1, 30, decimals=0, default=6, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_period = IntParameter(3, 50, default=12, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.02, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    startup_candle_count = 20

    minimal_roi = Config.minimal_roi
    trailing_stop = Config.trailing_stop
    trailing_stop_positive = Config.trailing_stop_positive
    trailing_stop_positive_offset = Config.trailing_stop_positive_offset
    trailing_only_offset_is_reached = Config.trailing_only_offset_is_reached
    stoploss = Config.stoploss
    timeframe = Config.timeframe
    process_only_new_candles = Config.process_only_new_candles
    use_sell_signal = Config.use_sell_signal
    sell_profit_only = Config.sell_profit_only
    ignore_roi_if_buy_signal = Config.ignore_roi_if_buy_signal
    order_types = Config.order_types

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
        """

        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['adx_delta'] = (dataframe['adx'] - self.buy_adx.value) / 100 # for display
        dataframe['adx_slope'] = ta.LINEARREG_SLOPE(dataframe['adx'], timeperiod=3)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)

        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])



        dataframe['sma'] = ta.SMA(dataframe, timeperiod=20)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=20)

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []




        conditions.append(dataframe['adx'].notnull())

        if self.buy_mfi_enabled.value:

            conditions.append(dataframe['mfi'] <= dataframe['adx'])




        conditions.append(dataframe['adx'] > self.buy_adx.value)

        conditions.append(qtpylib.crossed_below(dataframe['adx_slope'], 0))

        conditions.append(dataframe['dm_delta'] < 0)






        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """

        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull()), 'sell'] = 0
            return dataframe

        conditions = []

        conditions.append(dataframe['adx'] > self.buy_adx.value)

        conditions.append(qtpylib.crossed_below(dataframe['adx_slope'], 0))

        conditions.append(dataframe['dm_delta'] > 0)

        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)








        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe