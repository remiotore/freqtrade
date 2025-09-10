


from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
import math

from user_data.strategies import Config




class TEMABounce(IStrategy):
    """
    Simple strategy that looks for prices falling a sepcificed distance below a long-term TEMA

    """

    buy_params = {
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.1,
        "buy_diff": 0.094,
        "buy_long_period": 80,
        "buy_macd_enabled": False,
        "buy_short_period": 13,
    }

    buy_long_period = IntParameter(20, 100, default=50, space="buy")
    buy_short_period = IntParameter(5, 15, default=10, space="buy")

    buy_diff = DecimalParameter(0.01, 0.10, decimals=3, default=0.065, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=False, space="buy")

    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.04, space="buy")
    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_diff = DecimalParameter(0.01, 0.10, decimals=3, default=0.057, space="sell")
    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    startup_candle_count = max(buy_long_period.value, 20)

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
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """




































        dataframe['rsi'] = ta.RSI(dataframe)




















        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']







        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
        )
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])
























        
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=self.buy_long_period.value)
        dataframe['tema_short'] = ta.TEMA(dataframe, timeperiod=self.buy_short_period.value)
        dataframe['tema_angle'] = ta.LINEARREG_SLOPE(dataframe['tema_short'], timeperiod=3) / (2.0 * math.pi)
        dataframe['tema_diff'] = (((dataframe['tema'] - dataframe['close']) /
                                      dataframe['close'])) \
                                    - self.buy_diff.value

        dataframe['tema_angle'] = ta.LINEARREG_SLOPE(dataframe['tema_short'], timeperiod=3) / (2.0 * math.pi)



        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.buy_long_period.value)







        dataframe['sar'] = ta.SAR(dataframe)






























































        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        conditions = []


        conditions.append(dataframe['volume'] > 0)

        if self.buy_macd_enabled.value:
            conditions.append(dataframe['macdhist'] >= 0)

        conditions.append(dataframe['tema_diff'] > 0.0)

        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)


        conditions.append(qtpylib.crossed_above(dataframe['tema_angle'], 0))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        conditions = []

        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        else:

            conditions.append(qtpylib.crossed_below(dataframe['tema_angle'], 0))

            conditions.append(dataframe['close'] > dataframe['tema'])

            conditions.append(dataframe['tema_diff'] >= -self.sell_diff.value)

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
    