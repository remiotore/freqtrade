
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from user_data.strategies import Config


class Hammer(IStrategy):
    """
    Trades based on detection of Hammer-like candlestick patterns

    How to use it?
    > python3 ./freqtrade/main.py -s Hammer
    """

    buy_params = {
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.08,
        "buy_ema_enabled": True,
        "buy_mfi": 47.0,
        "buy_mfi_enabled": False,
        "buy_sma_enabled": True,
    }

    pattern_strength = 90
    buy_mfi = DecimalParameter(0, 50, decimals=0, default=47, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.08, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_hold_enabled = CategoricalParameter([True, False], default=True, space="sell")

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

        dataframe['mfi'] = ta.MFI(dataframe)

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)


        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)



        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)



        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)











        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)

        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)

























        dataframe['height'] = abs(dataframe['close']-dataframe['open'])
        dataframe['body'] = dataframe['height'].clip(lower=0.01)
        dataframe['top'] = dataframe[['close','open']].max(axis=1)
        dataframe['bottom'] = dataframe[['close','open']].min(axis=1)
        dataframe['upper_shadow'] = dataframe['high']-dataframe['top']
        dataframe['lower_shadow'] = dataframe['bottom']-dataframe['low']
        dataframe['upper_ratio'] = (dataframe['high']-dataframe['top'])/dataframe['body']
        dataframe['upper_ratio'] = dataframe['upper_ratio'].clip(upper=10)
        dataframe['lower_ratio'] = (dataframe['bottom']-dataframe['low'])/dataframe['body']
        dataframe['lower_ratio'] = dataframe['lower_ratio'].clip(upper=10)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        conditions = []


        conditions.append(dataframe['volume'] > 0)

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['close'] < dataframe['sma'])

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        if self.buy_ema_enabled.value:
            conditions.append(dataframe['close'] <= dataframe['ema10'])



        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)



        conditions.append(dataframe['body'] > 0.01)

        conditions.append(
            (dataframe['upper_ratio'] > 2) |
            (dataframe['lower_ratio'] > 2)
        )








        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        conditions = []

        if self.sell_hold_enabled.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        else:


            conditions.append(dataframe['volume'] > 0)











            conditions.append(dataframe['close'] >= dataframe['ema10'])

            conditions.append(dataframe['body'] > 0.01)

            conditions.append(
                (dataframe['upper_ratio'] > 2)


            )








            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe