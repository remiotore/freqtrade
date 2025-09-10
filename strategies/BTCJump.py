
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair

import Config



class BTCJump(IStrategy):
    """
    Simple strategy that looks for jumps in BTC and buys if the current pair has not yet risen as much
    This version doesn't issue a sell signal, just holds until ROI or stoploss kicks in

    How to use it?
    > python3 ./freqtrade/main.py -s BTCJump
    """


    buy_params = Config.strategyParameters["BTCJump"]











    buy_btc_jump = DecimalParameter(0.005, 0.015, decimals=3, default=0.009, space="buy")

    buy_bb_gain = DecimalParameter(0.01, 0.12, decimals=2, default=0.09, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.12, space="buy")

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

        if not self.dp:

            return dataframe

        inf_tf = '5m'
        btc_dataframe = self.dp.get_pair_dataframe(pair=Config.informative_pair, timeframe=inf_tf)

        dataframe = merge_informative_pair(dataframe, btc_dataframe, self.timeframe, "5m", ffill=True)

        dataframe['btc_gain'] = (dataframe['close_5m'] - dataframe['open_5m']) / dataframe['open_5m']
        dataframe['btc_zgain'] = dataframe['btc_gain'] - self.buy_btc_jump.value

        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe)

        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']

        dataframe['mfi'] = ta.MFI(dataframe)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)


        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)




        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sar'] = ta.SAR(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(dataframe['fisher_rsi'] <= self.buy_fisher.value)
        conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)


        conditions.append(qtpylib.crossed_above(dataframe['btc_zgain'], 0))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """

        dataframe.loc[(dataframe['close'] >= 0), 'sell'] = 0
        return dataframe