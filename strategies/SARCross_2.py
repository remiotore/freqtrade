


from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa



from user_data.strategies import Config


class SARCross_2(IStrategy):
    """
    Simple strategy that trades based on SAR crossing TEMA
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    buy_params = {
        "buy_adx": 36.0,
        "buy_adx_enabled": False,
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.1,
        "buy_dm_enabled": False,
        "buy_fisher": -0.29,
        "buy_fisher_enabled": False,
        "buy_mfi": 71.0,
        "buy_mfi_enabled": False,
    }
    buy_mfi = DecimalParameter(10, 100, decimals=0, default=86, space="buy")
    buy_adx = DecimalParameter(1, 99, decimals=0, default=25, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=-0.37, space="buy")

    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.04, space="buy")
    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")

    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_dm_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_hold = CategoricalParameter([True, False], default=False, space="sell")

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
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        dataframe['adx'] = ta.ADX(dataframe)

        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['di_plus'] = ta.PLUS_DI(dataframe)

        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['di_minus'] = ta.MINUS_DI(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']
        dataframe['di_delta'] = dataframe['di_plus'] - dataframe['di_minus']
























        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)





        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']







        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['mfi'] = ta.MFI(dataframe)





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























        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema25'] = ta.EMA(dataframe, timeperiod=25)








        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)




























































        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        conditions = []


        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] >= self.buy_adx.value)

        if self.buy_dm_enabled.value:
            conditions.append(dataframe['dm_delta'] > 0)

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'] > self.buy_mfi.value)

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_fisher.value)

        conditions.append(dataframe['volume'] > 0)

        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)


        conditions.append(qtpylib.crossed_below(dataframe['sar'], dataframe['tema']))

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

            conditions.append(qtpylib.crossed_above(dataframe['sar'], dataframe['tema']))

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
    