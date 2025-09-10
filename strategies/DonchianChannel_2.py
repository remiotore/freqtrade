
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from user_data.strategies import Config



class DonchianChannel_2(IStrategy):
    """
    Simple strategy based on Donchian Channel Breakouts

    How to use it?
    > python3 ./freqtrade/main.py -s DonchianChannel
    """

    buy_params = {
        "buy_adx": 2.0,
        "buy_adx_enabled": False,
        "buy_dc_period": 13,
        "buy_dm_enabled": False,
        "buy_ema_enabled": False,
        "buy_fisher": 0.06,
        "buy_fisher_enabled": True,
        "buy_macd_enabled": True,
        "buy_mfi": 5.0,
        "buy_mfi_enabled": True,
        "buy_sar_enabled": False,
        "buy_sma_enabled": True,
    }

    buy_dc_period = IntParameter(1, 50, default=27, space="buy")
    buy_adx = DecimalParameter(1, 99, decimals=0, default=30, space="buy")
    buy_mfi = DecimalParameter(1, 99, decimals=0, default=50, space="buy")
    buy_fisher = DecimalParameter(-1.0, 1.0, decimals=2, default=0.81, space="buy")

    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_dm_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    sell_adx_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_dm_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_sma_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_ema_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_sar_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_macd_enabled = CategoricalParameter([True, False], default=False, space="sell")

    startup_candle_count = max(buy_dc_period.value, 20)

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

        bollinger = qtpylib.weighted_bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_diff"] = (dataframe["bb_upperband"] - dataframe["close"])

        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_mid'] = ((dataframe['dc_upper'] + dataframe['dc_lower']) / 2)
        dataframe['dc_diff'] = (dataframe['dc_upper'] - dataframe['close'])

        dataframe['dc_dist'] = (dataframe['dc_upper']  - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib




        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)
        dataframe['dm_delta'] = dataframe['dm_plus'] - dataframe['dm_minus']

        dataframe['mfi'] = ta.MFI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)


        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []






        if self.buy_sar_enabled.value:
            conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['sma200'].notnull())
            conditions.append(dataframe['close'] > dataframe['sma200'])

        if self.buy_ema_enabled.value:
            conditions.append(dataframe['ema50'].notnull())
            conditions.append(dataframe['close'] > dataframe['ema50'])

        if self.buy_mfi_enabled.value:
            conditions.append(dataframe['mfi'].notnull())
            conditions.append(dataframe['mfi'] >= self.buy_mfi.value)

        if self.buy_adx_enabled.value:
            conditions.append(dataframe['adx'] >= self.buy_adx.value)

        if self.buy_dm_enabled.value:
            conditions.append(dataframe['dm_delta'] > 0)

        if self.buy_macd_enabled.value:
            conditions.append(dataframe['macd'] > dataframe['macdsignal'])

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi']  < self.buy_fisher.value)



        level = 'dc_upper'

        conditions.append(
            (dataframe[level].notnull()) &
            (dataframe['close'] >= dataframe[level])




















        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """

        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0
            return dataframe

        conditions = []

        level = 'dc_lower'

        conditions.append(
            (dataframe['dc_lf'].notnull()) &
            (
                    (
                            (dataframe['close'] < dataframe['open']) &
                            (qtpylib.crossed_below(dataframe['close'], dataframe[level]))
                    ) |
                    (
                            (dataframe['close'] <= dataframe[level]) &
                            (dataframe['close'].shift(1) > dataframe[level].shift(1))
                    )
            )
        )


        orconditions = []

        if self.sell_sar_enabled.value:

            orconditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['sar']))

        if self.sell_sma_enabled.value:

            orconditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['sma200']))

        if self.sell_ema_enabled.value:

            orconditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['ema50']))

        if self.sell_adx_enabled.value:
            conditions.append(dataframe['adx'] < self.buy_adx.value)

        if self.sell_dm_enabled.value:
            conditions.append(dataframe['dm_delta'] < 0)

        if self.sell_macd_enabled.value:
            orconditions.append(qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))

        r1 = False
        r2 = False
        if conditions:
             r1 = reduce(lambda x, y: x & y, conditions)

        if orconditions:
            r2 = reduce(lambda x, y: x | y, orconditions)

        dataframe.loc[(r1 | r2), 'sell'] = 1

        if orconditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, orconditions),
                'sell'] = 1

        return dataframe