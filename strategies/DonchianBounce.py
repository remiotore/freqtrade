
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from user_data.strategies import Config



class DonchianBounce(IStrategy):
    """
    Simple strategy based on Contrarian Donchian Channel Bounce from the bottom band

    How to use it?
    > python3 ./freqtrade/main.py -s DonchianBounce.py
    """


    buy_params = {
        "buy_adx": 43.0,
        "buy_adx_enabled": False,
        "buy_dc_gain": 0.05,
        "buy_dc_period": 60,
        "buy_ema_enabled": False,
        "buy_sar_enabled": False,
        "buy_sma_enabled": True,
    }

    buy_dc_period = IntParameter(10, 120, default=43, space="buy")
    buy_dc_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.05, space="buy")

    buy_adx = DecimalParameter(1, 99, decimals=0, default=60, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_sar_enabled = CategoricalParameter([True, False], default=False, space="sell")
    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

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

        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_dc_period.value)
        dataframe['dc_mid'] = ((dataframe['dc_upper'] + dataframe['dc_lower']) / 2)
        dataframe["dc_gain"] = ((dataframe["dc_upper"] - dataframe["close"]) / dataframe["close"])

        dataframe['dc_dist'] = (dataframe['dc_upper']  - dataframe['dc_lower'])
        dataframe['dc_hf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.236 # Highest Fib
        dataframe['dc_chf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.382 # Centre High Fib
        dataframe['dc_clf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.618 # Centre Low Fib
        dataframe['dc_lf'] = dataframe['dc_upper'] - dataframe['dc_dist'] * 0.764 # Low Fib




        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)

        dataframe['mfi'] = ta.MFI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']

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

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=200)


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []




        conditions.append(dataframe['dc_hf'].notnull())

        if self.buy_sar_enabled.value:
            conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['sma'].notnull())
            conditions.append(dataframe['close'] > dataframe['sma'])

        if self.buy_ema_enabled.value:
            conditions.append(dataframe['ema50'].notnull())
            conditions.append(dataframe['close'] > dataframe['ema50'])

        if self.buy_adx_enabled.value:
            conditions.append(
                (dataframe['adx'] > self.buy_adx.value)


            )

        conditions.append(dataframe['dc_gain'] >= self.buy_dc_gain.value)




        conditions.append(
            (dataframe['dc_lower'].notnull()) &





            (dataframe['close'] >= dataframe['open']) &
            (
                (qtpylib.crossed_above(dataframe['close'], dataframe['dc_lower'])) |
                (
                        (dataframe['close'] >= dataframe['dc_lower']) &
                        (dataframe['close'].shift(1) < dataframe['dc_lower'].shift(1))
                )
            )

        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        """

        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        else:

            conditions = []

            conditions.append(
                (dataframe['dc_upper'].notnull()) &
                (dataframe['close'] < dataframe['open']) &
                (
                        (qtpylib.crossed_below(dataframe['close'], dataframe['dc_upper'])) |
                        (
                                (dataframe['close'] <= dataframe['dc_upper']) &
                                (dataframe['close'].shift(1) > dataframe['dc_upper'].shift(1))
                        )
                )
            )

            if self.sell_sar_enabled.value:
                conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
