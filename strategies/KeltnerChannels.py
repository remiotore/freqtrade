
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from user_data.strategies import Config


class KeltnerChannels(IStrategy):
    """
    Simple strategy based on Keltner Channel Breakout

    How to use it?
    > python3 ./freqtrade/main.py -s KeltnerChannels
    """

    buy_params = {
        "buy_adx": 47.0,
        "buy_adx_enabled": True,
        "buy_dm_enabled": True,
        "buy_ema_enabled": False,
        "buy_fisher": 0.95,
        "buy_fisher_enabled": False,
        "buy_macd_enabled": False,
        "buy_mfi": 44.0,
        "buy_mfi_enabled": True,
        "buy_sar_enabled": True,
        "buy_sma_enabled": False,
    }

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

    startup_candle_count = 50

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

        dataframe['mfi'] = ta.MFI(dataframe)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upperband"] = keltner["upper"]
        dataframe["kc_lowerband"] = keltner["lower"]
        dataframe["kc_middleband"] = keltner["mid"]
        dataframe["kc_percent"] = (
            (dataframe["close"] - dataframe["kc_lowerband"]) /
            (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        )
        dataframe["kc_width"] = (
            (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        )
        dataframe['kc_diff'] = (dataframe['kc_upperband'] - dataframe['close'])

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']

        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        long position when close is above 200 SMA and close is above upper band

        """
        conditions = []

        if self.buy_sar_enabled.value:
            conditions.append(dataframe['sar'].notnull())
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_sma_enabled.value:
            conditions.append(dataframe['sma'].notnull())
            conditions.append(dataframe['close'] > dataframe['sma'])

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

        conditions.append(
            (dataframe['close'] >= dataframe['kc_upperband']) &
            (dataframe['close'].shift(1) < dataframe['kc_upperband'].shift(1))
        )

        if conditions:
            dataframe.loc[reduce(lambda x, y: x & y, conditions), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column

        close if price is below Keltner lower band
        """

        if self.sell_hold.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0
            return dataframe

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['kc_lowerband'])
            ),
            'sell'] = 1
        return dataframe