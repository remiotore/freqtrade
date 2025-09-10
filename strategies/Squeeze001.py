
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

from user_data.strategies import Config



class Squeeze001(IStrategy):
    """
    Strategy based on LazyBear Squeeze Momentum Indicator (on TradingView.com)

    How to use it?
    > python3 ./freqtrade/main.py -s Squeeze001
    """


    buy_params = {
        "buy_adx": 11.0,
        "buy_adx_enabled": False,
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.02,
        "buy_ema_enabled": True,
        "buy_period": 36,
        "buy_sqz_band": 0.004,
    }
    buy_period = IntParameter(3, 50, default=36, space="buy")
    buy_adx = DecimalParameter(10, 50, decimals=0, default=11, space="buy")
    buy_sqz_band = DecimalParameter(0.002, 0.02, decimals=4, default=0.004, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.02, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")

    sell_hold_enabled = CategoricalParameter([True, False], default=True, space="sell")

    startup_candle_count = max(buy_period.value, 20)

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

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.buy_period.value)

        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_period.value)
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['dm_plus'] = ta.PLUS_DM(dataframe)
        dataframe['dm_minus'] = ta.MINUS_DM(dataframe)

        dataframe['sar'] = ta.SAR(dataframe)

        dataframe['rsi'] = ta.RSI(dataframe)

        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)

        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])

        keltner = qtpylib.keltner_channel(dataframe)
        dataframe["kc_upper"] = keltner["upper"]
        dataframe["kc_lower"] = keltner["lower"]
        dataframe["kc_middle"] = keltner["mid"]

        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_period.value)
        dataframe['dc_mid'] = ta.EMA(((dataframe['dc_upper'] + dataframe['dc_lower']) / 2),
                                     timeperiod=self.buy_period.value)




        dataframe['sqz_upper'] = (dataframe['bb_upperband'] - dataframe["kc_upper"])
        dataframe['sqz_lower'] = (dataframe['bb_lowerband'] - dataframe["kc_lower"])
        dataframe['sqz_on'] = ((dataframe['sqz_upper'] < 0) & (dataframe['sqz_lower'] > 0))
        dataframe['sqz_off'] = ((dataframe['sqz_upper'] > 0) & (dataframe['sqz_lower'] < 0))



        dataframe['sqz_ave'] = ta.EMA(((dataframe['dc_mid'] + dataframe['ema']) / 2),
                                      timeperiod=self.buy_period.value)
        dataframe['sqz_delta'] = ta.EMA((dataframe['close'] - dataframe['sqz_ave']),
                                      timeperiod=self.buy_period.value)
        dataframe['sqz_val'] = ta.LINEARREG(dataframe['sqz_delta'], timeperiod=self.buy_period.value)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []


        conditions.append(dataframe['volume'] > 0)

        conditions.append(dataframe['sqz_upper'].notnull())

        if self.buy_adx_enabled.value:
            conditions.append(
                (dataframe['adx'] > self.buy_adx.value)


            )



        if self.buy_ema_enabled.value:
            conditions.append(dataframe['close'] < dataframe['ema'])


        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)


        conditions.append(
            (dataframe['sqz_val'] < -self.buy_sqz_band.value) &
            (dataframe['sqz_val'] > dataframe['sqz_val'].shift(1)) &
            (dataframe['sqz_val'].shift(1) <= dataframe['sqz_val'].shift(2))


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

        if self.sell_hold_enabled.value:
            dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        else:

            conditions = []


            conditions.append(dataframe['volume'] > 0)

            conditions.append(dataframe['sqz_upper'].notnull())






            conditions.append(
                (dataframe['sqz_val'] > self.buy_sqz_band.value) &
                (dataframe['sqz_val'] < dataframe['sqz_val'].shift(1)) &
                (dataframe['sqz_val'].shift(1) >= dataframe['sqz_val'].shift(2))


            )

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

        return dataframe