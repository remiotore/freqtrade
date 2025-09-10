
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter



class DCBBBounce(IStrategy):
    """
    Simple strategy based on Contrarian Donchian Channels crossing Bollinger Bands

    How to use it?
    > python3 ./freqtrade/main.py -s DCBBBounce.py
    """


    buy_params = {
        "buy_adx": 25.0,
        "buy_adx_enabled": True,
        "buy_ema_enabled": False,
        "buy_period": 52,
        "buy_sar_enabled": True,
        "buy_sma_enabled": False,
    }

    buy_period = IntParameter(10, 120, default=52, space="buy")

    buy_adx = DecimalParameter(1, 99, decimals=0, default=25, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_ema_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_adx_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")

    startup_candle_count = buy_period.value


    if sell_hold.value:

        minimal_roi = {
            "0": 0.278,
            "39": 0.087,
            "124": 0.038,
            "135": 0
        }

        trailing_stop = True
        trailing_stop_positive = 0.172
        trailing_stop_positive_offset = 0.212
        trailing_only_offset_is_reached = False

        stoploss = -0.333
    else:

        minimal_roi = {
            "0": 0.261,
            "40": 0.087,
            "95": 0.023,
            "192": 0
        }

        stoploss = -0.33

        trailing_stop = True
        trailing_stop_positive = 0.168
        trailing_stop_positive_offset = 0.253
        trailing_only_offset_is_reached = False

    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = True

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

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
        dataframe['bb_lowerband'] = bollinger['lower']

        dataframe['dc_upper'] = ta.MAX(dataframe['high'], timeperiod=self.buy_period.value)
        dataframe['dc_lower'] = ta.MIN(dataframe['low'], timeperiod=self.buy_period.value)

        dataframe["dcbb_diff_upper"] = (dataframe["dc_upper"] - dataframe['bb_upperband'])
        dataframe["dcbb_diff_lower"] = (dataframe["dc_lower"] - dataframe['bb_lowerband'])

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




        conditions.append(dataframe['dc_upper'].notnull())

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
                (dataframe['adx'] > self.buy_adx.value) &
                (dataframe['dm_plus'] >= dataframe['dm_minus'])
            )




        conditions.append(
            (dataframe['dcbb_diff_lower'].notnull()) &
            (dataframe['close'] >= dataframe['open']) &
            (qtpylib.crossed_above(dataframe['dcbb_diff_lower'], 0))
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
                (dataframe['dcbb_diff_upper'].notnull()) &

                (qtpylib.crossed_below(dataframe['dcbb_diff_upper'], 0))
            )

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
