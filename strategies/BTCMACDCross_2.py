


from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy.strategy_helper import merge_informative_pair


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa



from user_data.strategies import Config


class BTCMACDCross_2(IStrategy):
    """
    Triggers buys based on MACD crossing of BTC plus some indicators for the current pair
    """

    buy_params = {
        "buy_adx": 1.0,
        "buy_adx_enabled": False,
        "buy_bb_enabled": True,
        "buy_bb_gain": 0.04,
        "buy_dm_enabled": True,
        "buy_fisher": 0.18,
        "buy_fisher_enabled": True,
        "buy_mfi": 79.0,
        "buy_mfi_enabled": False,
        "buy_neg_macd_enabled": True,
        "buy_period": 16,
        "buy_sar_enabled": False,
    }


    buy_mfi = DecimalParameter(10, 100, decimals=0, default=79, space="buy")
    buy_adx = DecimalParameter(1, 99, decimals=0, default=1, space="buy")
    buy_fisher = DecimalParameter(-1, 1, decimals=2, default=0.18, space="buy")

    buy_period = IntParameter(3, 20, default=16, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.04, space="buy")
    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")

    buy_neg_macd_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_adx_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_dm_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_sar_enabled = CategoricalParameter([True, False], default=False, space="buy")
    buy_fisher_enabled = CategoricalParameter([True, False], default=True, space="buy")

    sell_hold = CategoricalParameter([True, False], default=True, space="sell")
    sell_pos_macd_enabled = CategoricalParameter([True, False], default=True, space="sell")

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
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """



        if not self.dp:

            return dataframe

        inf_tf = '5m'
        btc_dataframe = self.dp.get_pair_dataframe(pair="BTC/USD", timeframe=inf_tf)

        btc_macd = ta.MACD(btc_dataframe)
        dataframe['btc_macd'] = btc_macd['macd']
        dataframe['btc_macdsignal'] = btc_macd['macdsignal']
        dataframe['btc_macdhist'] = btc_macd['macdhist']

        dataframe = merge_informative_pair(dataframe, btc_dataframe, self.timeframe, "5m", ffill=True)



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

        if self.buy_sar_enabled.value:
            conditions.append(dataframe['close'] < dataframe['sar'])

        if self.buy_fisher_enabled.value:
            conditions.append(dataframe['fisher_rsi'] < self.buy_fisher.value)

        if self.buy_neg_macd_enabled.value:
            conditions.append(dataframe['macd'] < 0.0)

        if self.buy_bb_enabled.value:
            conditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)

        conditions.append(qtpylib.crossed_above(dataframe['btc_macd'], dataframe['btc_macdsignal']))

        conditions.append(dataframe['volume'] > 0)

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

            if self.sell_pos_macd_enabled:
                conditions.append((dataframe['macd'] > 0.0))

            conditions.append(qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']))

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
    