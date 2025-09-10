
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa
from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter

import Config


class Patterns2(IStrategy):
    """
    Trades based on detection of the 3 White Soldiers candlestick pattern

    How to use it?
    > python3 ./freqtrade/main.py -s Patterns2
    """

    buy_params = {
        "buy_bb_enabled": False,
        "buy_bb_gain": 0.01,
        "buy_mfi": 19.0,
        "buy_mfi_enabled": True,
        "buy_rsi": 4.0,
        "buy_rsi_enabled": False,
        "buy_sma_enabled": True,
    }
    pattern_strength = 90
    buy_rsi = DecimalParameter(1, 50, decimals=0, default=31, space="buy")
    buy_mfi = DecimalParameter(1, 50, decimals=0, default=50, space="buy")
    buy_bb_gain = DecimalParameter(0.01, 0.10, decimals=2, default=0.02, space="buy")

    buy_bb_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_sma_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_rsi_enabled = CategoricalParameter([True, False], default=True, space="buy")
    buy_mfi_enabled = CategoricalParameter([True, False], default=True, space="buy")
















    buy_CDLHAMMER_enabled = True
    buy_CDLINVERTEDHAMMER_enabled = True
    buy_CDLDRAGONFLYDOJI_enabled = True
    buy_CDLPIERCING_enabled = True
    buy_CDLMORNINGSTAR_enabled = False
    buy_CDL3WHITESOLDIERS_enabled = False

    buy_CDL3LINESTRIKE_enabled = False
    buy_CDLSPINNINGTOP_enabled = False
    buy_CDLENGULFING_enabled = False
    buy_CDLHARAMI_enabled = False
    buy_CDL3OUTSIDE_enabled = False
    buy_CDL3INSIDE_enabled = True

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





        dataframe['rsi'] = ta.RSI(dataframe)




        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['lower']
        dataframe["bb_gain"] = ((dataframe["bb_upperband"] - dataframe["close"]) / dataframe["close"])









        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)



        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)

        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)

        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)

        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]

        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]

        dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]



        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)

        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)

        dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)

        dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)

        dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)

        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)




        dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)

        dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]

        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]

        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]

        dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]

        dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]


        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        gconditions = []

        if self.buy_rsi_enabled.value:
            gconditions.append(
                (dataframe['rsi'] <= self.buy_rsi.value) &
                (dataframe['rsi'] > 0)
            )

        if self.buy_sma_enabled.value:
            gconditions.append(dataframe['close'] < dataframe['sma'])

        if self.buy_mfi_enabled.value:
            gconditions.append(dataframe['mfi'] <= self.buy_mfi.value)

        if self.buy_bb_enabled.value:
            gconditions.append(dataframe['bb_gain'] >= self.buy_bb_gain.value)

        tconditions = []

        if self.buy_CDL3WHITESOLDIERS_enabled:
            tconditions.append(dataframe['CDL3WHITESOLDIERS'] >= self.pattern_strength)
        if self.buy_CDLMORNINGSTAR_enabled:
            tconditions.append(dataframe['CDLMORNINGSTAR'] >= self.pattern_strength)
        if self.buy_CDL3LINESTRIKE_enabled:
            tconditions.append(dataframe['CDL3LINESTRIKE'] >= self.pattern_strength)
        if self.buy_CDL3OUTSIDE_enabled:
            tconditions.append(dataframe['CDL3OUTSIDE'] >= self.pattern_strength)

        if self.buy_CDLHAMMER_enabled:
            tconditions.append(dataframe['CDLHAMMER'] >= self.pattern_strength)
        if self.buy_CDLINVERTEDHAMMER_enabled:
            tconditions.append(dataframe['CDLINVERTEDHAMMER'] >= self.pattern_strength)
        if self.buy_CDLDRAGONFLYDOJI_enabled:
            tconditions.append(dataframe['CDLDRAGONFLYDOJI'] >= self.pattern_strength)
        if self.buy_CDLPIERCING_enabled:
            tconditions.append(dataframe['CDLPIERCING'] >= self.pattern_strength)

        if self.buy_CDLSPINNINGTOP_enabled:
            tconditions.append(dataframe['CDLSPINNINGTOP'] >= self.pattern_strength)
        if self.buy_CDLENGULFING_enabled:
            tconditions.append(dataframe['CDLENGULFING'] >= self.pattern_strength)
        if self.buy_CDLHARAMI_enabled:
            tconditions.append(dataframe['CDLHARAMI'] >= self.pattern_strength)
        if self.buy_CDL3INSIDE_enabled:
            tconditions.append(dataframe['CDL3INSIDE'] >= self.pattern_strength)

        gr = False
        pr = False
        if gconditions:
            gr = reduce(lambda x, y: x & y, gconditions)
        if tconditions:
            pr = reduce(lambda x, y: x | y, tconditions)

        dataframe.loc[(gr & pr), 'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        conditions = []

        dataframe.loc[(dataframe['close'].notnull() ), 'sell'] = 0

        return dataframe