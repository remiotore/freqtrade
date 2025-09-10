


from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame

from freqtrade.strategy.hyper import CategoricalParameter, DecimalParameter, IntParameter


import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa




class EMABreakout_2(IStrategy):
    """
    Simple strategy that trades based on Prices breaking above/below the EMA
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

    buy_period = CategoricalParameter([20, 30, 30, 50, 60, 70, 80, 90, 100], default=90, space="buy")
    buy_macd_enabled = CategoricalParameter([True, False], default=True, space="buy")
    sell_hold = CategoricalParameter([True, False], default=False, space="sell")

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

    timeframe = '5m'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = True
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 30

    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    
    plot_config = {

        'main_plot': {
            'ema': {},
            'sma': {},
            'sar': {'color': 'white'},
        },
        'subplots': {

            "MACD": {
                'macdhist': {'color': 'blue'}
            },
            "RSI": {
                'rsi': {'color': 'red'},
            },
        }
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
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """




































        dataframe['rsi'] = ta.RSI(dataframe)




















        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']







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
























        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.buy_period.value)

        dataframe['sma'] = ta.SMA(dataframe, timeperiod=self.buy_period.value)







        dataframe['sar'] = ta.SAR(dataframe)






























































        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        conditions = []


        conditions.append(dataframe['volume'] > 0)

        if self.buy_macd_enabled.value:
            conditions.append(dataframe['macdhist'] >= 0)


        conditions.append(qtpylib.crossed_above(dataframe['close'], dataframe['ema']))

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

            conditions.append(qtpylib.crossed_below(dataframe['close'], dataframe['ema']))

            if conditions:
                dataframe.loc[reduce(lambda x, y: x & y, conditions), 'sell'] = 1

        return dataframe
    