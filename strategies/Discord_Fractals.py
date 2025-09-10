# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from functools import reduce
import talib.abstract as ta

from freqtrade.strategy import IStrategy
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Fractals(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        # "60": 0.01,
        # "30": 0.02,
        "0": 100
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }
    plot_config = {
        'main_plot': {
            'ma1': {'color': 'green'},
            'ma2': {'color': 'red'},
        },
        'subplots': {
            "fractal": {},
            "rsi": {},
            "adx": {}
        }
    }

    # Hyperoptable parameters
    fractal_len = IntParameter(low=2, high=12, default=5, space='buy', optimize=True, load=True)
    ma1_len = IntParameter(low=2, high=50, default=10, space='buy', optimize=True, load=True)
    ma2_len = IntParameter(low=50, high=300, default=100, space='buy', optimize=True, load=True)
    adx_len = IntParameter(low=5, high=50, default=14, space='buy', optimize=True, load=True)
    rsi_len = IntParameter(low=5, high=50, default=14, space='buy', optimize=True, load=True)

    def fractals(self, dataframe, period=5):
        high = np.array(dataframe.high)
        low = np.array(dataframe.low)
        l = len(high)
        arr = []
        f = 0
        for i in range(l):
            if i >= (2*period+1):
                if high[i-(period+1)] == max(high[i-2*period-1:i]):
                    f = 1
                elif low[i-(period+1)] == min(low[i-2*period-1:i]):
                    f = -1
                elif f == 1:
                    f = 1
                elif f == -1:
                    f = -1
                else:
                    f = 0
            else:
                f = 0
            arr.append(f)
        return np.array(arr)

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

        # Indicators
        for val in self.fractal_len.range:
            dataframe[f'fractal_{val}'] = self.fractals(dataframe, period=val)
        for val in self.ma1_len.range:
            dataframe[f'ma1_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.ma2_len.range:
            dataframe[f'ma2_{val}'] = ta.EMA(dataframe, timeperiod=val)
        for val in self.rsi_len.range:
            dataframe[f'rsi_{val}'] = ta.RSI(dataframe, timeperiod=val)
        for val in self.adx_len.range:
            dataframe[f'adx_{val}'] = ta.ADX(dataframe, timeperiod=val)
        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        conditions = []
        conditions.append(dataframe[f'ma1_{self.ma1_len.value}']
                          > dataframe[f'ma2_{self.ma2_len.value}'])
        conditions.append(dataframe[f'fractal_{self.fractal_len.value}'] == -1)
        conditions.append(dataframe[f'rsi_{self.rsi_len.value}'] > 30)
        conditions.append(dataframe[f'adx_{self.adx_len.value}'] > 25)
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        conditions = []
        conditions.append((dataframe[f'ma1_{self.ma1_len.value}']
                          < dataframe[f'ma2_{self.ma2_len.value}']) &
                          (dataframe[f'rsi_{self.rsi_len.value}'] > 70))
        conditions.append((dataframe[f'fractal_{self.fractal_len.value}'] == 1) &
                          (dataframe[f'rsi_{self.rsi_len.value}'] > 70))
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'] = 1
        return dataframe
