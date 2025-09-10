# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# --- Do not remove these libs ---
from typing import Optional

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import sys
import logging

from datetime import datetime
from freqtrade.persistence import Trade


class MyTestStrategy(IStrategy):
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

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    unfilledtimeout = {
        'buy': 8640,
        'sell': 8640
    }

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        #"60": 0.01,
        #"30": 0.02,
        #"0": 0.04
        "0": 10
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    #stoploss = -0.10
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(60, 90, default=70, space="sell")

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

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.logger = self._create_logger()

        self.custom_sell_cnt = 0
        self.confirm_trade_entry_cnt = 0
        self.confirm_trade_exit_cnt = 0
        self.custom_entry_price_cnt = 0
        self.custom_exit_price_cnt = 0
        self.entry_price = None
        self.exit_price = None

    def _create_logger(self):
        logger = logging.getLogger(__name__)
        fmt = "%(asctime)s %(filename)s line %(lineno)s in %(funcName)s() - %(levelname)s - %(message)s"
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, datefmt)
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.setLevel(logging.DEBUG)
        return logger

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
                }
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
        
        dataframe['df_index'] = 0

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        df_lenth = len(dataframe.to_dict()['open'])
        for index in range(df_lenth):
            dataframe.iloc[index, dataframe.columns.get_loc('df_index')] = index
            if index % 2 == 0:
                dataframe.iloc[index, dataframe.columns.get_loc('buy')] = 1

        """
        dataframe.loc[
            (
                (dataframe['volume'] > 0)
            ),
            'buy'] = 1
        for index in range(df_lenth):
            if index % 2 == 1:
                dataframe.iloc[index, dataframe.columns.get_loc('buy')] = 0
        """

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        return dataframe

    def custom_entry_price(self, pair: str, current_time: datetime, proposed_rate: float,
                           entry_tag: Optional[str], **kwargs) -> float:
        self.custom_entry_price_cnt += 1

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        self.logger.debug(f"===== {current_time} {current_candle['df_index']} -- {self.custom_entry_price_cnt} =====")

        self.entry_price = proposed_rate * 0.99
        self.logger.debug(f"===== {current_time} set entry_price={self.entry_price}, proposed_rate={proposed_rate} =====")
        return self.entry_price

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            **kwargs) -> bool:
        self.confirm_trade_entry_cnt += 1

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        self.logger.debug(f"===== {current_time} {current_candle['df_index']} -- {self.confirm_trade_entry_cnt} =====")

        if self.entry_price is not None and self.dp and self.wallets:
            free_btc = self.wallets.get_free('BTC')
            free_usdt = self.wallets.get_free('USDT')
            self.logger.debug(f"===== {current_time} confirm_trade_entry return True, entry_price={self.entry_price}, rate={rate}, "
                              f"free_btc={free_btc}, free_usdt={free_usdt} =====")
            self.entry_price = None
            return True

        return False

    def custom_sell(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs):
        self.custom_sell_cnt += 1

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        self.logger.debug(f"===== {current_time} {current_candle['df_index']} -- {self.custom_sell_cnt} =====")
        return "my_custom_sell"

    def custom_exit_price(self, pair: str, trade: Trade,
                          current_time: datetime, proposed_rate: float,
                          current_profit: float, **kwargs) -> float:
        self.custom_exit_price_cnt += 1

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        self.logger.debug(f"===== {current_time} {current_candle['df_index']} -- {self.custom_exit_price_cnt} =====")

        #self.exit_price = proposed_rate * 1.00
        self.exit_price = proposed_rate * 1.01
        self.logger.debug(f"===== {current_time} set exit_price={self.exit_price}, proposed_rate={proposed_rate} =====")
        return self.exit_price

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        self.confirm_trade_exit_cnt += 1

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        self.logger.debug(f"===== {current_time} {current_candle['df_index']} -- {self.confirm_trade_exit_cnt} =====")

        if sell_reason != "my_custom_sell":
            self.logger.error("===== sell reason is not custom_sell!!! =====")
            return False

        if self.exit_price is not None and self.dp and self.wallets:
            free_btc = self.wallets.get_free('BTC')
            free_usdt = self.wallets.get_free('USDT')
            self.logger.debug(f"===== {current_time} confirm_trade_exit return True, exit_price={self.exit_price}, rate={rate}, "
                              f"free_btc={free_btc}, free_usdt={free_usdt} =====")
            self.exit_price = None
            return True

        return False
