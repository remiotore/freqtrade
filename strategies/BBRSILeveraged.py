



import numpy as np
import pandas as pd
from freqtrade.persistence import Trade
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
from functools import reduce

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)


import talib.abstract as ta
from technical import qtpylib
import re


class BBRSILeveraged(IStrategy):
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
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """


    INTERFACE_VERSION = 3

    timeframe = '1h'

    can_short: bool = False

















    buy_ema = IntParameter(150, 200, default=200, space="buy", optimize=False)
    buy_above_ema_window = IntParameter(5, 15, default=13, space="buy",
                                        optimize=True)
    buy_bb_window = IntParameter(14, 21, default=17, space="buy", optimize=True)
    buy_bb_std = DecimalParameter(2.0, 3.0, default=2.0, decimals=1,
                                  optimize=True)
    sell_rsi_bullish = IntParameter(65, 90, default=65, space="sell",
                                    optimize=True)
    sell_rsi_bearish = IntParameter(10, 35, default=35, space="sell",
                                    optimize=True)
    sell_unclog = IntParameter(5, 14, default=6, space="sell", optimize=True)
    sell_rr_ratio = DecimalParameter(1.5, 3.0, default=1.5, decimals=1,
                                     space="sell", optimize=True)


    stoploss = -0.10


    minimal_roi = {
        "0": sell_rr_ratio.value * abs(stoploss)
    }

    trailing_stop = True




    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count: int = int(
        max(
            buy_ema.value,
            buy_above_ema_window.value,
            buy_bb_window.value
        )
    )

    order_types = {
        'entry': 'limit',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    
    @property
    def plot_config(self):
        return {

            'main_plot': {



            },
            'subplots': {

                "RSI_INF": {
                    f'rsi_{self.timeframe}': {}
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

        pairs = self.dp.current_whitelist()

        stake_currency = self.config['stake_currency']

        informative_pairs = [
            (
                f"{re.findall(f'.*[^(UP|DOWN)/{stake_currency}]', pair)[0]}/{stake_currency}",
                self.timeframe
            ) for pair in pairs
        ]
        return informative_pairs

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

        quote_currency = self.config['stake_currency']
        base_currency = re.findall(f'.*[^(UP|DOWN)/{quote_currency}]', metadata['pair'])[0]

        informative = self.dp.get_pair_dataframe(pair=f"{base_currency}/{quote_currency}",
                                                 timeframe=self.timeframe)

        informative['rsi'] = ta.RSI(informative)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(informative),
            window=self.buy_bb_window.value,
            stds=self.buy_bb_std.value)
        informative['bb_lowerband'] = bollinger['lower']
        informative['bb_upperband'] = bollinger['upper']

        informative[f'ema{self.buy_ema.value}'] = ta.EMA(informative, timeperiod=self.buy_ema.value)






        dataframe = merge_informative_pair(
            dataframe,
            informative,
            self.timeframe,
            self.timeframe,
            ffill=True
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        quote_currency = self.config['stake_currency']
        base_currency = re.findall(f'.*[^(UP|DOWN)/{quote_currency}]', metadata['pair'])[0]


        conditions_bullish = []
        if metadata['pair'] == f"{base_currency}UP/{quote_currency}":


            for i in range(self.buy_above_ema_window.value-1, -1, -1):
                conditions_bullish.append(
                    dataframe[[f'open_{self.timeframe}', f'close_{self.timeframe}']].shift(i).min(axis=1) >=
                    dataframe[f'ema{self.buy_ema.value}_{self.timeframe}'].shift(i)
                )

            conditions_bullish.append(
                qtpylib.crossed_below(dataframe[f'close_{self.timeframe}'], dataframe[f'bb_lowerband_{self.timeframe}'])
            )

            conditions_bullish.append((dataframe['volume'] > 0))

            if conditions_bullish:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions_bullish),
                    'enter_long'] = 1


        conditions_bearish = []
        if metadata['pair'] == f"{base_currency}DOWN/{quote_currency}":


            for i in range(self.buy_above_ema_window.value - 1, -1, -1):
                conditions_bearish.append(
                    dataframe[[f'open_{self.timeframe}',
                               f'close_{self.timeframe}']].shift(i).max(axis=1) <=
                    dataframe[f'ema{self.buy_ema.value}_{self.timeframe}'].shift(i)
                )

            conditions_bearish.append(
                qtpylib.crossed_above(dataframe[f'close_{self.timeframe}'],
                                      dataframe[f'bb_upperband_{self.timeframe}'])
            )

            conditions_bearish.append((dataframe['volume'] > 0))

            if conditions_bearish:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions_bearish),
                    'enter_long'] = 1


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        quote_currency = self.config['stake_currency']
        base_currency = re.findall(f'.*[^(UP|DOWN)/{quote_currency}]',
                                   metadata['pair'])[0]

        if metadata['pair'] == f"{base_currency}UP/{quote_currency}":
            dataframe.loc[
                (

                    (qtpylib.crossed_above(dataframe[f'rsi_{self.timeframe}'], self.sell_rsi_bullish.value)) &

                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'exit_long'] = 1

        if metadata['pair'] == f"{base_currency}DOWN/{quote_currency}":
            dataframe.loc[
                (

                    (qtpylib.crossed_below(dataframe[f'rsi_{self.timeframe}'], self.sell_rsi_bearish.value)) &

                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'exit_long'] = 1

        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Union[str, bool]]:


        if (
                ((current_time - trade.open_date_utc).days >= self.sell_unclog.value)
        ):
            return 'unclog'
