



from datetime import datetime
from typing import Optional

import numpy as np  # noqa
import pandas as pd  # noqa


import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import (
    IStrategy, informative, IntParameter, DecimalParameter,
)

class FSampleStrategy_1(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1m"
    informative_timeframe = '1h'

    can_short = True

    trailing_stop = False

    process_only_new_candles = True

    rsi_overbought = IntParameter(70, 90, default=80, space='sell')
    macd_signal_diff = DecimalParameter(0, 0.5, default=0.1, space='sell')

    @property
    def protections(self):
        return [
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 360,
                "trade_limit": 2,
                "stop_duration_candles": 360,
                "required_profit": 0.0,
                "only_per_pair": False,
                "only_per_side": False,
            }
        ]

    def leverage(
            self,
            pair: str,
            current_time: datetime,
            current_rate: float,
            proposed_leverage: float,
            max_leverage: float,
            entry_tag: Optional[str],
            side: str,
            **kwargs
    ) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return 10

    def informative_pairs(self):
        return [("BTC/USDT", self.informative_timeframe)]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=14, stds=2.2)

        dataframe["bb_upperband"] = bollinger["upper"]
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['rsi_min'] = dataframe['rsi'].rolling(window=30, min_periods=1).min()
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    @informative('30m')
    def populate_indicators_inf2(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe, 14)

        return dataframe

    @informative('5m')
    def populate_indicators_inf3(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe, 14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    dataframe["volume"] > 0

            ),
            "enter_long",
        ] = 0

        dataframe.loc[
            (
                    (dataframe["rsi"] > 60)
                    &
                    (dataframe["rsi_30m"] > 50)
                    &
                    (dataframe['close'] < dataframe['bb_upperband'])
                    &
                    (dataframe["adx"] > 40)
                    &
                    (dataframe['macd'] < dataframe['macdsignal'])
            )
            ,
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_condition_rsi = dataframe["rsi"] < 20

        dataframe.loc[
            (
                exit_condition_rsi
            ),
            "exit_short",
        ] = 1

        dataframe.loc[
            (
                    dataframe["volume"] > 0

            ),
            "exit_long",
        ] = 0

        return dataframe

















