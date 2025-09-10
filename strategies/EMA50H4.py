
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame, Series
from datetime import datetime, timedelta
import talib.abstract as ta

import numpy as np

from freqtrade.rpc import RPCMessageType
from freqtrade.utils.trades_manager import TradeManager

from freqtrade.utils.tradingview import generate_tv_url
from freqtrade.utils.binance_rest_api import get_ongoing_candle

from typing import List, Tuple, Dict

import logging

logger = logging.getLogger(__name__)


class EMA50H4(IStrategy):
    minimal_roi = {
        "0": 10
    }

    stoploss = -0.99

    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema50"] = ta.EMA(dataframe, 50)
        dataframe["distance_low_ema50"] = (
            (dataframe["low"] - dataframe["ema50"]) / dataframe["ema50"] * 100
        )

        dataframe["buy_criteria"] = (
            (dataframe["close"].shift(1) > dataframe["ema50"].shift(1)) &
            (dataframe["distance_low_ema50"] > 0) &
            (dataframe["distance_low_ema50"] <= 1)
        )
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe["buy_criteria"]
            ), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
            ),
            'sell'] = 1
        return dataframe
