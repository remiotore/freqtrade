
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


def calculate_distance_percentage(current_price: float, green_line_price: float) -> float:
    distance = abs(current_price - green_line_price)
    return distance * 100 / current_price


def calculate_percentage_change(start_value: float, final_value: float) -> float:
    return (final_value - start_value) / start_value * 100


def get_symbol_from_pair(pair: str) -> str:
    return pair.split('/')[0]


class DNSClosed(IStrategy):
    minimal_roi = {
        "0": 0.02
    }

    stoploss = -0.01

    timeframe = '1h'

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        buy_criteria = [False]
        strong_price_movement_rsi_threshold = 25
        strong_price_movement_candles_lookahead_count = 4
        for i in range(1, len(dataframe)):
            short_df = dataframe.head(i)
            short_df = short_df.tail(500)
            short_df["bull_engulf_red_line"] = self.calculate_bull_engulf_red_line(
                dataframe=short_df).tolist()
            bull_engulf_red_line = short_df["bull_engulf_red_line"].tolist()
            previous_range = short_df["open"].shift(1) - short_df["close"].shift(1)
            bull_engulf_green_line = self.calculate_bull_engulf_green_line(
                bull_engulf_red_line=short_df["bull_engulf_red_line"],
                previous_range=previous_range, dataframe=short_df,
                strong_price_movement_rsi_threshold=strong_price_movement_rsi_threshold,
                strong_price_movement_candles_lookahead_count=strong_price_movement_candles_lookahead_count). \
                tolist()

            short_df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(short_df)
            last_open = short_df["open"].iloc[-1]
            last_close = short_df["close"].iloc[-1]
            last_low = short_df["low"].iloc[-1]
            should_buy = False
            for j in range(0, len(bull_engulf_green_line)):
                if not short_df['CDLSPINNINGTOP'].iloc[-1] and \
                        (last_close > last_open) and \
                        (bull_engulf_red_line[j] < last_low < bull_engulf_green_line[j]) and \
                        (last_close > bull_engulf_green_line[j]):
                    should_buy = True
            buy_criteria.append(should_buy)







        dataframe["buy_criteria"] = buy_criteria
        return dataframe

    def calculate_bull_engulf_red_line(self, dataframe: DataFrame) -> Series:
        low = dataframe["low"]
        return np.where(low < low.shift(1), low, low.shift(1))

    def calculate_bull_engulf_green_line(self, bull_engulf_red_line: Series, previous_range: Series,
                                         dataframe: DataFrame,
                                         strong_price_movement_candles_lookahead_count: int,
                                         strong_price_movement_rsi_threshold: float) -> Series:
        open = dataframe["open"]
        low = dataframe["low"]
        close = dataframe["close"]

        is_bull_engulf = (
                (previous_range > 0) &
                (close > open.shift(1))
        )

        low_list = low.tolist()
        min_low_to_end = []
        for i in range(0, len(low_list)):
            min_low_to_end.append(min(low_list[i:]))
        dataframe["min_low_to_end"] = min_low_to_end

        rsi = ta.RSI(dataframe, timeperiod=14).tolist()
        rsi_percentage_increase = [0.0] * len(rsi)
        for i in range(0, len(rsi) - strong_price_movement_candles_lookahead_count):
            next_rsi_list_from_index = i + 1
            next_rsi_list_to_index = (i + strong_price_movement_candles_lookahead_count + 1)
            next_rsi_list = rsi[next_rsi_list_from_index:next_rsi_list_to_index]
            rsi_percentage_increase[i] = self.calculate_max_rsi_percentage_increase_in_the_next_candles(
                current_rsi=rsi[i], next_rsi_list=next_rsi_list
            )
        dataframe["rsi_percentage_increase"] = rsi_percentage_increase

        return np.where(
            is_bull_engulf &
            (dataframe["min_low_to_end"] >= bull_engulf_red_line) &
            (dataframe["rsi_percentage_increase"].shift(1) >= strong_price_movement_rsi_threshold),
            open.shift(1),
            np.nan
        )

    def calculate_max_rsi_percentage_increase_in_the_next_candles(self, current_rsi: float,
                                                                  next_rsi_list: List[float]) -> float:
        max_rsi_percentage_increase_in_the_next_candles = 0.0
        for next_rsi in next_rsi_list:
            rsi_percentage_increase = calculate_percentage_change(
                start_value=current_rsi, final_value=next_rsi
            )
            if rsi_percentage_increase > max_rsi_percentage_increase_in_the_next_candles:
                max_rsi_percentage_increase_in_the_next_candles = rsi_percentage_increase
        return max_rsi_percentage_increase_in_the_next_candles

    def add_backtest_missing_candles(self, dataframe: DataFrame):
        from datetime import datetime
        import pytz
        utc = pytz.UTC



        dataframe.append(
            {"date": utc.localize(datetime(year=2021, month=5, day=31, minute=0, second=0, microsecond=0)),
             "open": 0,
             "high": 0,
             "low": 0,
             "close": 0,
             "volume": 0}, ignore_index=True)

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
