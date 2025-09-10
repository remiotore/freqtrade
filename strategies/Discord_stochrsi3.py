# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy, merge_informative_pair
from freqtrade.persistence import Trade
from typing import Optional, Tuple

# --------------------------------
# Add your lib to import here
import pandas_ta as ta
import talib.abstract as classicTA
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime


class stochrsi3(IStrategy):
    INTERFACE_VERSION = 2

    minimal_roi = {
        "0": 1
    }

    custom_trade_info = {}

    stoploss = -0.30

    # Trailing stoploss
    trailing_stop = False

    timeframe = '1h'
    informative_timeframe = '1h'

    process_only_new_candles = False

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    startup_candle_count: int = 50

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
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'EMA_8': {'color': 'red'},
            'EMA_50': {'color': 'blue'},
            'EMA_15': {'color': 'purple'},
        },
        'subplots': {
            "ATR": {
                'ATR': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)

        informative.ta.stochrsi(append=True)
        informative.ta.ema(length=50, append=True)
        informative.ta.ema(length=15, append=True)
        informative.ta.ema(length=8, append=True)
        informative['ATR'] = classicTA.ATR(informative, timeperiod=14)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
                                           ffill=True)
        skip_columns = [(s + "_" + self.informative_timeframe) for s in
                        ['date', 'open', 'high', 'low', 'close', 'volume']]
        dataframe.rename(columns=lambda s: s.replace("_{}".format(self.informative_timeframe), "") if (
            not s in skip_columns) else s, inplace=True)

        dataframe['stop'] = (dataframe['ATR'] * 2) / dataframe['close']
        dataframe['stop'].fillna(method='ffill', inplace=True)
        dataframe['roi'] = dataframe['ATR'] / dataframe['close']
        dataframe['roi'].fillna(method='ffill', inplace=True)

        self.custom_trade_info[metadata['pair']] = dataframe[['date', 'stop', 'roi']].copy().set_index(
            'date')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        custom_info_pair = self.custom_info[pair]
        if custom_info_pair is not None:
            open_date = trade.open_date_utc if hasattr(trade, 'open_date_utc') else trade.open_date.replace(
                tzinfo=custom_info_pair.index.tz)
            open_date_mask = custom_info_pair.index.unique().get_loc(open_date, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            if open_df is None or len(open_df) == 0:
                return 1
            else:
                return -open_df['stop']

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                    ((dataframe['EMA_15'].gt(dataframe['EMA_50'])) & (dataframe['EMA_8'].gt(dataframe['EMA_15']))) &
                    (dataframe['close'].gt(dataframe['EMA_8'])) &
                    (qtpylib.crossed_above(dataframe['STOCHRSIk_14_14_3_3'], dataframe['STOCHRSId_14_14_3_3']))

            ),
            'buy'] = 1

        return dataframe

    def min_roi_reached(self, trade: Trade, current_profit: float, current_time: datetime) -> bool:
        trade_dur = int((current_time.timestamp() - trade.open_date_utc.timestamp()) // 60)

        _, roi = self.min_roi_reached_dynamic(trade, current_profit, current_time, trade_dur)

        if roi is None:
            return False
        else:
            return current_profit > roi

    def min_roi_reached_dynamic(self, trade: Trade, current_profit: float, current_time: datetime, trade_dur: int) -> \
            Tuple[Optional[int], Optional[float]]:
        _, table_roi = self.min_roi_reached_entry(trade_dur)

        custom_info_pair = self.custom_trade_info[trade.pair]
        if custom_info_pair is not None:
            open_date = trade.open_date_utc if hasattr(trade, 'open_date_utc') else trade.open_date.replace(
                tzinfo=custom_info_pair.index.tz)
            open_date_mask = custom_info_pair.index.unique().get_loc(open_date, method='ffill')
            open_df = custom_info_pair.iloc[open_date_mask]

            min_roi = table_roi

            if open_df['roi']:
                min_roi = open_df['roi']

        else:
            min_roi = table_roi

        return trade_dur, min_roi

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                ((dataframe['EMA_15'].lt(dataframe['EMA_50'])) | (dataframe['EMA_8'].lt(dataframe['EMA_15'])))
            ),
            'sell'] = 1
        return dataframe
