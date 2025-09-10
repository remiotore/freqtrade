# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from typing import Optional, Tuple

# --------------------------------
# Add your lib to import here
import pandas_ta as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime


class stochrsi(IStrategy):
    INTERFACE_VERSION = 2


    minimal_roi = {
        "0": 0.05
    }

    custom_trade_info = {}

    stoploss = -0.05

    # Trailing stoploss
    trailing_stop = False
 
    timeframe = '1h'

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
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.ta.stochrsi(append=True)
        dataframe.ta.ema(length=50, append=True)
        dataframe.ta.ema(length=15, append=True)
        dataframe.ta.ema(length=8, append=True)
        dataframe['ATR'] = ta.atr(high=dataframe['high'], close=dataframe['close'], low=dataframe['low'])

        self.custom_trade_info[metadata['pair']] = self.populate_trades(metadata['pair'])

        if self.dp.runmode.value in ('backtest', 'hyperopt'):
            self.custom_trade_info[metadata['pair']]['ATR'] = dataframe[['date', 'ATR']].copy().set_index(
                'date')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        if self.config['runmode'].value in ('live', 'dry_run'):
            dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            atr = dataframe['ATR'].iat[-1]
        else:
            atr = self.custom_trade_info[trade.pair]['ATR'].loc[current_time]['ATR']

        return atr * 3

    def populate_trades(self, pair: str) -> dict:
        # Initialize the trades dict if it doesn't exist, persist it otherwise
        if not pair in self.custom_trade_info:
            self.custom_trade_info[pair] = {}

        # init the temp dicts and set the trade stuff to false
        trade_data = {'active_trade': False}

        if self.config['runmode'].value in ('live', 'dry_run'):
            active_trade = Trade.get_trades([Trade.pair == pair, Trade.is_open.is_(True), ]).all()

            if active_trade:
                # get current price and update the min/max rate
                current_rate = self.get_current_price(pair, True)
                active_trade[0].adjust_min_max_rates(current_rate)

        return trade_data

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

        if self.custom_trade_info and trade and trade.pair in self.custom_trade_info:
            if self.config['runmode'].value in ('live', 'dry_run'):
                dataframe, last_updated = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
                ATR = dataframe['ATR'].iat[-1]
            else:
                ATR = self.custom_trade_info[trade.pair]['ATR'].loc[current_time]['ATR']

            min_roi = table_roi

            if ATR:
                min_roi = ATR * 2

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
