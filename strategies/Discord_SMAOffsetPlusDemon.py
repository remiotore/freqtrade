# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series, DatetimeIndex, merge
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open


# PLEAS CHANGE THIS SETTINGS WITH BACKTEST
# base_nb_candles = 30 #something higher than 1
low_offset = 0.958 # something lower than 1
high_offset = 1.012 # something higher than 1


class SMAOffsetPlusDemon(IStrategy):
    INTERFACE_VERSION = 2
    # ROI table:
    minimal_roi = {
        "0": 0.035,
        "60": 0.01,
    }

    # Stoploss:
    stoploss = -0.2
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.016
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'
    startup_candle_count = 200
    process_only_new_candles = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 4
        }
    ]

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # Prevent ROI trigger, if there is more potential, in order to maximize profit
        if (sell_reason == 'roi') & (last_candle['rsi'] > 50):
            return False
        return True

    def informative_pairs(self):
        informative_pairs = [("BTC/USDT", "5m"),]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Informative
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe
        inf_tf = '5m'
        informative = self.dp.get_pair_dataframe("BTC/USDT", "5m")
        informative['plus_di'] = ta.PLUS_DI(informative)
        informative['minus_di'] = ta.MINUS_DI(informative)
        informative['cmf'] = self.chaikin_mf(informative)
        informative['sma50'] = ta.SMA(informative, timeperiod=50)
        informative['sma200'] = ta.SMA(informative, timeperiod=200)
        informative['pct'] = informative['close'].pct_change(1)
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # Checks
        dataframe['btc_bear'] = dataframe['sma200_5m'].lt(dataframe['sma200_5m'].shift(10))
        dataframe['btc_sma_delta'] = ( ( (dataframe['sma50_5m'] - dataframe['sma200_5m']) / dataframe['sma50_5m']) * 100)
        dataframe['pct'] = dataframe['close'].pct_change(1)
        dataframe['body_pct'] = ((dataframe['open'] - dataframe['close']) * 100) / ((dataframe['open'] - dataframe['close']) + (dataframe['close'] - dataframe['low']))

        #SMA
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < (dataframe['sma_30'] * low_offset)) &
                (dataframe['volume'] > 0)
                    &
                (dataframe['btc_sma_delta'] > -2) &
                    (
                        (dataframe['cmf_5m'] < -0.25) &
                        (dataframe['minus_di_5m'] > dataframe['plus_di_5m']) &
                        (dataframe['close_5m'] < dataframe['sma50_5m']) &
                            (
                                (dataframe['btc_bear']) &
                                    (
                                        (dataframe['rsi'] < 30) &
                                        (dataframe['pct'] > -0.05) &
                                            (
                                                (dataframe['close'] <= 0.975 * dataframe['bb_lowerband']) &
                                                (dataframe['body_pct'] > 70)
                                            ) |
                                        (dataframe['pct'] < -0.05) &
                                            (
                                                (dataframe['close'] <= 0.70 * dataframe['bb_lowerband'])
                                            ) &
                                        (dataframe['close'] < dataframe['close'].shift(1)) &
                                        (dataframe['volume'] > 0)
                                    )
                            )
                    )
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > (dataframe['sma_30'] * high_offset)) &
                (dataframe['volume'] > 0)
            ),

            'sell'] = 1
        return dataframe

    def chaikin_mf(self, df, periods=20):
        close = df['close']
        low = df['low']
        high = df['high']
        volume = df['volume']

        mfv = ((close - low) - (high - close)) / (high - low)
        mfv = mfv.fillna(0.0)
        mfv *= volume
        cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()

        return Series(cmf, name='cmf')