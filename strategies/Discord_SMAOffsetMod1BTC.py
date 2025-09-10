# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open


# PLEAS CHANGE THIS SETTINGS WITH BACKTEST
# base_nb_candles = 30 #something higher than 1
low_offset = 0.958 # something lower than 1
high_offset = 1.012 # something higher than 1


class SMAOffsetMod1BTC(IStrategy):
    INTERFACE_VERSION = 2
    # ROI table:
    minimal_roi = {
        "0": 0.16,
    }

    # Stoploss:
    stoploss = -0.12

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.12
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 60

    # Optional order type mapping.
    order_types = {
        'buy': 'market',
        'sell': 'market',
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
    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [("BTC/USDT", "5m")
                            ]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.timeframe)

        # SMA
        informative['sma_35'] = ta.SMA(informative, timeperiod=35)
        informative['sma_5'] = ta.SMA(informative, timeperiod=5)

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, ffill=True)
        
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['sma_2'] = ta.SMA(dataframe, timeperiod=2)
#        dataframe['sma_35'] = ta.SMA(dataframe, timeperiod=35)
#        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] < (dataframe['sma_30'] * low_offset)) 
                &
                (dataframe['sma_2'] < dataframe['close'])
                &
                (dataframe['sma_3'] < dataframe['sma_2'])
                &
                (dataframe['sma_35'] < dataframe['sma_5'])
                &
                (dataframe['volume'] > 40000)
            ),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['close'] > (dataframe['sma_30'] * high_offset)) 
                &
                (dataframe['sma_2'] > dataframe['close'])
                &
                (dataframe['sma_3'] > dataframe['sma_2'])
                &
                (dataframe['volume'] > 40000)
            ),
            'sell'] = 1
        return dataframe