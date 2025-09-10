# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter

# author @tirail

ma_types = {
        'SMA': ta.SMA,
        'EMA': ta.EMA,
        }

class SMAOffsetBZed(IStrategy):
    INTERFACE_VERSION = 2

    # Buy hyperspace params:
    # buy_params = {
    #     "buy_dip_candles_1": 2,
    #     "buy_dip_candles_2": 12,
    #     "buy_dip_candles_3": 130,
    #     "buy_dip_threshold_1": 0.12,
    #     "buy_dip_threshold_2": 0.28,
    #     "buy_dip_threshold_3": 0.36,
    #     "base_nb_candles_buy": 19,  # value loaded from strategy
    #     "buy_trigger": "SMA",  # value loaded from strategy
    #     "low_offset": 0.969,  # value loaded from strategy
    # }

    # # Sell hyperspace params:
    # sell_params = {
    #     "base_nb_candles_sell": 30,  # value loaded from strategy
    #     "high_offset": 1.012,  # value loaded from strategy
    #     "sell_trigger": "EMA",  # value loaded from strategy
    # }
    # Buy hyperspace params:
    buy_params = {
        "buy_dip_candles_1": 3,
        "buy_dip_candles_2": 29,
        "buy_dip_candles_3": 130,
        "buy_dip_threshold_1": 0.13,
        "buy_dip_threshold_2": 0.2,
        "buy_dip_threshold_3": 0.25,
        "base_nb_candles_buy": 19,  # value loaded from strategy
        "buy_trigger": "SMA",  # value loaded from strategy
        "low_offset": 0.969,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 30,  # value loaded from strategy
        "high_offset": 1.012,  # value loaded from strategy
        "sell_trigger": "EMA",  # value loaded from strategy
    }



    # Stoploss:
    stoploss = -0.23

    # ROI table:
    minimal_roi = {
            "0": 1,
            }

    inf_1h = '1h' # informative tf

    optimize_dip = False
    optimize_non_dip = True
    
    base_nb_candles_buy = IntParameter(5, 80, default=30, space='buy', optimize=optimize_non_dip, load=True)
    low_offset = DecimalParameter(0.8, 0.99, default=0.958, space='buy', optimize=optimize_non_dip, load=True)

    buy_dip_threshold_1 = DecimalParameter(0.08, 0.2, default=0.12, space='buy', decimals=2, optimize=optimize_dip, load=True)
    buy_dip_threshold_2 = DecimalParameter(0.02, 0.5, default=0.28, space='buy', decimals=2, optimize=optimize_dip, load=True)
    buy_dip_threshold_3 = DecimalParameter(0.02, 0.5, default=0.28, space='buy', decimals=2, optimize=optimize_dip, load=True)
    buy_dip_candles_1 = IntParameter(1, 20, default=2,  space='buy', optimize=optimize_dip, load=True)
    buy_dip_candles_2 = IntParameter(1, 40, default=10, space='buy', optimize=optimize_dip, load=True)
    buy_dip_candles_3 = IntParameter(40, 140, default=132, space='buy', optimize=optimize_dip, load=True)

    # sell params
    base_nb_candles_sell = IntParameter(5, 80, default=30, load=True, optimize=True, space='sell')
    high_offset = DecimalParameter(0.8, 1.1, default=1.012, load=True, optimize=True, space='sell')

    # they are fine as they are.
    sell_trigger = CategoricalParameter(ma_types.keys(), default='EMA', space='sell', load=True, optimize=False)
    buy_trigger = CategoricalParameter(ma_types.keys(), default='SMA', space='buy', optimize=False, load=True)

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False

    process_only_new_candles = True
    startup_candle_count = 200

    plot_config = {
            'main_plot': {
                'ma_offset_buy': {'color': 'orange'},
                'ma_offset_sell': {'color': 'orange'},
                },
            }

    use_custom_stoploss = False

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_1h) for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)
        # EMA

        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        return informative_1h

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float, current_profit: float, **kwargs) -> float:
        return 1

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if not self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_buy'] = ma_types[self.buy_trigger.value](dataframe, int(self.base_nb_candles_buy.value)) * self.low_offset.value
            dataframe['ma_offset_sell'] = ma_types[self.sell_trigger.value](dataframe, int(self.base_nb_candles_sell.value)) * self.high_offset.value
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_buy'] = ma_types[self.buy_trigger.value](dataframe, int(self.base_nb_candles_buy.value)) * self.low_offset.value

        dataframe.loc[
                (
                    (dataframe['close'] > dataframe['ema_200']) &
                    (dataframe['close'] > dataframe['ema_200_1h']) &
                    (dataframe['ema_50_1h'] > dataframe['ema_100_1h']) &
                    (dataframe['ema_50_1h'] > dataframe['ema_200_1h']) &

                    (((dataframe['open'].rolling(int(self.buy_dip_candles_1.value)).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_1.value) &
                    (((dataframe['open'].rolling(int(self.buy_dip_candles_1.value + self.buy_dip_candles_2.value)).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_2.value) &
                    (((dataframe['open'].rolling(int(self.buy_dip_candles_1.value + self.buy_dip_candles_2.value + self.buy_dip_candles_3.value)).max() - dataframe['close']) / dataframe['close']) < self.buy_dip_threshold_3.value) &

                    (dataframe['close'] < dataframe['ma_offset_buy']) &
                    (dataframe['volume'] > 0)
                    ),
                'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config['runmode'].value == 'hyperopt':
            dataframe['ma_offset_sell'] = ma_types[self.sell_trigger.value](dataframe, int(self.base_nb_candles_sell.value)) * self.high_offset.value

        dataframe.loc[
                (
                    (dataframe['close'] > dataframe['ma_offset_sell']) &
                    (dataframe['volume'] > 0)
                    ),
                'sell'] = 1
        return dataframe
