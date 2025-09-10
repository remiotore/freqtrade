# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------
import pandas as pd
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
import technical.indicators as ftt
from finta import TA



SMA = 'SMA'
EMA = 'EMA'

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 71,
    "ewo_high": 2.724,
    "ewo_low": -18.377,
    "low_offset": 0.929,
    "buy_trigger": "SMA",  # value loaded from strategy
    "fast_ewo": 50,  # value loaded from strategy
    "rsi_buy": 50,  # value loaded from strategy
    "slow_ewo": 200,  # value loaded from strategy
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 45,
    "high_offset": 1.004,
    "sell_trigger": "SMA",  # value loaded from strategy
}


def EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df, timeperiod=ema_length)
    ema2 = ta.EMA(df, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['close'] * 100
    return emadif



class WhenLambo(IStrategy):
    INTERFACE_VERSION = 2



    inf_1h = '1h' # informative tf
    # ROI table:
    minimal_roi = {
        "0": 0.01
    }

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_1h) for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        
        
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h['ha_open'] = heikinashi['open']
        informative_1h['ha_close'] = heikinashi['close']
        informative_1h['ha_high'] = heikinashi['high']
        informative_1h['ha_low'] = heikinashi['low']

        WMA_ha9 = ta.WMA(informative_1h['ha_close'], timeperiod=9)
        informative_1h['WMA_ha9'] = WMA_ha9
        WMA_ha13 = ta.WMA(informative_1h['ha_close'], timeperiod=13)
        informative_1h['WMA_ha13'] = WMA_ha13

        macd = ta.MACD(informative_1h, fastperiod=100, slowperiod=200, signalperiod=9)
        informative_1h['macd'] = macd['macd']
        informative_1h['macdsignal'] = macd['macdsignal']
        HMA_9=TA.HMA(heikinashi, 9)
        HMA_13=TA.HMA(heikinashi, 13)
        informative_1h['HMA_9'] = HMA_9
        informative_1h['HMA_13'] = HMA_13
        return informative_1h

    # Stoploss:
    stoploss = -0.5

    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)
    buy_trigger = CategoricalParameter(
        [SMA, EMA], default=buy_params['buy_trigger'], space='buy', optimize=False)
    sell_trigger = CategoricalParameter(
        [SMA, EMA], default=sell_params['sell_trigger'], space='sell', optimize=False)

    # Protection
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    fast_ewo = IntParameter(
        10, 50, default=buy_params['fast_ewo'], space='buy', optimize=False)
    slow_ewo = IntParameter(
        100, 200, default=buy_params['slow_ewo'], space='buy', optimize=False)
    rsi_buy = IntParameter(30, 70, default=50, space='buy', optimize=False, load=True)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_sell_signal = True
    sell_profit_only = True
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    # Optimal timeframe for the strategy
    timeframe = '5m'
    informative_timeframe = '1h'

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

    # def get_informative_indicators(self, metadata: dict):

    #     dataframe = self.dp.get_pair_dataframe(
    #         pair=metadata['pair'], timeframe=self.informative_timeframe)

    #     dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
    #     dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

    #     return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # informative = self.get_informative_indicators(metadata)
        # dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.informative_timeframe,
        #                                    ffill=True)
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        # SMAOffset
        if self.buy_trigger.value == 'EMA':
            dataframe['ma_buy'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_buy.value)
        else:
            dataframe['ma_buy'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_buy.value)

        if self.sell_trigger.value == 'EMA':
            dataframe['ma_sell'] = ta.EMA(dataframe, timeperiod=self.base_nb_candles_sell.value)
        else:
            dataframe['ma_sell'] = ta.SMA(dataframe, timeperiod=self.base_nb_candles_sell.value)

        dataframe['ma_offset_buy'] = dataframe['ma_buy'] * self.low_offset.value
        dataframe['ma_offset_sell'] = dataframe['ma_sell'] * self.high_offset.value

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo.value, self.slow_ewo.value)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        
        HMA_9_5M=TA.HMA(heikinashi, 9)
        HMA_13_5M=TA.HMA(heikinashi, 13)
        dataframe['HMA_9_5M'] = HMA_9_5M
        dataframe['HMA_13_5M'] = HMA_13_5M

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions.append(
            (
                (dataframe['macd_1h'] > dataframe['macdsignal_1h'])&
                (dataframe['HMA_9_1h'] > dataframe['HMA_13_1h']) &
                qtpylib.crossed_above(dataframe['HMA_9_5M'], dataframe['HMA_13_5M']) &
                (dataframe['rsi'] < 60)&
                (dataframe['volume'] > 0)
            )
        )


        conditions.append(
            (
                (dataframe['close'] < dataframe['ma_offset_buy']) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                (dataframe['close'] < dataframe['ma_offset_buy']) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ]=1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(
            (
                qtpylib.crossed_below(dataframe['HMA_9_5M'], dataframe['HMA_13_5M']) &
                (dataframe['volume'] > 0)
            )
        )
        conditions.append(
            (
                (dataframe['macd_1h'] < dataframe['macdsignal_1h'])&
                (dataframe['HMA_9_1h'] < dataframe['HMA_13_1h']) & 
                (dataframe['volume'] > 0)
            )
        )
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ]=1

        return dataframe
