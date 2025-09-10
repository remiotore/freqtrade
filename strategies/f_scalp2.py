
from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
)


import talib.abstract as ta
import talib
import freqtrade.vendor.qtpylib.indicators as qtpylib

class f_scalp2(IStrategy):

    INTERFACE_VERSION = 3
    timeframe = "15m"
    minimal_roi = {
        "0": 100,
        "5": 100
    }


    stoploss = -0.99
    can_short = True





    
    trailing_stop = False
    trailing_stop_positive = 0.204

    trailing_only_offset_is_reached = False
    process_only_new_candles = True

    startup_candle_count: int = 14

    
    def get_leverage(self, pair: str):
        return 6

    
    def custom_stake_amount(self, pair: str, current_time, current_rate: float,
                        proposed_stake: float, min_stake, max_stake: float,
                        leverage: float, entry_tag, side: str,
                        **kwargs) -> float:
        leverage = self.get_leverage(pair)

        return proposed_stake * leverage

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_short'] = 0
        dataframe['exit_long'] = 0

        dataframe['ema_high'] = ta.EMA(dataframe, timeperiod=10, price='high')
        dataframe['ema_close'] = ta.EMA(dataframe, timeperiod=10, price='close')
        dataframe['ema_low'] = ta.EMA(dataframe, timeperiod=10, price='low')

        dataframe['fastk'], dataframe['fastd'] = talib.STOCH(dataframe['high'], dataframe['low'], dataframe['close'], 
                                                    fastk_period=14, slowk_period=3, slowk_matype=0, 
                                                    slowd_period=3, slowd_matype=0)
        dataframe['adx'] = ta.ADX(dataframe)

        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_middleband'] = bollinger['mid']
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macd_diff'] = abs(dataframe['macd'].shift(1) - dataframe['macdsignal'].shift(1))
        
        dataframe['exit_short'] = ((dataframe['rsi'] < 30) & qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])).astype(int)
        dataframe['exit_long'] = ((dataframe['rsi'] > 70) & qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])).astype(int)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                ((dataframe['rsi'] < 30)) &
                qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']) 

                ), 'enter_long'] = 1

        dataframe.loc[
            (   
                ((dataframe['rsi'] > 70))&
                qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']) 

                ), 'enter_short'] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        
        dataframe.loc[
            (
                ((dataframe['rsi'] < 30)) &
                qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal']) 


            ), 'exit_short'] = 1

        dataframe.loc[
            (   
                ((dataframe['rsi'] > 70)) &
                qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal']) 


            ), 'exit_long'] = 1
        
        return dataframe