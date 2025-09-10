import datetime
import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from datetime import datetime
from technical import qtpylib
from freqtrade.strategy import IStrategy, IntParameter
from scipy.stats import linregress
from scipy.signal import argrelextrema
from sklearn.preprocessing import MinMaxScaler


'''
Usage:
    freqtrade download-data  --config /freqtrade/user_data/config.json --timerange 20240801-20240806 --timeframes 5m
    freqtrade backtesting    --config /freqtrade/user_data/config.json --timerange 20240801-20240806 --strategy EMA_Fibbonaci 
    freqtrade plot-dataframe --config /freqtrade/user_data/config.json --timerange 20240801-20240806 --strategy EMA_Fibbonaci
'''

class EMA_Fibbonaci(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    can_short = True
    use_exit_signal = True

    minimal_roi = {}

    stoploss = -0.99

    trailing_stop = False

    max_open_trades = -1

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
                'typ' : {},
                '34': { 'color': '#6A6A6A' }, 
                '55': { 'color': '#727272' }, 
                '89': { 'color': '#7A7A7A' }, 
                '144': { 'color': '#828282' }, 
                '233': { 'color': '#8A8A8A' }, 
            },
            'subplots' : {
            },
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['typ'] = qtpylib.typical_price(dataframe)
        dataframe['34']  = ta.EMA(dataframe['typ'], 34)
        dataframe['55']  = ta.EMA(dataframe['typ'], 55)
        dataframe['89']  = ta.EMA(dataframe['typ'], 89)
        dataframe['144'] = ta.EMA(dataframe['typ'], 144)
        dataframe['233'] = ta.EMA(dataframe['typ'], 233)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['typ'] > dataframe['34'])  &
                (dataframe['typ'] > dataframe['55'])  &
                (dataframe['typ'] > dataframe['89'])  &
                (dataframe['typ'] > dataframe['144']) &
                (dataframe['typ'] > dataframe['233'])
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['typ'] < dataframe['34'])  &
                (dataframe['typ'] < dataframe['55'])  &
                (dataframe['typ'] < dataframe['89'])  &
                (dataframe['typ'] < dataframe['144']) &
                (dataframe['typ'] < dataframe['233'])
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_below(dataframe['typ'], dataframe['34'])
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['typ'], dataframe['34'])
            ),
        'exit_short'] = 1

        return dataframe