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
    freqtrade backtesting    --config /freqtrade/user_data/config.json --timerange 20240801-20240806 --strategy EMA_Fibonacci_v2
    freqtrade plot-dataframe --config /freqtrade/user_data/config.json --timerange 20240801-20240806 --strategy EMA_Fibonacci_v2
'''

class EMA_Fibonacci_V3(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.1

    minimal_roi = {}

    stoploss = -0.99

    trailing_stop = False

    max_open_trades = -1

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
                'typ' : { 'color': '#000000' },
                '30'  : { 'color' : 'red' },
                '60'  : { 'color' : 'red' },
            },
            'subplots' : {
                'Directional Indicator' : {

                    'pdi_34'  : { 'color': '#6AFF6A' }, 
                    'pdi_55'  : { 'color': '#72FF72' }, 
                    'pdi_89'  : { 'color': '#7AFF7A' }, 
                    'pdi_144' : { 'color': '#82FF82' }, 
                    'pdi_233' : { 'color': '#8AFF8A' }, 

                    'mdi_34'  : { 'color': '#6A6AFF' }, 
                    'mdi_55'  : { 'color': '#7272FF' }, 
                    'mdi_89'  : { 'color': '#7A7AFF' }, 
                    'mdi_144' : { 'color': '#8282FF' }, 
                    'mdi_233' : { 'color': '#8A8AFF' }, 
                },
            },
        }

        return plot_config
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag:str, side: str, **kwargs) -> float:
        return 10.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['typ'] = qtpylib.typical_price(dataframe)
        dataframe['30'] = ta.EMA(dataframe['typ'], 30)
        dataframe['60'] = ta.EMA(dataframe['typ'], 60)

        dataframe['pdi_34']  = ta.PLUS_DI(dataframe, 34)
        dataframe['pdi_55']  = ta.PLUS_DI(dataframe, 55)
        dataframe['pdi_89']  = ta.PLUS_DI(dataframe, 89)
        dataframe['pdi_144'] = ta.PLUS_DI(dataframe, 144)
        dataframe['pdi_233'] = ta.PLUS_DI(dataframe, 233)

        dataframe['mdi_34']  = ta.MINUS_DI(dataframe, 34)
        dataframe['mdi_55']  = ta.MINUS_DI(dataframe, 55)
        dataframe['mdi_89']  = ta.MINUS_DI(dataframe, 89)
        dataframe['mdi_144'] = ta.MINUS_DI(dataframe, 144)
        dataframe['mdi_233'] = ta.MINUS_DI(dataframe, 233)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['pdi_34']  > dataframe['pdi_55'])  &
                (dataframe['pdi_55']  > dataframe['pdi_89'])  &
                (dataframe['pdi_89']  > dataframe['pdi_144']) &
                (dataframe['pdi_144'] > dataframe['pdi_233']) &
                (dataframe['typ']     > dataframe['30'])
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['mdi_34']  > dataframe['mdi_55'])  &
                (dataframe['mdi_55']  > dataframe['mdi_89'])  &
                (dataframe['mdi_89']  > dataframe['mdi_144']) &
                (dataframe['mdi_144'] > dataframe['mdi_233']) &
                (dataframe['30']      > dataframe['typ'])
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['30'] < dataframe['60'])
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe['30'] > dataframe['60'])
            ),
        'exit_short'] = 1

        return dataframe