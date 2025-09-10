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
    freqtrade backtesting    --config /freqtrade/user_data/config.json --timerange 20240801-20240806 --strategy EMA_Fibonacci_v3
    freqtrade plot-dataframe --config /freqtrade/user_data/config.json --timerange 20240801-20240806 --strategy EMA_Fibonacci_v3
'''

class EMA_Fibonacci_V4(IStrategy):
    INTERFACE_VERSION = 3
    
    timeframe = '5m'
    can_short = True
    use_exit_signal = True
    exit_profit_only = True
    exit_profit_offset = 0.5

    minimal_roi = {}

    stoploss = -0.99

    trailing_stop = False

    max_open_trades = -1

    @property
    def plot_config(self):
        plot_config = {
            'main_plot' : {
                'typ'     : { 'color': 'black' },
                'ema_34'  : { 'color': '#FF6A6A' },
                'ema_55'  : { 'color': '#FF7272' },
                'ema_89'  : { 'color': '#FF7A7A' },
                'ema_144' : { 'color': '#FF8282' },
                'ema_233' : { 'color': '#FF8A8A' },
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
        dataframe['ema_34']  = ta.EMA(dataframe['typ'], 34)
        dataframe['ema_55']  = ta.EMA(dataframe['typ'], 55)
        dataframe['ema_89']  = ta.EMA(dataframe['typ'], 89)
        dataframe['ema_144'] = ta.EMA(dataframe['typ'], 144)
        dataframe['ema_233'] = ta.EMA(dataframe['typ'], 233)

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
                (dataframe['pdi_233'] > dataframe['mdi_233']) &
                (dataframe['mdi_233'] > dataframe['mdi_144']) &
                (dataframe['mdi_144'] > dataframe['mdi_89'])  &
                (dataframe['mdi_89']  > dataframe['mdi_55'])  &
                (dataframe['mdi_55']  > dataframe['mdi_34'])
            ),
        'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe['mdi_34']  > dataframe['mdi_55'])  &
                (dataframe['mdi_55']  > dataframe['mdi_89'])  &
                (dataframe['mdi_89']  > dataframe['mdi_144']) &
                (dataframe['mdi_144'] > dataframe['mdi_233']) &
                (dataframe['mdi_233'] > dataframe['pdi_233']) &
                (dataframe['pdi_233'] > dataframe['pdi_144']) &
                (dataframe['pdi_144'] > dataframe['pdi_89'])  &
                (dataframe['pdi_89']  > dataframe['pdi_55'])  &
                (dataframe['pdi_55']  > dataframe['pdi_34'])
            ),
        'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['typ'], dataframe['ema_233']) |
                qtpylib.crossed_below(dataframe['typ'], dataframe['ema_233'])
            ),
        'exit_long'] = 1

        dataframe.loc[
            (
                qtpylib.crossed_above(dataframe['typ'], dataframe['ema_233']) |
                qtpylib.crossed_below(dataframe['typ'], dataframe['ema_233'])
            ),
        'exit_short'] = 1

        return dataframe