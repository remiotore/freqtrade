# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import imp
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import ta as ta
from ta.trend import KSTIndicator
import talib.abstract as taa
import freqtrade.vendor.qtpylib.indicators as qtpylib


class Trend23t(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.15,        
        # "30": 0.04,        
        # "60": 0.08,        

    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.045

    # Trailing stoploss
    trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
 
    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'kc_upperband': {'color': 'blue'},
            'kc_lowerband': {'color': 'blue'},           
            'kc_middleband': {'color': 'blue'},           
        },
        'subplots': {          
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},  
            },                     
            "Plus DI": {
                'plus_di': {'color': 'green'},
                'dil': {'color': 'black'},  
            },       

            "KST": {
                'kst': {'color': 'blue'},
                'kst_sig': {'color': 'orange'},

            }  
                  
        }
    }


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Momentum Indicators
        # ------------------------------------

        kst = ta.KSTIndicator(dataframe)
        dataframe['kst'] = kst['kst']
        dataframe['kst_diff'] = kst['kst_diff']
        dataframe['kst_sig'] = kst['kst_sig']


        # MACD
        macd = taa.MACD(dataframe, fastperiod=8, slowperiod=18, signalperiod=13)

        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        macds = taa.MACD(dataframe, fastperiod=8, slowperiod=18, signalperiod=13)

        dataframe['macds'] = macds['macd']
        dataframe['macdsignals'] = macds['macdsignal']
        dataframe['macdhists'] = macds['macdhist']

        # # Ultimate Oscillator
        dataframe['uo'] = taa.ULTOSC(dataframe)

        # Plus Directional Indicator / Movement
        dataframe['plus_dm'] = taa.PLUS_DM(dataframe)
        dataframe['plus_di'] = taa.PLUS_DI(dataframe)
        dataframe['dil'] = 25

        # RSI
        dataframe['rsi'] = taa.RSI(dataframe)
        dataframe['rsil'] = 20      

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        dataframe['iftrsil'] = 0.5

        # # Keltner Channel
        keltner = qtpylib.keltner_channel(dataframe, window=16, atrs=2)
        dataframe['kc_upperband'] = keltner['upper']
        dataframe['kc_lowerband'] = keltner['lower']
        dataframe['kc_middleband'] = keltner['mid']

        # # Awesome Oscillator
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        dataframe['aol'] = 0
        

        # # # # SMA - Simple Moving Average
        dataframe['sma7'] = taa.SMA(dataframe, timeperiod=7)
        dataframe['sma20'] = taa.SMA(dataframe, timeperiod=20)
        dataframe['sma25'] = taa.SMA(dataframe, timeperiod=25)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # Triggers
                (dataframe['macd'] > 0) &                 
                (dataframe['plus_di'] < 35) &              
                (dataframe['plus_di'] > 26) &                        
                (dataframe['fisher_rsi'] > 0.8) &
                (dataframe['open'] > dataframe['kc_middleband']) &
                (dataframe['close'] > dataframe['kc_middleband']) &

                # Actual Trigger
                (qtpylib.crossed_above(dataframe['sma7'], dataframe['sma20'])) 
                (qtpylib.crossed_above(dataframe['kst'], dataframe['kst_sig'])) 

                (dataframe['volume'] > 0)  # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Low below KC')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                # MACD below MACDsignal
                (qtpylib.crossed_below(dataframe['macds'], dataframe['macdsignals'])) &              
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            
            ),

            'exit_long'] = 1
        return dataframe
