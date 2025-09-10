# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

#OnlyProfitHyperOptLoss   
buy_rsi_value    = 60
rsi_buy_trigger  = 30
rolling_mean     = 20
rolling_coef     = 1.02837

sell_rsi_value   = 72
rsi_sell_trigger = 38


class TestCandles11(IStrategy):
  
    INTERFACE_VERSION = 2
    minimal_roi = {
        "0": 0.03026,
        "222": 0.02754,
        "365": 0.02437,
        "606": 0.02165,
        "904": 0.01828,
        "1201": 0.01646,
        "1325": 0.01194,
        "1584": 0.00859,
        "1706": 0.00639,
        "1929": 0.00412,
        "2175": 0
    }

    # Stoploss:
    stoploss = -0.21616

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.09502
    trailing_stop_positive_offset = 0.19301
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
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
 
     
        dataframe['rsi_buy'] = ta.RSI(dataframe, timeperiod=rsi_buy_trigger )
        dataframe['rsi_sell'] = ta.RSI(dataframe, timeperiod=rsi_sell_trigger )
        dataframe['CDLDOJI'] = ta.CDLDOJI(dataframe)
        dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe)
        dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe)
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe)
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
         
                (dataframe['rsi_buy'] < buy_rsi_value) &
                (   (dataframe['CDLDOJI'] == 100) | 
                    (dataframe['CDLHARAMI'] == 100) | 
                    (dataframe['CDLPIERCING'] == 100) |
                    (dataframe['CDLMORNINGSTAR'] == 100)
                ) &
                ((dataframe['close'].rolling(rolling_mean).mean()  / dataframe['close']) > rolling_coef)   &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (  (dataframe['rsi_sell'] > sell_rsi_value) 
                ) &

                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
    